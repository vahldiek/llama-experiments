import argparse
import time
import json
import pathlib
import os
import timeit
import time
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import (LlamaTokenizer, LlamaTokenizerFast, AutoConfig, 
                            AutoModelForCausalLM, AutoTokenizer, pipeline,
                            LlamaForCausalLM, LlamaModel, Conversation,
                            TextStreamer)

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from typing import List, Union

from langchain.callbacks import get_openai_callback
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import intel_extension_for_pytorch as ipex
from accelerate import init_empty_weights
import toml
import sys
import logging
from typing import Any, Tuple, Dict, Optional, Callable



DEFAULT_CONFIG_FILE = "./.llama2_config.toml"
logger = logging.getLogger('llama2_streamlit.llama_utils')


class TextCallbackStreamer(TextStreamer):
    """
    Streamer that calls a callback each time a full word is computed.  Useful for tools like
    streamlit to avoid the need to create a worker thread to call generate

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        callback (`Callable[[str, bool], None]`)
        callback_kwargs (`dict`)
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.


    """
    def __init__(
        self, callback : Callable[[str, bool], None], callback_args : Dict[str, Any], tokenizer: "AutoTokenizer", skip_prompt: bool = False,  **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.callback = callback
        self.callback_args = callback_args

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Call the callback"""
        if self.callback_args is not None:
            self.callback(text, stream_end, **self.callback_args)
        else:
            self.callback(text, stream_end)



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def read_config():
    parser = argparse.ArgumentParser(prog=sys.argv[0], add_help=True)
    parser.add_argument(
        "config_file", default=DEFAULT_CONFIG_FILE,
        type=str,
        nargs='?',
        help="Configuration file in JSON format",
    )

    cmd_args = parser.parse_args()
    if(os.path.isfile(cmd_args.config_file)):
        with open(cmd_args.config_file, "r") as f:
            conf = toml.load(f)
            config = dotdict(conf)
    else:
        logger.error("Configuration file required.")
        exit(1)

    #Verify all of the needed configurations are there
    #or set to defaults
    if "llm_model_id" not in config:
        logger.error("missing llm_model_id in conifguration")
        exit(1)
    if "llm_system_prompt" not in config:
        logger.error("missing llm_system_prompt in conifguration")
        exit(1)
    if "quantized_model_path" not in config:
        config['quantized_model_path'] = None
    if "use_GPU" not in config:
        config.use_GPU = False
    if "use_conversation_history" not in config:
        config.use_conversation_history = True
    if "max_response_tokens" not in config:
        config.max_response_tokens = 500
    if "max_prompt_tokens" not in config:
        config.max_prompt_tokens = 1000
    if "rag_file_dir" not in config:
        config.rag_file_dir = None
    if "rag_db_dir" not in config:
        config.rag_db_dir = None
    if "use_RAG" not in config:
        config.use_RAG = False
    if "always_use_RAG_prompt" not in config:
        config.always_use_RAG_prompt = False
    if "reset_RAG_db" not in config:
        config.reset_RAG_db = False
    if "rescan_RAG_files" not in config:
        config.rescan_RAG_files = False
    if "rag_relevance_limit" not in config:
        config.rag_relevance_limit = 0.5
    if "rag_doc_max_chars" not in config:
        config.rag_doc_max_chars = 256
    if "max_rag_documents" not in config:
        config.max_rag_documents = 3

    #check for arg consistency
    if config.use_GPU:
        config.quantized_model_path = None

    if config.use_RAG and ((config.rag_file_dir is None) or (config.rag_db_dir is None)):
        logger.error("When using rag, must set rag_file_dir and rag_db_dir in configuration")
        exit(1)

    return config





def load_optimized_model(llama_config):
    
    config = AutoConfig.from_pretrained(llama_config.llm_model_id, torchscript=True)
    model_id = llama_config.llm_model_id
    quantized_model_path = llama_config.quantized_model_path

    if(llama_config.use_GPU):
        inference_device_map="auto"
        use_bitsandbytes_quantization=True
    else:
        inference_device_map=torch.device('cpu')
        use_bitsandbytes_quantization=False

    #Load the base model.  While we seem to be able to avoid loading the full model when using the Intel
    #quantized version, neither start time nor memory consumption are reduced.  Needs more investigation.
    if quantized_model_path is None:
        start = time.perf_counter()      
        original_model = LlamaForCausalLM.from_pretrained(
                            model_id, config=config,
                                load_in_4bit=use_bitsandbytes_quantization, device_map=inference_device_map)
        end = time.perf_counter()
        logger.debug(f"Base model load took {end - start:0.4f} seconds")
    else:
        start = time.perf_counter()
        num_hidden_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        config.num_hidden_layers = 0
        config.hidden_size = 0
        #Just build a shell model class witn no hidden layers since it will be replaced by the
        #quantized model anyway
        original_model = LlamaForCausalLM(config=config)
        #Need to set the hidden layers config back so the intel optimized greedy_search function
        #can prep the inputs
        original_model.config.num_hidden_layers = num_hidden_layers
        original_model.config.hidden_size = hidden_size
        end = time.perf_counter()
        logger.debug(f"Base model load took {end - start:0.4f} seconds")

    #Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=True, device_map=inference_device_map)

    if not (quantized_model_path is None):
        #begin optimization for using Intel quantized model
        torch._C._jit_set_texpr_fuser_enabled(False)
        qconfig = ipex.quantization.default_static_qconfig_mapping

        #"monkey patches" the original_model object to swap in a few optimized functions
        original_model = ipex.optimize_transformers(
            original_model.eval(),
            dtype=torch.float,
            inplace=True,
            quantization_config=qconfig,
            deployment_mode=False,
        )

        logger.debug("About to load quantized model")
        #Load the Intel quantized model
        start = time.perf_counter()
        self_jit = torch.jit.load(quantized_model_path)
        end = time.perf_counter()
        self_jit = torch.jit.freeze(self_jit.eval())
        logger.debug(f"quantized model load took {end - start:0.4f} seconds")
        #Set self_jit as the optimized model
        ipex._set_optimized_model_for_generation(original_model, optimized_model=self_jit)

    return original_model, tokenizer




def query_model(model, tokenizer, input_tensor):

    #Args for generate
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_size = input_tensor.size()
    num_input_tokens = input_size[1]

    output_tensor = model.generate(
        input_tensor, max_new_tokens=2000, 
        streamer=streamer, **generate_kwargs
    )

    #Collect and return a string containing the entire answer
    #Drop the last token because it is the end of string token and next
    #func is not leaving it out for some reason.
    chatbot_answer_tensor = output_tensor[:,num_input_tokens:]
    chatbot_answer_list = tokenizer.batch_decode(chatbot_answer_tensor, skip_special_tokens=True)
    chatbot_answer_str = chatbot_answer_list[0]
    return chatbot_answer_str


#Add appropriate tags around system prompt
def _build_system_prompt(prompt_content: str):
    system_begin = "<<SYS>>\n "
    system_end = "\n<</SYS>>\n\n"
    return system_begin + prompt_content + system_end


#Use this trick with globals to use timeit with function, makes the function certainly not thread safe
g_results = []
g_input_tensor = None
g_model = None
g_tokenizer = None
def send_model_queries(model, tokenizer, query_list, llama_config, time_queries=True):
    global g_results, g_input_tensor, g_model, g_tokenizer

    g_model = model

    conv = Conversation()
    g_results = []
    g_input_tensor = None
    #prepend the first request with the system string
    for i in range(len(query_list)):
        #Add the system directive to the first query
        if i == 0:
            conv.add_user_input(_build_system_prompt(llama_config.llm_system_prompt) + query_list[0])
        else:
            conv.add_user_input(query_list[i])

        logger.debug("Sending query to model: " + query_list[i])
        input_ids = tokenizer._build_conversation_input_ids(conv)
        input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
        if(llama_config.use_GPU):
            input_tensor = input_tensor.to('cuda')
        #Use this trick to allow timeit to time function that both takes and returns a value

        g_input_tensor = input_tensor
        g_model = model
        g_tokenizer = tokenizer
        logger.debug("***********************************************************************************")
        elapsed_time = timeit.timeit('g_results.append(query_model(g_model, g_tokenizer, g_input_tensor))',
                                                    setup='from llama_utils import query_model, g_model, g_tokenizer, g_results, g_input_tensor', number=1)
        logger.debug("\n***********************************************************************************")
        if time_queries:
            logger.info(f"Query took {elapsed_time} seconds")

        chatbot_answer_str = g_results[i]
        conv.append_response(chatbot_answer_str)
        conv.mark_processed()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def merge_rag_results(vector_store, query, llama_config):
    #Only receive results if the similarity is above the threshold
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs={'k': llama_config.max_rag_documents, 'score_threshold' : llama_config.rag_relevance_limit})
    docs = retriever.invoke(query)
    #If no local docs met the similarity threshold, 
    if(len(docs) == 0):
        logger.debug(f"+++ No relevant RAG docs found for \"{query}\"+++")
        if(llama_config.always_use_RAG_prompt):
            full_query = llama_config.rag_prompt_template.format(context="None", question=query)
        else:
            full_query = query
    else:
        logger.debug(f"%%%%% found rag docs for \"{query}\"")
        #merge the results into one string
        context = format_docs(docs)
        full_query = llama_config.rag_prompt_template.format(context=context, question=query)

    return full_query


#Use this trick with globals to use timeit with function, makes the function certainly not thread safe
g_results = []
g_input_tensor = None
g_model = None
g_tokenizer = None
def send_rag_queries(vector_store, model, tokenizer, query_list, llama_config, time_queries=True):
    global g_results, g_input_tensor, g_model, g_tokenizer

    g_model = model

    conv = Conversation()
    g_results = []
    g_input_tensor = None

    #prepend the first request with the system string
    for i in range(len(query_list)):
        #If the rag query returned any results, use those.
        query = merge_rag_results(vector_store, query_list[i], llama_config)

        #Add the system directive to the first query
        if i == 0:
            conv.add_user_input(_build_system_prompt(llama_config.llm_system_prompt) + query)
        else:
            conv.add_user_input(query)

        logger.debug("Sending query to model: " + query)
        input_ids = tokenizer._build_conversation_input_ids(conv)
        input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
        if(llama_config.use_GPU):
            input_tensor = input_tensor.to('cuda')
        #Use this trick to allow timeit to time function that both takes and returns a value

        g_input_tensor = input_tensor
        g_model = model
        g_tokenizer = tokenizer
        logger.debug("***********************************************************************************")
        elapsed_time = timeit.timeit('g_results.append(query_model(g_model, g_tokenizer, g_input_tensor))',
                                                    setup='from llama_utils import query_model, g_model, g_tokenizer, g_results, g_input_tensor', number=1)
        logger.debug("\n***********************************************************************************")
        if time_queries:
            logger.info(f"Query took {elapsed_time} seconds")

        chatbot_answer_str = g_results[i]
        conv.append_response(chatbot_answer_str)
        conv.mark_processed()
