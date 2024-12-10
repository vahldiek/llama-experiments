import argparse
import time
import os
import timeit
import time
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline,
                            TextStreamer, LlamaTokenizer)

import torch
from typing import List, Union

import toml
import sys
import logging
from ipex_inference_transformers import IpexAutoInferenceTransformer
from typing import Any, Dict, Callable
import intel_extension_for_pytorch as ipex

#Get the beginning and ending system tokens
#If using models other than llama, need to determine where to get these
from  transformers.models.llama import tokenization_llama


DEFAULT_CONFIG_FILE = "./.transformers_config.toml"
logger = logging.getLogger('transformers_streamlit.transformers_utils')


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





def load_optimized_model(transformers_config):

    config = AutoConfig.from_pretrained(transformers_config.llm_model_id, torchscript=True)
    model_id = transformers_config.llm_model_id
    quantized_model_path = transformers_config.quantized_model_path

    if(transformers_config.use_GPU):
        inference_device_map="auto"
        use_bitsandbytes_quantization=True
    else:
        inference_device_map=torch.device('cpu')
        use_bitsandbytes_quantization=False

    #Load either the base or quantized model
    start = time.perf_counter()
    if transformers_config.use_GPU or (quantized_model_path is None):

        original_model = AutoModelForCausalLM.from_pretrained(
                            model_id, config=config,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,)

        torch.cpu.amp.autocast(enabled=True)

        original_model = ipex.optimize(original_model.eval(),
                    dtype=torch.bfloat16,
                    inplace=True
                )

    else:
        original_model = IpexAutoInferenceTransformer.from_ipex_pretrained(model_id, quantized_model_path, config)

    end = time.perf_counter()
    logger.debug(f"Model {model_id} load took {end - start:0.4f} seconds")
    #Load the tokenizer.  Must be a llama tokenizer since it has the _ function to build prompt from conversation
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
    return tokenization_llama.B_SYS + prompt_content + tokenization_llama.E_SYS


#Use this trick with globals to use timeit with function, makes the function certainly not thread safe
g_results = []
g_input_tensor = None
g_model = None
g_tokenizer = None
def send_model_queries(model, tokenizer, query_list, transformers_config, time_queries=True):
    global g_results, g_input_tensor, g_model, g_tokenizer

    g_model = model

    conv = pipeline("conversational")
    g_results = []
    g_input_tensor = None
    #prepend the first request with the system string
    for i in range(len(query_list)):
        #Add the system directive to the first query
        if i == 0:
            conv.add_user_input(_build_system_prompt(transformers_config.llm_system_prompt) + query_list[0])
        else:
            conv.add_user_input(query_list[i])

        logger.debug("Sending query to model: " + query_list[i])
        input_ids = tokenizer._build_conversation_input_ids(conv)
        input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
        if(transformers_config.use_GPU):
            input_tensor = input_tensor.to('cuda')
        #Use this trick to allow timeit to time function that both takes and returns a value

        g_input_tensor = input_tensor
        g_model = model
        g_tokenizer = tokenizer
        logger.debug("***********************************************************************************")
        elapsed_time = timeit.timeit('g_results.append(query_model(g_model, g_tokenizer, g_input_tensor))',
                                                    setup='from transformers_utils import query_model, g_model, g_tokenizer, g_results, g_input_tensor', number=1)
        logger.debug("\n***********************************************************************************")
        if time_queries:
            logger.info(f"Query took {elapsed_time} seconds")

        chatbot_answer_str = g_results[i]
        conv.append_response(chatbot_answer_str)
        conv.mark_processed()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def merge_rag_results(vector_store, query, transformers_config):
    #Only receive results if the similarity is above the threshold
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs={'k': transformers_config.max_rag_documents, 'score_threshold' : transformers_config.rag_relevance_limit})
    docs = retriever.invoke(query)
    #If no local docs met the similarity threshold,
    if(len(docs) == 0):
        logger.debug(f"+++ No relevant RAG docs found for \"{query}\"+++")
        if(transformers_config.always_use_RAG_prompt):
            full_query = transformers_config.rag_prompt_template.format(context="None", question=query)
        else:
            full_query = query
    else:
        logger.debug(f"%%%%% found rag docs for \"{query}\"")
        #merge the results into one string
        context = format_docs(docs)
        full_query = transformers_config.rag_prompt_template.format(context=context, question=query)

    return full_query


#Use this trick with globals to use timeit with function, makes the function certainly not thread safe
g_results = []
g_input_tensor = None
g_model = None
g_tokenizer = None
def send_rag_queries(vector_store, model, tokenizer, query_list, transformers_config, time_queries=True):
    global g_results, g_input_tensor, g_model, g_tokenizer

    g_model = model

    conv = Conversation()
    g_results = []
    g_input_tensor = None

    #prepend the first request with the system string
    for i in range(len(query_list)):
        #If the rag query returned any results, use those.
        query = merge_rag_results(vector_store, query_list[i], transformers_config)

        #Add the system directive to the first query
        if i == 0:
            conv.add_user_input(_build_system_prompt(transformers_config.llm_system_prompt) + query)
        else:
            conv.add_user_input(query)

        logger.debug("Sending query to model: " + query)
        input_ids = tokenizer._build_conversation_input_ids(conv)
        input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
        if(transformers_config.use_GPU):
            input_tensor = input_tensor.to('cuda')
        #Use this trick to allow timeit to time function that both takes and returns a value

        g_input_tensor = input_tensor
        g_model = model
        g_tokenizer = tokenizer
        logger.debug("***********************************************************************************")
        elapsed_time = timeit.timeit('g_results.append(query_model(g_model, g_tokenizer, g_input_tensor))',
                                                    setup='from transformers_utils import query_model, g_model, g_tokenizer, g_results, g_input_tensor', number=1)
        logger.debug("\n***********************************************************************************")
        if time_queries:
            logger.info(f"Query took {elapsed_time} seconds")

        chatbot_answer_str = g_results[i]
        conv.append_response(chatbot_answer_str)
        conv.mark_processed()
