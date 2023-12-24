import argparse
import time
import json
import pathlib
import os
import timeit
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




#use this version to load model from local directory
original_model_id = "./model_llama-2-7b-chat-hf"
#original_model_id = "meta-llama/Llama-2-7b-chat-hf"

#Set quantized_model_path to None to just use the original model
#quantized_model_path = "./saved_results/int8.pt"
quantized_model_path="./saved_results/model_llama-2-7b-chat-int8.pt"
#quantized_model_path=None

if quantized_model_path is None:
    use_GPU = True
else:
    use_GPU = False


DEFAULT_SYSTEM_PROMPT = f"""<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"""
_ = load_dotenv()



config = AutoConfig.from_pretrained(original_model_id, torchscript=True)

if(use_GPU):
    inference_device_map="auto"
    use_bitsandbytes_quantization=True
else:
    inference_device_map=torch.device('cpu')
    use_bitsandbytes_quantization=False

#Load the base model
original_model = LlamaForCausalLM.from_pretrained(
                    original_model_id, config=config,
                        load_in_4bit=use_bitsandbytes_quantization, device_map=inference_device_map)


#Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(original_model_id, use_fast=True, device_map=inference_device_map)

if not (quantized_model_path is None):
    #begin optimization for using Intel quantized model
    torch._C._jit_set_texpr_fuser_enabled(False)
    qconfig = ipex.quantization.default_static_qconfig_mapping
    original_model = ipex.optimize_transformers(
        original_model.eval(),
        dtype=torch.float,
        inplace=True,
        quantization_config=qconfig,
        deployment_mode=False,
    )

    #Load the Intel quantized model
    self_jit = torch.jit.load(quantized_model_path)
    self_jit = torch.jit.freeze(self_jit.eval())
    #Not sure exactly what this does.  Swaps in the quantized model for the original?
    ipex._set_optimized_model_for_generation(original_model, optimized_model=self_jit)




def query_model(input_tensor):

    #Args for generate
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    input_size = input_tensor.size()
    num_input_tokens = input_size[1]

    output_tensor = original_model.generate(
        input_tensor, max_new_tokens=2000, 
        streamer=streamer, **generate_kwargs
    )

    #Collect and return a string containing the entire answer
    chatbot_answer_tensor = output_tensor[:,num_input_tokens:]
    chatbot_answer_list = tokenizer.batch_decode(chatbot_answer_tensor, skip_special_tokens=True)
    chatbot_answer_str = chatbot_answer_list[0]
    return chatbot_answer_str


system_directive = """<<SYS>>\nYou are a helpful AI assistant, alwasy respond in a serious and professional tone.
If anyone asks who you are be sure to tell them that you are running on an Intel Xeon processor and remember to tell
them that you are Llama.\n<</SYS>>\n\n"""


query_list = ["Hello my name is Anjo", "who are you?", "tell me about Intel corproation", "tell me a joke"]


#Below should be in a function, but a bit confusing to figure out how to use timeit internally that way
#prepend the first request with the system string
conv = Conversation()
result = []
input_tensor = None
for i in range(len(query_list)):
    #Add the system directive to the first query
    if i == 0:
        conv.add_user_input(system_directive + query_list[0])
    else:
        conv.add_user_input(query_list[i])

    print("Sending query to model: " + query_list[i])
    input_ids = tokenizer._build_conversation_input_ids(conv)
    input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
    if(use_GPU):
        input_tensor = input_tensor.to('cuda')
    #Use this trick to allow timeit to time function that both takes and returns a value

    print("***********************************************************************************")
    elapsed_time = timeit.timeit('result.append(query_model(input_tensor))',
                                                setup='from __main__ import result, query_model, input_tensor', number=1)
    print("\n***********************************************************************************")
    print(f"Query took {elapsed_time} seconds")
    chatbot_answer_str = result[i]
    conv.append_response(chatbot_answer_str)
    conv.mark_processed()


