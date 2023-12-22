import argparse
import time
import json
import pathlib
import os
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import LlamaTokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, LlamaForCausalLM, LlamaModel

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



user_queries = ["tell me a joke", "write a poem about pirates", "what is snow?"]

#use this version to load model from local directory
original_model_id = "./model_llama-2-7b-chat-hf"
#original_model_id = "meta-llama/Llama-2-7b-chat-hf"

#Set quantized_model_path to None to just use the original model
#quantized_model_path = "./int8.pt"
quantized_model_path=None

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


for user_query in user_queries:
    #Add the system prompt to the query
    prompt="<s>[INST]" + DEFAULT_SYSTEM_PROMPT + user_query + "[/INST]</s>"
    print("************************************************")
    print(prompt)

    #Args for generate
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if(use_GPU):
        input_ids = input_ids.to('cuda')

    output = original_model.generate(
        input_ids, max_new_tokens=2000, **generate_kwargs
    )
    gen_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    for answer in gen_text:
        print(answer)
    print("\n\n\n")

