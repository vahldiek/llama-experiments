#This file is intended to be launched by streamlit using the command "streamlit run ./llama2_streamlit.py"
#This code is based upon the sample code in the insightful Medium article by Moto DEI
#The article can be found here:  https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e


import argparse
import time
import json
import pathlib
import os
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import  (LlamaTokenizer, AutoConfig, AutoModelForCausalLM,
                            AutoTokenizer, pipeline, LlamaForCausalLM,
                            Conversation)

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader

from typing import List, Union

import streamlit as st
import intel_extension_for_pytorch as ipex
from parse import *

#This is the model quantized by the Intel tools
quantized_model_path = "./saved_results/model_llama-2-7b-chat-int8.pt"
#Set to None to bypass loading the quantized model
#quantized_model_path = None

#use this version to load model from local directory
original_model_id = "./model_llama-2-7b-chat-hf"

#can use this version to download model again
#original_model_id = "meta-llama/Llama-2-7b-hf"

#Set to True to use 4 bit auto-quanitzed model on GPU
use_GPU=False

system_prompt_content = """You are a helpful AI assistant. Reply in mardkown format.
If anyone asks who you are be sure to tell them that you are running on an Intel Xeon processor."""

additional_context = """The Security and Privacy Research group (SPR) led by Intel Labs Vice President Sridhar Iyengar
is part of Intel Labs in Intel corporation.  It is a collection of some of the most accomplished scientists in the world.
Their achievements in security, CPU architecture, cryptography, confidential computing, and machine learning are truly world leading"""






def init_page() -> None:
    st.set_page_config(
        page_title="Personal LLM"
    )
    st.header("Personal LLM")
    st.sidebar.title("Options")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "conversation" not in st.session_state:
        #In the initial conversation set the system prompt and set additional context, sort of a poor-man's
        #RAG
        st.session_state.conversation = Conversation(build_system_prompt(system_prompt_content) + "set context")
        st.session_state.conversation.append_response(additional_context)
        st.session_state.conversation.mark_processed()



#streamlit will only call this function if the model name parameter value changes
#The @st.cache_resource decoration indcates this to streamlit
@st.cache_resource
def load_llm(model_name : str) -> Union[LlamaForCausalLM]:
    if model_name.startswith("llama-2-"):
        config = AutoConfig.from_pretrained(original_model_id, torchscript=True)
        if not hasattr(config, "text_max_length"):
            config.text_max_length = 64

        st.session_state.tokenizer = LlamaTokenizer.from_pretrained(original_model_id, use_fast=True)

        if(use_GPU):
            inference_device_map="auto"
            use_bitsandbytes_quantization=True
        else:
            inference_device_map=torch.device('cpu')
            use_bitsandbytes_quantization=False

        st.session_state.model = LlamaForCausalLM.from_pretrained(
                            original_model_id, config=config, device_map=inference_device_map,
                            load_in_4bit=use_bitsandbytes_quantization)
            
        #Load the Intel quantized model if not using GPU
        if (not use_GPU) and not (quantized_model_path is None):
            #begin optimization for using Intel quantized model
            torch._C._jit_set_texpr_fuser_enabled(False)
            qconfig = ipex.quantization.default_static_qconfig_mapping
            st.session_state.model = ipex.optimize_transformers(
                st.session_state.model.eval(),
                dtype=torch.float,
                inplace=True,
                quantization_config=qconfig,
                deployment_mode=False,
            )

            #Load the Intel quantized model
            self_jit = torch.jit.load(quantized_model_path)
            self_jit = torch.jit.freeze(self_jit.eval())
            #Not sure exactly what this does.  Swaps in the quantized model for the original?
            ipex._set_optimized_model_for_generation(st.session_state.model, optimized_model=self_jit)

        return st.session_state.model
    else:
        return None


#Select a model using the radio buttons
def select_llm() -> Union[LlamaForCausalLM]:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ["llama-2-7b-int8"])
    #ignoring temperature for now
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    return load_llm(model_name)


#Add appropriate tags around system prompt
def build_system_prompt(prompt_content: str):
    system_begin = "<<SYS>>\n "
    system_end = "\n<</SYS>>\n\n"
    return system_begin + prompt_content + system_end

#Retrieve only the tokens received following what was input to the model
def get_chat_response_tensor(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    input_size = input_tensor.size()

    #return a two dimensional array, should be same as input, which instructs calls to tokenizer.batch_decode
    #to return the full string as the first array element, and an array of tokens as the second element
    return output_tensor[:,input_size[1]:]

#Called each time the user enters something into the chat box and sends it
def get_answer_from_llm() -> str:
    tokenizer = st.session_state.tokenizer
    conversation = st.session_state.conversation
    llm = st.session_state.model
    if isinstance(st.session_state.model, LlamaForCausalLM):
        #allow the LLama model class to actually format the prompt
        input_ids = tokenizer._build_conversation_input_ids(conversation)
        input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
        if(use_GPU):
            input_tensor = input_tensor.to('cuda')
        #Args for generate
        generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)

        #This call exercises the model
        output_tensor = llm.generate(
            input_tensor, max_new_tokens=2000, **generate_kwargs
        )

        #separate the response from the input that was sent to the model
        response_tensor = get_chat_response_tensor(input_tensor, output_tensor)

        gen_text = tokenizer.batch_decode(response_tensor, skip_special_tokens=True)
        #first element returned is the full string, second element is an array of strings
        #one for each token
        response_str = gen_text[0]
        conversation.append_response(response_str)
        #mark ready to add another user message
        conversation.mark_processed()

        return response_str
    else:
        return "Model not initialized"

#The first user message to the model contains the system directive
#but we don't want to display that.
def remove_system_prompt_preamble(input_str: str):
    system_prompt = build_system_prompt(system_prompt_content)
    #If the system prompt is found at the beginning of the string, remove it.
    if input_str.find(system_prompt) == 0:
        len_system_prompt = len(system_prompt)
        return input_str[len_system_prompt:]
    else:
        return input_str



#main function.  This script is launched multiple times by streamlit
def main() -> None:
    _ = load_dotenv()

    init_page()
    llm = select_llm()
    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Enter your question!"):
        if st.session_state.conversation is None:
            st.session_state.conversation = Conversation(build_system_prompt(system_prompt_content) + user_input)
        else:
            st.session_state.conversation.add_user_input(user_input)
        with st.spinner("The LLM is typing ..."):
            answer = get_answer_from_llm()

    # Display chat history
    if not st.session_state.conversation is None:
        # Walk through all of text in the conversation
        num_user_messages = 0
        num_assistant_messages = 0
        for is_user, text in st.session_state.conversation.iter_texts():
            if is_user:
                #Skip the first user message because that just sets the system prompt
                if num_user_messages > 0:
                    with st.chat_message("user"):
                        st.markdown(text)
                num_user_messages = num_user_messages + 1
            else:
                #Skip the first assistant message since that only sets context
                if num_assistant_messages > 0:
                    with st.chat_message("assistant"):
                        st.markdown(text)
                num_assistant_messages = num_assistant_messages + 1


# This app will be launched multiple times by streamlit
if __name__ == "__main__":
    main()

