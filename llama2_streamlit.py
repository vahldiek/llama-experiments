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
from transformers import LlamaTokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, LlamaForCausalLM

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader

from typing import List, Union

import streamlit as st
import intel_extension_for_pytorch as ipex
from parse import *

#This is the model quantized by the Intel tools
#quantized_model_path = "./int8.pt"
#Set to None to bypass loading the quantized model
quantized_model_path = None

#use this version to load model from local directory
original_model_id = "./model_llama-2-7b-chat-hf"

#can use this version to download model again
#original_model_id = "meta-llama/Llama-2-7b-hf"

#Set to True to use 4 bit auto-quanitzed model on GPU
use_GPU=False






def init_page() -> None:
    st.set_page_config(
        page_title="Personal LLM"
    )
    st.header("Personal LLM")
    st.sidebar.title("Options")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        #Prime the message stream with a message to the system
        st.session_state.messages = [{"role" : "system", "content" : "You are a helpful AI assistant. Reply in mardkown format."}]


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
#Must have at least two options.  With only one option streamlit appears to
#become confused and list each character as a separate option.
def select_llm() -> Union[LlamaForCausalLM]:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("llama-2-7b-int8",
                                   "llama-2-7b-int8"))
    #ignoring temperature for now
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    return load_llm(model_name)


#Loop through the entire response until we get the last section after the last instruction
#Keep going until we hit the final close inst tag.
def retrive_last_llama2_response(response_str: str):
    remaining_string = response_str
    return_val = None
    while return_val is None:
        items = parse("{instructions}[/INST]{remainder}", remaining_string)
        if items is not None:
            remaining_string = items['remainder']
        else:
            return_val = remaining_string

    #Double check to see if we ended with an instruction and no response
    final_parse = parse("{instructions}[/INST]", return_val)
    if not final_parse is None:
        return None
    return return_val


#Called each time the user enters something into the chat box and sends it
def get_answer() -> tuple[str, float]:
    if isinstance(st.session_state.model, LlamaForCausalLM):
        prompt = llama_v2_prompt_from_messages()


        #Args for generate
        generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
        input_ids = st.session_state.tokenizer(prompt, return_tensors="pt").input_ids
        if(use_GPU):
            input_ids = input_ids.to('cuda')

        #This call begins exercising the model
        output = st.session_state.model.generate(
            input_ids, max_new_tokens=2000, **generate_kwargs
        )
        gen_text = st.session_state.tokenizer.batch_decode(output, skip_special_tokens=True)
        #response is a single string, following all of the data we already sent
        response = retrive_last_llama2_response(gen_text[0])

        if not response is None:
            return response
        else:
            return "The model did not return any results"
    else:
        return "Model not initialized"



#Turn the dictionary used to store prompts and responses into a character string formatted
#to be consumed by llama2
def llama_v2_prompt_from_messages() -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    messages = st.session_state.messages
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    #If the first message in the list is not a system message, prepend the default system message
    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages

    #Combine the system message with the first user message
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]


    #Some very pythonic code to look at each pair of two contiguous list entries, and create a tuple
    #with the 2nd dictionary entry fom each in each of the two items
    # (a bit dangerous, assumes 'content' is the 2nd dictionary entry)
    #Then formats a string to put the first piece of content, the user question inside of the [INST]
    #block and puts the second piece of content, presumably the answer from the assistant, outside
    #of the block and wrap the whole message in <s> blocks.
    #Took me much longer to write this comment than it would have to write a more readable for loop
    #to do the same thing, but this is reused code and maybe it is common in Python.
    messages_list = [
        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]

    #This pythonic code also assumes an odd number of list elements with the last being the most 
    #recent question to the language model.
    messages_list.append(
        f"{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)



#main function.  This script is launched multiple times by streamlit
def main() -> None:
    _ = load_dotenv()

    init_page()
    llm = select_llm()
    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Enter your question!"):
        st.session_state.messages.append({"role" : "user", "content" : user_input})
        with st.spinner("The LLM is typing ..."):
            answer = get_answer()
        st.session_state.messages.append({"role" : "assistant", "content" : answer})

    # Display chat history
    messages = st.session_state.messages
    for message in messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        elif message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])


# This app will be launched multiple times by streamlit
if __name__ == "__main__":
    main()

