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
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from typing import List, Union

from langchain.callbacks import get_openai_callback
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import intel_extension_for_pytorch as ipex

#can use this version to download model again
#original_model_id = "meta-llama/Llama-2-7b-hf"

#Set to True to use 4 bit auto-quanitzed model on GPU
use_GPU=False

#use this version to load model from local directory
original_model_id = "./model_llama-2-7b-chat-hf"

#This is the model quantized by the Intel tools
quantized_model_path = "./int8.pt"


def init_page() -> None:
    st.set_page_config(
        page_title="Personal LLM"
    )
    st.header("Personal LLM")
    st.sidebar.title("Options")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in mardkown format.")
        ]
        st.session_state.costs = []


#streamlit will only call this function if the model name parameter value changes
#The @st.cache_resource decoration indcates this to streamlit
@st.cache_resource
def load_llm(model_name : str) -> HuggingFacePipeline:
    if model_name.startswith("llama-2-"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        config = AutoConfig.from_pretrained(original_model_id, torchscript=True)
        if not hasattr(config, "text_max_length"):
            config.text_max_length = 64


        tokenizer = LlamaTokenizer.from_pretrained(original_model_id, use_fast=True)

        if(use_GPU):
            inference_device_map="auto"
            use_bitsandbytes_quantization=True
        else:
            inference_device_map=torch.device('cpu')
            use_bitsandbytes_quantization=False

        with ipex.OnDevice(dtype=torch.float, device="meta"):
            original_model = LlamaForCausalLM.from_pretrained(
                                original_model_id, config=config, device_map=inference_device_map,
                                load_in_4bit=use_bitsandbytes_quantization)
            
        #Load the Intel quantized model if not using GPU
        if(not use_GPU):
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
            ipex._set_optimized_model_for_generation(original_model, optimized_model="poop")
        

        #Create a Hugging Face pipeline which can then be used by langchain
        if(use_GPU):
            pipe = pipeline("text-generation", model=original_model, tokenizer=tokenizer, max_new_tokens=2000, device_map="auto")
        else:
             pipe = pipeline("text-generation", model=original_model, tokenizer=tokenizer, max_new_tokens=2000, device=-1)
           
        return HuggingFacePipeline(pipeline=pipe)
    else:
        return None

#Select a model using the radio buttons
#Must have at least two options.  With only one option streamlit appears to
#become confused and list each character as a separate option.
def select_llm() -> Union[HuggingFacePipeline]:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("llama-2-7b-int8",
                                   "llama-2-7b-int8"))
    #ignoring temperature for now
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    return load_llm(model_name)

#Called each time the user enters something into the chat box and sends it
def get_answer(llm, messages) -> tuple[str, float]:
    if isinstance(llm, HuggingFacePipeline):
        return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0


#Determine type of role
def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


#Put into dictionary format so streamlit can process
def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


#Turn the dictionary, used by streamlit, into a character string formatted
#to be consumed by llama2
def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


#main function.  This script is launched multiple times by streamlit
def main() -> None:
    _ = load_dotenv()

    init_page()
    llm = select_llm()
    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Enter your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("The LLM is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


# This app will be launched multiple times by streamlit
if __name__ == "__main__":
    main()

