#This file is intended to be launched by streamlit using the command "streamlit run ./llama2_streamlit.py"
#This code is based upon the sample code in the insightful Medium article by Moto DEI
#The article can be found here:  https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e


import argparse
import time
import json
import pathlib
import os
from dotenv import load_dotenv
from threading import Thread

from datasets import load_dataset
from transformers import  (LlamaTokenizer, AutoConfig, AutoModelForCausalLM,
                            AutoTokenizer, pipeline, LlamaForCausalLM,
                            Conversation, TextIteratorStreamer)

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader

from typing import List, Union

import streamlit as st
import intel_extension_for_pytorch as ipex
from parse import *
import pathlib
import sys
import json

#Import local modules
import llama_utils
import chroma_utils






def init_page() -> None:
    st.set_page_config(
        page_title=st.session_state.llama_config.chat_page_name
    )
    st.header(st.session_state.llama_config.chat_page_name)
    st.sidebar.title("Options")

    st.sidebar.slider("Temperature:", min_value=0.0,
                        max_value=1.0, value=0.5, step=0.01,
                        key="temperature")



def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "conversation" not in st.session_state:
        #In the initial conversation set the system prompt and set additional context, sort of a poor-man's
        #RAG
        st.session_state.conversation = None
        st.session_state.user_messages = []



#streamlit will only call this function if the model name parameter value changes
#The @st.cache_resource decoration indcates this to streamlit
@st.cache_resource
def load_llm(model_name : str) -> Union[LlamaForCausalLM]:
    if model_name.startswith("llama-2-"):
        st.session_state.model, st.session_state.tokenizer = llama_utils.load_optimized_model(
                                                                st.session_state.llama_config)
        return st.session_state.model
    else:
        return None


#Select a model using the radio buttons
def select_llm() -> Union[LlamaForCausalLM]:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ["llama-2-7b-chat-int8"])

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
def get_answer_from_llm(user_input) -> str:

    response_string = ""

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

    if ("model" not in st.session_state) or (
        "conversation" not in st.session_state) or (
        "tokenizer" not in st.session_state) or (
        st.session_state.tokenizer is None) or (
        st.session_state.conversation is None) or (
        st.session_state.model is None):
        message_placeholder.markdown("Model not initialized")
        return
    
    tokenizer = st.session_state.tokenizer
    conversation = st.session_state.conversation
    llm = st.session_state.model
    temperature = st.session_state.temperature

    if isinstance(llm, LlamaForCausalLM):
        #allow the LLama model class to actually format the prompt
        input_ids = tokenizer._build_conversation_input_ids(conversation)
        input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
        if(st.session_state.llama_config.use_GPU):
            input_tensor = input_tensor.to('cuda')
  
        #Set up a streamer to get words one by one and do the generate in
        #  another thread.
        decode_kwargs = dict(skip_special_tokens=True)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, **decode_kwargs)

        generate_kwargs = dict(inputs=input_tensor, streamer=streamer, max_new_tokens=2000,
                                do_sample=False, temperature=temperature, num_beams=1)
        thread = Thread(target=llm.generate, kwargs=generate_kwargs)
        thread.start()

        #very last text 
        for new_text in streamer:

            response_string += new_text
            message_placeholder.markdown(response_string + "▌")

        message_placeholder.markdown(response_string)
        conversation.append_response(response_string)
        #mark ready to add another user message
        conversation.mark_processed()
    else:
        message_placeholder.markdown("Model not initialized")
    return

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

#Only call this function once
@st.cache_resource
def init_rag():
    if ("vector_store" not in st.session_state):
        if st.session_state.llama_config.use_RAG:
            st.session_state.vector_store = chroma_utils.get_vector_store(st.session_state.llama_config)
        else:
            st.session_state.vector_store = None


#main function.  This script is launched multiple times by streamlit
def main() -> None:
    _ = load_dotenv()

    config = llama_utils.read_config()
    if "llama_config" not in st.session_state:
        st.session_state.llama_config = config
    init_page()
    init_rag()
    llm = select_llm()
    init_messages()

    # Display chat history
    if ("conversation" in st.session_state) and (not st.session_state.conversation is None):
        # Walk through all of text in the conversation
        current_user_message = 0
        for is_user, text in st.session_state.conversation.iter_texts():
            if is_user:
                with st.chat_message("user"):
                    #Use messages from here since full messages may be augmented with
                    #RAG data
                    st.markdown(st.session_state.user_messages[current_user_message])
                    current_user_message = current_user_message + 1
            else:
                with st.chat_message("assistant"):
                    st.markdown(text)


    # Supervise user input
    if user_input := st.chat_input("Enter your question!"):
        #Augment the query with information from the vector store if needed
        if config.use_RAG:
            full_query = llama_utils.merge_rag_results(st.session_state.vector_store, user_input, config)
        else:
            full_query = user_input
        print("*******************************************")
        print(full_query)
        print("*******************************************")
        if st.session_state.conversation is None:
            st.session_state.conversation = Conversation(build_system_prompt(st.session_state.llama_config.llm_system_prompt) + full_query)
            st.session_state.user_messages = [user_input]
        else:
            st.session_state.conversation.add_user_input(user_input)
            st.session_state.user_messages.append(user_input)

        
        answer = get_answer_from_llm(user_input)


# This app will be launched multiple times by streamlit
if __name__ == "__main__":
    main()

