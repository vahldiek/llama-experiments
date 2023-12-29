#This file is intended to be launched by streamlit using the command "streamlit run ./llama2_streamlit.py"
#This code is based upon the sample code in the insightful Medium article by Moto DEI
#The article can be found here:  https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e


import argparse
import time
import json
import pathlib
import os
import logging
from dotenv import load_dotenv
from threading import Thread

from datasets import load_dataset
from transformers import  (LlamaTokenizer, AutoConfig, AutoModelForCausalLM,
                            AutoTokenizer, pipeline, LlamaForCausalLM,
                            Conversation, TextIteratorStreamer, PreTrainedModel)

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
from token_conversation import TokenConversation






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
    if clear_button or "llama_config" not in st.session_state:
        #reread the configuration file in case any changes were made
        st.session_state.llama_config = llama_utils.read_config()
    if clear_button or "user_messages" not in st.session_state:
        st.session_state.user_messages = []
    #If the tokenizer has been loaded, reset the conversation
    #otherwise the conversation will be reset once the tokenizer is loaded
    if clear_button and "tokenizer" in st.session_state:
        st.session_state.conversation = TokenConversation(st.session_state.tokenizer,
                                                   st.session_state.llama_config.llm_system_prompt,
                                                   st.session_state.llama_config.max_prompt_tokens)



#streamlit will only call this function if the model name parameter value changes
#The @st.cache_resource decoration indcates this to streamlit
@st.cache_resource
def load_llm(model_name : str) -> Union[LlamaForCausalLM]:
    if model_name.startswith("llama-2-"):
        st.session_state.model, st.session_state.tokenizer = llama_utils.load_optimized_model(
                                                                st.session_state.llama_config)
        if "conversation" in st.session_state:
            st.session_state.conversation.reset_config(st.session_state.tokenizer,
                                                            st.session_state.llama_config.llm_system_prompt,
                                                            st.session_state.llama_config.max_prompt_tokens)
        else:
            st.session_state.conversation = TokenConversation(st.session_state.tokenizer,
                                                            st.session_state.llama_config.llm_system_prompt,
                                                            st.session_state.llama_config.max_prompt_tokens)
        return st.session_state.model
    else:
        return None


#Select a model using the radio buttons
def select_llm() -> Union[LlamaForCausalLM]:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ["llama-2-7b-chat-int8"])

    return load_llm(model_name)



#Retrieve only the tokens received following what was input to the model
def get_chat_response_tensor(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    input_size = input_tensor.size()

    #return a two dimensional array, should be same as input, which instructs calls to tokenizer.batch_decode
    #to return the full string as the first array element, and an array of tokens as the second element
    return output_tensor[:,input_size[1]:]




#Call generate and count the number of new tokens
def generate_and_add(model : PreTrainedModel,
                     conversation : TokenConversation,
                     generate_kwargs) -> str:

    
    output_tensor = model.generate(**generate_kwargs)
    return conversation.append_response_from_tokens(output_tensor)




#Send the provided input tensor to the LLM and update
#the conversation history if it needs to be trimmed
def send_prompt_to_llm(input_tensor)  -> str:
    tokenizer = st.session_state.tokenizer
    conversation = st.session_state.conversation
    llm = st.session_state.model
    temperature = st.session_state.temperature
    config = st.session_state.llama_config

    if(st.session_state.llama_config.use_GPU):
        input_tensor = input_tensor.to('cuda')

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

    #Set up a streamer to get words one by one and do the generate in
    #  another thread.
    decode_kwargs = dict(skip_special_tokens=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, **decode_kwargs)

    generate_kwargs = dict(inputs=input_tensor, streamer=streamer, max_new_tokens=config.max_response_tokens,
                        do_sample=False, temperature=temperature, num_beams=1)
    
    gc_kwargs = dict(model=llm, conversation=conversation, generate_kwargs=generate_kwargs)
    #Get returned content and count tokens returned
    thread = Thread(target=generate_and_add, kwargs=gc_kwargs)
    thread.start()

    response_string = ""
    #very last text 
    for new_text in streamer:
        response_string += new_text
        message_placeholder.markdown(response_string + "▌")

    message_placeholder.markdown(response_string)
    return response_string


#Called each time the user enters something into the chat box and sends it
def get_answer_from_llm(full_query, user_input) -> str:

    response_string = ""

    with st.chat_message("user"):
        st.markdown(user_input)

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
        #This call does a lot.  Prunes the conversation if necessary to stay under the
        #specified max tokens, and returns the full set of tokesn for the prompt
        input_tensor = st.session_state.conversation.create_next_prompt_tokens(full_query)
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Model not initialized")
            return "Model not initialized"

    
    return send_prompt_to_llm(input_tensor)




#Only call this function once unless use_RAG configuration
#changes
@st.cache_resource
def init_rag(use_RAG):
    if use_RAG and ("vector_store" not in st.session_state):
            st.session_state.vector_store = chroma_utils.get_vector_store(st.session_state.llama_config)
    else:
        st.session_state.vector_store = None


#main function.  This script is launched multiple times by streamlit
def main() -> None:
    _ = load_dotenv()
    logger = logging.getLogger('llama2_streamlit')
    logger.setLevel(logging.DEBUG)
    logger.debug("streamlit executing main() [again]")
    st.session_state.logger = logger
    if not "llama_config" in st.session_state:
        st.session_state.llama_config = llama_utils.read_config()

    config = st.session_state.llama_config
    init_page()
    init_rag(config.use_RAG)
    #load the default LLM
    load_llm("llama-2-7b-chat-int8")
    init_messages()

    #Just return if main has been called again and initialization has not
    #completed
    if ("model" not in st.session_state) or (st.session_state.model is None):
        logger.info("Returning early from main because LLM not yet initialized")
        return
    
    #If RAG is supposed to be configured and it isn't yet, also return
    if config.use_RAG and (("vector_store" not in st.session_state) or (st.session_state.vector_store is None)):
        logger.info("Returning early from main because RAG not yet initialized")
        return

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
        #Track this is separate list since user messages in chat object may have system promp or RAG
        #context appended to them
        st.session_state.user_messages.append(user_input)
        st.session_state.logger.debug("*******************************************")
        st.session_state.logger.debug(full_query)
        st.session_state.logger.debug("*******************************************")
        answer = get_answer_from_llm(full_query, user_input)


# This app will be launched multiple times by streamlit
if __name__ == "__main__":
    main()

