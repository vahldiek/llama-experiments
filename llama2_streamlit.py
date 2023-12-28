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
        #reread the configuration file in case any changes were made
        st.session_state.llama_config = llama_utils.read_config()

        #reset the token counting state as well
        #One dimensional list so it can be updated by generate worker thread
        st.session_state.total_prompt_tokens = [0]
        st.session_state.user_tokens_per_query = []
        st.session_state.answer_tokens_per_query = []
        #This will be remeasured when the next user query it sent
        st.session_state.system_prompt_tokens = 0



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


#Update queues tracking the tokens used for each input and prune off early conversation elements if
#conversation has become too long
def rightsize_conversation(input_tensor):
    tokenizer = st.session_state.tokenizer
    conversation = st.session_state.conversation
    have_previous_data = (not conversation.past_user_inputs is None) and (len(conversation.past_user_inputs) > 0)

    #Init the token counting state variables if necessary
    if "total_prompt_tokens" not in st.session_state:
        #This is a single element list so it can be updated from the generate worker thread
        st.session_state.total_prompt_tokens = [0]
        st.session_state.user_tokens_per_query = []
        st.session_state.answer_tokens_per_query = []
    #remeasure the system prompt tokens if necessary
    if ("system_prompt_tokens" not in st.session_state) or (st.session_state.system_prompt_tokens==0):
        full_system_prompt = build_system_prompt(st.session_state.llama_config.llm_system_prompt)
        system_prompt_tokens = tokenizer.encode(full_system_prompt, return_tensors='pt')
        #Determine the length of the system prompt
        st.session_state.system_prompt_tokens = len(system_prompt_tokens[0])
    
    new_total_tokens = len(input_tensor[0])
    #Don't count the system tokens against this user prompt
    tokens_added = new_total_tokens - st.session_state.total_prompt_tokens[0]

    #Append the number of tokens for this query
    st.session_state.user_tokens_per_query.append(tokens_added)

    #If we are under the allowed tokens or if we only have this one request just return the input tensor
    if (new_total_tokens <= st.session_state.llama_config.max_prompt_tokens) or not have_previous_data:
        st.session_state.logger.debug(f"Sending {new_total_tokens} tokens")
        st.session_state.total_prompt_tokens[0] = new_total_tokens
        return input_tensor
    
    st.session_state.logger.info(f"&&&& Request to send {new_total_tokens} tokens, must be pruned &&&&")
    #otherwise, we need to start pruning old elements from the conversation
    #until we get under the limit
    #count the number of request/response pairs that we need to prune
    i=0
    #Use answers for length because user queries should be one larger at this point
    while (new_total_tokens > st.session_state.llama_config.max_prompt_tokens) and (
        i < len(st.session_state.answer_tokens_per_query)):
        round_tokens = (st.session_state.user_tokens_per_query[i] +
                                               st.session_state.answer_tokens_per_query[i])
        new_total_tokens = new_total_tokens - round_tokens
        i = i+1
    st.session_state.logger.debug(f"pruned {i} conversation rounds")

    #Adjust our stored length vectors
    st.session_state.user_tokens_per_query = st.session_state.user_tokens_per_query[i:]
    st.session_state.answer_tokens_per_query = st.session_state.answer_tokens_per_query[i:]

    new_conversation = Conversation()
    trimmed_responses = conversation.generated_responses[i:]
    trimmed_inputs = conversation.past_user_inputs[i:]
    
    #Add the system prompt to the first new input
    trimmed_inputs[0] = build_system_prompt(st.session_state.llama_config.llm_system_prompt) + trimmed_inputs[0]
    #Add the old trimmed inputs and responses to the new conversation
    for input, response in zip(trimmed_inputs, trimmed_responses):
        new_conversation.add_user_input(input)
        new_conversation.append_response(response)
        new_conversation.mark_processed()

    #Add the current request to the conversation
    new_conversation.add_user_input(conversation.new_user_input)

    #update the conversation
    st.session_state.conversation = new_conversation

    #finally build new input ids and return a new tensor
    input_ids = tokenizer._build_conversation_input_ids(new_conversation)
    input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
    new_total_tokens = len(input_tensor[0])
    #Will be updated again once response is received
    st.session_state.total_prompt_tokens[0] = new_total_tokens
    st.session_state.logger.debug("reduced to {} tokens".format(new_total_tokens))
    return input_tensor


#Call generate and count the number of new tokens
def generate_and_count(model, answer_list, total_prompt_tokens, inputs, streamer, max_new_tokens, do_sample, temperature, num_beams):
    generate_kwargs = dict(inputs=inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                        do_sample=do_sample, temperature=temperature, num_beams=num_beams)
    
    output_tensor = model.generate(**generate_kwargs)
    new_total_tokens = len(output_tensor[0])
    answer_len = new_total_tokens - len(inputs[0])

    #Append the length of the answer so we can use it for pruning if necessary later
    answer_list.append(answer_len)
    total_prompt_tokens[0] = new_total_tokens
    return output_tensor



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

    gc_kwargs = dict(model=llm, answer_list=st.session_state.answer_tokens_per_query,
                     total_prompt_tokens=st.session_state.total_prompt_tokens, inputs=input_tensor,
                     streamer=streamer, max_new_tokens=config.max_response_tokens, do_sample=False,
                     temperature=temperature, num_beams=1)
    #Get returned content and count tokens returned
    thread = Thread(target=generate_and_count, kwargs=gc_kwargs)
    thread.start()

    response_string = ""
    #very last text 
    for new_text in streamer:
        response_string += new_text
        message_placeholder.markdown(response_string + "▌")

    message_placeholder.markdown(response_string)
    conversation.append_response(response_string)
    #mark ready to add another user message
    conversation.mark_processed()


#Called each time the user enters something into the chat box and sends it
def get_answer_from_llm(user_input) -> str:

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
        #allow the LLama model class to actually format the prompt
        input_ids = tokenizer._build_conversation_input_ids(conversation)
        input_tensor = tokenizer.encode(input_ids, return_tensors='pt')
        #Update the number of tokens with this current addition and
        #prune the conversation if necessary
        input_tensor = rightsize_conversation(input_tensor)
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Model not initialized")
            return "Model not initialized"

    
    return send_prompt_to_llm(input_tensor)




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
    
    #If RAG is supposed to be configured and it isn't, also return
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
        st.session_state.logger.debug("*******************************************")
        st.session_state.logger.debug(full_query)
        st.session_state.logger.debug("*******************************************")
        if (st.session_state.conversation is None) or (not config.use_conversation_history):
            st.session_state.conversation = Conversation(build_system_prompt(st.session_state.llama_config.llm_system_prompt) + full_query)
            st.session_state.user_messages = [user_input]
        else:
            st.session_state.conversation.add_user_input(user_input)
            st.session_state.user_messages.append(user_input)

        
        answer = get_answer_from_llm(user_input)


# This app will be launched multiple times by streamlit
if __name__ == "__main__":
    main()

