#This file is intended to be launched by streamlit using the command "streamlit run ./transformers_streamlit.py"
#This code originated from the sample code in the insightful Medium article by Moto DEI, but has been expanded substantially.
#The article can be found here:  https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e


import logging
from dotenv import load_dotenv
from threading import Thread

from datasets import load_dataset
from transformers import  (PreTrainedModel, PreTrainedTokenizer)

import torch
from typing import List, Union
import streamlit as st

#Import local modules
import transformers_utils
import chroma_utils
from token_conversation import TokenConversation
from typing import Tuple


logger = logging.getLogger('llama2_streamlit')

def init_page() -> None:
    st.set_page_config(
        page_title=st.session_state.transformers_config.chat_page_name
    )
    st.header(st.session_state.transformers_config.chat_page_name)
    st.sidebar.title("Options")

    st.sidebar.slider("Temperature:", min_value=0.0,
                        max_value=1.0, value=0.5, step=0.01,
                        key="temperature")



def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "transformers_config" not in st.session_state:
        #reread the configuration file in case any changes were made
        st.session_state.transformers_config = transformers_utils.read_config()
    if clear_button or "user_messages" not in st.session_state:
        st.session_state.user_messages = []
    #If the tokenizer has been loaded, reset the conversation
    #otherwise the conversation will be reset once the tokenizer is loaded
    if clear_button and "tokenizer" in st.session_state:
        st.session_state.conversation = TokenConversation(st.session_state.tokenizer,
                                                   st.session_state.transformers_config.llm_system_prompt,
                                                   st.session_state.transformers_config.max_prompt_tokens)



#Call this once.  Don't set session values in here because resources are cached
#accross sessions
@st.cache_resource
def load_llm(model_name : str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if model_name.startswith("llama-2-"):
        model, tokenizer = transformers_utils.load_optimized_model(
                                        st.session_state.transformers_config)
        if(len(st.session_state) == 0):
            logger.warning("************  User pressed stop during model load *******************")
            st.stop()
        return model, tokenizer
    else:
        return None, None


#Select a model using the radio buttons
def select_llm() -> PreTrainedModel:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ["llama-2-7b-chat-int8"])

    return load_llm(model_name)



#Retrieve only the tokens received following what was input to the model
def get_chat_response_tensor(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    input_size = input_tensor.size()

    #return a two dimensional array, should be same as input, which instructs calls to tokenizer.batch_decode
    #to return the full string as the first array element, and an array of tokens as the second element
    return output_tensor[:,input_size[1]:]




#See if the user stopped an LLM response in the middle
#If so, finish out recording the response
def check_for_stopped_response():
    if ("generate_response" in st.session_state) and (
        st.session_state.generate_response is not None
    ):
        st.session_state.logger.info("Found leftover partial string from stopped response, adding")
        st.session_state.conversation.append_partial_response(st.session_state.generate_response)
        st.session_state.generate_response = None

class GenerateStoppedException(Exception):
    def __init__(self, details: str):
        self.details = details

    
#This is called back each time model generate created a word
def model_generate_callback(word: str, is_done: bool) -> None:
    #If user stops generate in the middle, the session state is reset but generate is not
    #actually killed
    if len(st.session_state) == 0:
        exception = GenerateStoppedException("User stopped generate!")
        raise exception

    st.session_state.generate_response += word
    st.session_state.message_placeholder.markdown(st.session_state.generate_response + "▌")


#Send the provided input tensor to the LLM and update
#the conversation history if it needs to be trimmed
def send_prompt_to_llm(input_tensor)  -> str:
    tokenizer = st.session_state.tokenizer
    conversation = st.session_state.conversation
    llm = st.session_state.model
    temperature = st.session_state.temperature
    config = st.session_state.transformers_config
    full_response = None

    if(st.session_state.transformers_config.use_GPU):
        input_tensor = input_tensor.to('cuda')

    with st.chat_message("assistant"):
        msg_window = st.empty()
        st.session_state.message_placeholder = msg_window

        #Set up a streamer to get words one by one once generate is called
        decode_kwargs = dict(skip_special_tokens=True)
        streamer = transformers_utils.TextCallbackStreamer(model_generate_callback, None,
                                                    tokenizer, skip_prompt=True, **decode_kwargs)

        generate_kwargs = dict(inputs=input_tensor, streamer=streamer, max_new_tokens=config.max_response_tokens,
                            do_sample=False, temperature=temperature, num_beams=1)
        
        #indicate that we expect a response from generate in case user stops the
        #response in the middle
        st.session_state.generate_response = ""
        try:
            output_tensor = llm.generate(**generate_kwargs)
        #If user pressed stop during long running generate, session state will be reset, just return
        except GenerateStoppedException as ex:
            print("User stopped generate!!!")
            return None

        full_response = conversation.append_response_from_tokens(output_tensor)
        #indicate that we are done processing the generated text
        st.session_state.generate_response = None
        st.session_state.message_placeholder.markdown(full_response)
        st.session_state.message_placeholder = None

    return full_response


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


    if isinstance(llm, PreTrainedModel):
        #This call does a lot.  Prunes the conversation if necessary to stay under the
        #specified max tokens, and returns the full set of tokesn for the prompt
        input_tensor, rounds_pruned = st.session_state.conversation.create_next_prompt_tokens(full_query)
        #redraw the chat history if conversation trimmed
        #This does not seem optimal either, sometimes streamlit behaves oddly when chat session
        #becomes too large
        if rounds_pruned > 0:
            #Need to prune our list of user messages
            st.session_state.user_messages = st.session_state.user_messages[rounds_pruned:]
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Model not initialized")
            return "Model not initialized"

    
    return send_prompt_to_llm(input_tensor)




#Call this once.  Don't set session values in here because resources are cached
#accross sessions
@st.cache_resource
def init_rag(use_RAG):
    if use_RAG:
            vector_store = chroma_utils.get_vector_store(st.session_state.transformers_config)
    else:
        vector_store = None
    return vector_store


def display_chat_history():
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


def on_new_question():
    user_question = st.session_state.user_question
    #Augment the query with information from the vector store if needed
    if st.session_state.transformers_config.use_RAG:
        full_query = transformers_utils.merge_rag_results(st.session_state.vector_store, user_question, st.session_state.transformers_config)
    else:
        full_query = user_question
    #Track this is separate list since user messages in chat object may have system promp or RAG
    #context appended to them
    st.session_state.user_messages.append(user_question)
    st.session_state.logger.debug("*******************************************")
    st.session_state.logger.debug(full_query)
    st.session_state.logger.debug("*******************************************")
    answer = get_answer_from_llm(full_query, user_question)


#main function.  This script is launched multiple times by streamlit
def main() -> None:
    _ = load_dotenv()
    logger.setLevel(logging.DEBUG)
    logger.debug("streamlit executing main() [again]")
    st.session_state.logger = logger
    if not "transformers_config" in st.session_state:
        st.session_state.transformers_config = transformers_utils.read_config()
        #User can press stop during config file load
        if(len(st.session_state) == 0):
            logger.warning("User pressed stop during config file load")
            st.stop()
        
    config = st.session_state.transformers_config
    init_page()
    st.session_state.vector_store = init_rag(config.use_RAG)
    #User could press stop during this short window too
    if(len(st.session_state) == 0):
        logger.warning("User pressed stop during init_rag")
        st.stop()
    

    #load the default LLM
    st.session_state.model, st.session_state.tokenizer = load_llm("llama-2-7b-chat-int8")
    #If user pressed stop during model load, session state will be
    #wiped out
    if(len(st.session_state) == 0):
        st.stop()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = TokenConversation(st.session_state.tokenizer,
                                                        st.session_state.transformers_config.llm_system_prompt,
                                                        st.session_state.transformers_config.max_prompt_tokens)
    init_messages()
    check_for_stopped_response()


    display_chat_history()

    # Add user input area
    #callbacks don't really seem to be the answer to fixing display issues either
    #st.chat_input("Enter your question!", key="user_question", on_submit=on_new_question)
    if user_question := st.chat_input("Enter your question!"):
        st.session_state.user_question = user_question
        on_new_question()



# This app will be launched multiple times by streamlit
if __name__ == "__main__":
    main()

