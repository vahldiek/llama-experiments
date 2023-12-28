import llama_utils
from dotenv import load_dotenv
from transformers import (PreTrainedModel, PreTrainedTokenizer)
import logging

config = llama_utils.read_config()
logging.getLogger('llama2_streamlit').setLevel(logging.DEBUG)

example_context = """The Security and Privacy Research group (SPR) led by Intel Labs Vice President Sridhar Iyengar
is part of Intel Labs in Intel corporation.  It is a collection of some of the most accomplished scientists in the world.
Their achievements in security, CPU architecture, cryptography, confidential computing, and machine learning are truly world leading"""


query_list = ["Hello my name is Anjo",
"who are you?",
"""The following context may help you answer the following question, enhance the context with other information
that you know the context is delimited by three single quotes the question follows.\n
context: '''""" + example_context + "'''\n" + "question:  What can you tell me about SPR?",
"tell me a joke"]


model, tokenizer = llama_utils.load_optimized_model(config)

llama_utils.send_model_queries(model, tokenizer, query_list, config)









