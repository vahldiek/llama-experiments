from dotenv import load_dotenv
from transformers import LlamaTokenizer, AutoConfig, LlamaForCausalLM, AutoModel, AutoTokenizer
import torch
import logging

#.env file should contain huggingface token for download
#HF_TOKEN="yourtoken"
load_dotenv()
logging.getLogger('llama2_streamlit').setLevel(logging.DEBUG)

#Download the chat version of the model
model_id = "meta-llama/Llama-2-7b-chat-hf"
#model_id = "sentence-transformers/all-mpnet-base-v2"
output_dir = "./model_llama-2-7b-chat-hf"
#output_dir = "./all-mpnet-base-v2"

#Download the non chat version of the model
#model_id = "meta-llama/Llama-2-7b-hf"
#output_dir = "./model_llama-2-7b-hf"

#Download the model and tokenizer
config = AutoConfig.from_pretrained(model_id, torchscript=True)
if not hasattr(config, "text_max_length"):
    config.text_max_length = 64

user_model = AutoModel.from_pretrained(
    model_id, config=config, low_cpu_mem_usage=True)
user_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#now save the model locally
user_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)