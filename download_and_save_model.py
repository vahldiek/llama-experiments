from dotenv import load_dotenv
from transformers import LlamaTokenizer, AutoConfig, LlamaForCausalLM
import torch

#.env file should contain huggingface token for download
#HF_TOKEN="yourtoken"
load_dotenv()

#Download the chat version of the model
model_id = "meta-llama/Llama-2-7b-chat-hf"
output_dir = "./model_llama-2-7b-chat-hf"

#Download the non chat version of the model
#model_id = "meta-llama/Llama-2-7b-hf"
#output_dir = "./model_llama-2-7b-hf"

#Download the model and tokenizer
config = AutoConfig.from_pretrained(model_id, torchscript=True)
if not hasattr(config, "text_max_length"):
    config.text_max_length = 64

user_model = LlamaForCausalLM.from_pretrained(
    model_id, config=config, low_cpu_mem_usage=True, device_map="auto", load_in_4bit=True)
user_model.eval()
tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=True)

#now save the model locally
user_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)