import chromadb
from chromadb.utils import embedding_functions
import torch
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
# use this to configure the Chroma database  
from chromadb.config import Settings
from dotenv import load_dotenv
from transformers import AutoConfig, pipeline, LlamaTokenizer, LlamaForCausalLM
import time

original_model_id="all-mpnet-base-v2"
_ = load_dotenv()

"""
config = AutoConfig.from_pretrained(original_model_id, torchscript=True)
#Load the tokenizer

#Load the base model
original_model = LlamaForCausalLM.from_pretrained(
                    original_model_id, config=config,
                        load_in_4bit=True, device_map="auto")
"""

DB_DIR = "./chroma_db"

#tokenizer = LlamaTokenizer.from_pretrained(original_model_id, use_fast=True, device_map="auto")
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

embeddings = HuggingFaceEmbeddings(
    model_name=original_model_id,
)


"""
loader = DirectoryLoader("./rag_documents", glob="**/*.md", show_progress=True)
docs = loader.load()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

# split the document data
split_docs = text_splitter.split_documents(docs)
"""

#config = AutoConfig.from_pretrained(original_model_id, torchscript=True)
# configure our database
client_settings = Settings(
    persist_directory=DB_DIR, #location to store 
    anonymized_telemetry=False # optional but showing how to toggle telemetry
)

client = chromadb.PersistentClient(DB_DIR)
collection = client.get_or_create_collection("Transcripts_Store")

# create a class level variable for the vector store
vector_store = None
persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("Transcripts_Store")

# check if the database exists already
# if not, create it, otherwise read from the database
if not os.path.exists(DB_DIR):
    # Create the database from the document(s) above and use the OpenAI embeddings for the word to vector conversions. We also pass the "persist_directory" parameter which means this won't be a transient database, it will be stored on the hard drive at the DB_DIR location. We also pass the settings we created earlier and give the collection a name
    vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings, client=client,
                    collection_name="Transcripts_Store", persist_directory=DB_DIR, client_settings=client_settings)

    print("About to persist db")
    # It's key to called the persist() method otherwise it won't be saved 
    vector_store.persist()
else:
    # As the database already exists, load the collection from there
    vector_store = Chroma(collection_name="Transcripts_Store", client_settings=client_settings, client=client,
                            embedding_function=embeddings, persist_directory=DB_DIR)


docs = vector_store.similarity_search("tell me more about llama experiments")
print(str(docs))

print("goodbye")

