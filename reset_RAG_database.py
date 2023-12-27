#!/usr/bin/env python3
import llama_utils
import chroma_utils

config = llama_utils.read_config()

#This script always destroys and rebuilds the DB
#regardless of the rest of the contents of the config file
vector_store = chroma_utils.get_vector_store(config, True, True)