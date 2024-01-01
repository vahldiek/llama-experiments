#!/usr/bin/env python3
import transformers_utils
import chroma_utils
import logging

logging.getLogger('transformers_streamlit').setLevel(logging.DEBUG)

config = transformers_utils.read_config()

#This script always destroys and rebuilds the DB
#regardless of the rest of the contents of the config file
vector_store = chroma_utils.get_vector_store(config, True, True)