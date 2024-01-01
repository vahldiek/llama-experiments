# Llama Experiments
This project contains sample code for experimenting with the Llama2 language model and the Intel® Extension for PyTorch\*
The project includes a development container that can be used with VSCode.  Alternatively the Dockerfile in the
`transformers-devcontainer` directory can be examined to determine the environment requirements for these samples.
The Python scripts used in this project load their configuration data from the file `.transformers_config.toml` in the top
level directory.  A different configuration file can be specified as a command line argument.

Very little of the code is specific to Llama2, it should be applicable to all Hugging Face transfomers models with minimal changes
but it has only been tested with Llama2.

## Installation
It is easiest to build a container image for the development container.  The script build_dev_container is provided to perform this operation.
When launching the container it is important to allocate enough memory to it, even the 7 Billion parameter Llama models can require 80GB or more.

If you want the scripts to automatically pull images from Hugging Face, you should rename the file .env_template to .env and put a valid
Hugging Face token in the file.

## streamlit
The sample Python code is designed to work with `streamlit` a simple HTTP server that presents a chatbot interface.
Before launching streamlit, edit the parameters in `.transformers_config.toml` to point to the location of your origin and quantized models.
To launch streamlit, you can use the script launch_streamlit.
Note that streamlit will execute the entire Python file several times, each time anything changes in the GUI.  But it has a mechansim to avoid
redundant expensive operations like model loading.  `transformers_streamlit.py` uses streamlit function decorations to ensure that models are only loaded
once.

Streamlit will not begin loading the model until a browser connects to the server.  VSCode will auto-launch the browser but sometimes the
browser will attempt to connect to streamlit before the HTTP server has started.  In such cases, it looks like streamlit is hung but simply
reconnecting to the streamlit HTTP server usually gets things going.

## Downlaoding and saving Hugging Face models
The Python script `download_and_save_model.py` is provided to download models automatically from Hugging Face and store them in a local directory for later retrieval.  This is useful if running the scripts in a container environment, the downloaded models can be persisted locally and retreived even if the development container is rebuilt.

## Retrieval Augmented Generation (RAG)
The scripts can optionally attampt to find relevant content by first querying a Chroma in-memory vector database, and sending this content along to the LLM.  Options in `.transformers_config.toml` enable or disable this feature.  Executing the script `reset_RAG_database.py` will build or rebuild the database using the data file directory speficied in the configuration file.
