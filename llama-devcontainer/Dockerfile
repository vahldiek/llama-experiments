# pull from devel image instead of base
FROM mcr.microsoft.com/devcontainers/base:ubuntu
WORKDIR /home/vscode

# Set bash as the default shell
ENV SHELL=/bin/bash

# Copy the streamlit configuration file to the home directory
RUN mkdir -p /home/vscode/.streamlit

# Change ownership to vscode since build is done as root.
RUN chown vscode:vscode /home/vscode/.streamlit
COPY config.toml /home/vscode/.streamlit/config.toml
RUN chown vscode:vscode /home/vscode/.streamlit/config.toml

RUN apt -y update
RUN apt -y install python3 python3-pip build-essential
COPY requirements.txt requirements.txt
RUN pip install -r ./requirements.txt
COPY app.py app.py

# Create directory as a good point to place a shared volume
# with the host OS if desired.
RUN mkdir -p /etc/models
RUN git clone https://github.com/intel/intel-extension-for-pytorch.git
RUN cd intel-extension-for-pytorch && git checkout v2.1.0+cpu



