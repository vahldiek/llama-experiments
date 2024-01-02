# pull from devel image instead of base
FROM mcr.microsoft.com/devcontainers/base:ubuntu
WORKDIR /home/vscode

# Set bash as the default shell
ENV SHELL=/bin/bash

COPY apt_requirements.txt apt_requirements.txt
RUN apt -y update
RUN xargs apt -y install < ./apt_requirements.txt

#execute the rest of the commands as the vscode user
USER vscode
# Copy the streamlit configuration file to the home directory
RUN mkdir -p /home/vscode/.streamlit

COPY config.toml /home/vscode/.streamlit/config.toml

COPY pip_requirements.txt pip_requirements.txt
RUN pip install -r ./pip_requirements.txt
COPY app.py app.py

# Create directory as a good point to place a shared volume
# with the host OS if desired.

RUN git clone https://github.com/intel/intel-extension-for-pytorch.git
RUN cd intel-extension-for-pytorch && git checkout v2.1.0+cpu
USER root
RUN mkdir -p /etc/models
RUN chown vscode:vscode /etc/models
USER vscode


