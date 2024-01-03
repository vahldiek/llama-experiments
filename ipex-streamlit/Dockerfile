# pull from devel image instead of base
FROM ubuntu:24.04


ARG DEVCONTINER_DIR="./transformers-devcontainer"
ARG DEFAULT_USER="streamlit"
ARG INSTALL_DIR="/etc/transformers-experiments"
ARG RUN_DIR="/home/streamlit"
ARG IPEX_STREAMLIT_DIR="./ipex-streamlit"

#Add the default user in case they haven't already been added
#Continue if the user is already there
RUN useradd -m $DEFAULT_USER
RUN mkdir -p /home/$DEFAULT_USER/.local/bin
RUN chown $DEFAULT_USER:$DEFAULT_USER /home/$DEFAULT_USER/.local
RUN chown $DEFAULT_USER:$DEFAULT_USER /home/$DEFAULT_USER/.local/bin

COPY $DEVCONTINER_DIR/config.toml /home/$DEFAULT_USER/.streamlit/config.toml

COPY $DEVCONTINER_DIR/apt_requirements.txt apt_requirements.txt
RUN apt -y update
RUN xargs apt -y install < ./apt_requirements.txt
#Not sure what this junk is about but this fixes.
RUN mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old
COPY $IPEX_STREAMLIT_DIR/.profile /home/$DEFAULT_USER/.profile
RUN chown $DEFAULT_USER:$DEFAULT_USER /home/$DEFAULT_USER/.profile
WORKDIR $RUN_DIR

RUN mkdir -p $INSTALL_DIR/expansive_rag_documents
RUN mkdir -p $INSTALL_DIR/limited_rag_documents
COPY expansive_rag_documents $INSTALL_DIR/expansive_rag_documents
COPY limited_rag_documents $INSTALL_DIR/limited_rag_documents

USER $DEFAULT_USER
ENV PATH=/home/$DEFAULT_USER/.local/bin:$PATH
COPY $DEVCONTINER_DIR/pip_requirements.txt pip_requirements.txt
RUN pip install -r ./pip_requirements.txt



COPY ["chroma_utils.py", "download_and_save_model.py", "ipex_inference_transformers.py", \
"reset_RAG_database.py", "token_conversation.py", "transformers_streamlit.py", \
"transformers_utils.py", "$INSTALL_DIR"]

COPY $IPEX_STREAMLIT_DIR/container_config.toml  $RUN_DIR/.transformers_config.toml

# Create directory as a good point to place a shared volume
# with the host OS if desired.
USER root
RUN mkdir -p /etc/models
RUN mkdir -p /etc/shared
RUN chown $DEFAULT_USER:$DEFAULT_USER /etc/models
RUN chown $DEFAULT_USER:$DEFAULT_USER /etc/shared
RUN chown -R $DEFAULT_USER:$DEFAULT_USER /home/$DEFAULT_USER
RUN chown -R $DEFAULT_USER:$DEFAULT_USER $INSTALL_DIR
USER $DEFAULT_USER

WORKDIR $RUN_DIR
RUN python3 $INSTALL_DIR/reset_RAG_database.py
#streamlit listening here
EXPOSE 8880
#ENV SCRIPT_FULL_PATH="$INSTALL_DIR/transformers_streamlit.py"
ENTRYPOINT ["streamlit", "run", "/etc/transformers-experiments/transformers_streamlit.py"]





