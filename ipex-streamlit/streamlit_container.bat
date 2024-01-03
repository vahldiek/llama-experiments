set MODEL_DIR=C:\Users\Matt\Documents\llama-experiments
set CONFIG_DIR=%MODEL_DIR%\ipex-streamlit
docker run -it --rm -p 8880:8880 --gpus all --shm-size 128G --mount type=bind,source="%MODEL_DIR%",target=/etc/models --mount type=bind,source="%CONFIG_DIR%",target=/etc/shared ipex-streamlit:llama2 /etc/shared/container_config.toml