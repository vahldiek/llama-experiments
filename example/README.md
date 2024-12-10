# LLM Chat Examples

## Prerequisits

* Tested on Ubuntu 23.10
* [Logged in to huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
* [Installed docker cli](https://docs.docker.com/engine/install/)

## Build Docker Images

Build trusted and untrusetd image.

```
DOCKER_BUILDKIT=1 docker build -f example/Dockerfile.trusted -t llm-chat-trusted .
DOCKER_BUILDKIT=1 docker build -f example/Dockerfile.untrusted -t llm-chat-untrusted .
```

## Run Image

Run trusted image:
```
docker run --rm -v .:/wdir/llama-experiments -v ~/.cache/huggingface:/root/.cache/huggingface -p 8501:8501 -it llm-chat-trusted
```

The streamlit chat is then available at [localhost:8501](localhost:8501)

## Stop Image

CRTL + c