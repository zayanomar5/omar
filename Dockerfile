# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf

Run git clone "https://huggingface.co/sentence-transformers/paraphrase-TinyBERT-L6-v2"

RUN mkdir /.cache

RUN chmod 777 /.cache

# RUN mkdir /.cache/huggingface

# RUN mkdir /.cache/huggingface/hub

# RUN mkdir /.cache/huggingface/hub/models--sentence-transformers--paraphrase-TinyBERT-L6-v2

# RUN mkdir /.cache/huggingface/hub/models--sentence-transformers--paraphrase-TinyBERT-L6-v2/blobs

# RUN mkdir /.cache/huggingface/hub/models--sentence-transformers--paraphrase-TinyBERT-L6-v2/snapshots

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "300", "main:app"]
