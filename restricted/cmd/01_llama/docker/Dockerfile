FROM ubuntu:25.04


WORKDIR /dist

RUN apt update

RUN apt install -y bash curl unzip wget

COPY cmd/01_llama/docker/run.sh .
COPY cmd/01_llama/llama3_1b/llama3_1b .
COPY cmd/01_llama/prompt.txt .

RUN chmod +x llama3_1b
RUN chmod +x run.sh

ENV Enable_log=false

ENTRYPOINT ["./run.sh"]