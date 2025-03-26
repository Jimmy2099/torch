FROM golang:1.23.7


RUN apt update
RUN apt install -y python3 python3-venv python3-pip

# Create and activate a virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

#pytorch for data_loader and trainer
RUN pip install torch==2.6.0

RUN mkdir /torch
ADD . /torch
RUN ls

WORKDIR  /torch
RUN go mod tidy

WORKDIR cmd/0_standalone_example
RUN go build -o 0_standalone_example main.go
RUN ls

ENTRYPOINT ["/torch/cmd/0_standalone_example/0_standalone_example"]