FROM ubuntu:25.04
WORKDIR /
RUN apt update


RUN apt install -y golang
RUN apt install -y autoconf automake autotools-dev curl python3 python3-pip python3-tomli libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev
RUN git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
RUN cd riscv-gnu-toolchain && git checkout 2025.01.20
WORKDIR /riscv-gnu-toolchain
RUN git submodule update --init --recursive
RUN ./configure --prefix=/opt/riscv --with-arch=rv64gc
#RUN make
WORKDIR /

