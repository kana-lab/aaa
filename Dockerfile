FROM rust:latest

# This Dockerfile is intended for automated trading on Bybit.
# Prior to running `docker build`, you need to generate `model.safetensors` & `settings.json`.

RUN mkdir -p /app/tensors/models/
COPY ./tensors/models/model.safetensors /app/tensors/models/
COPY ./settings.json /app/

WORKDIR /app
RUN apt update && apt install -y wget unzip
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
RUN unzip libtorch.zip
ENV LIBTORCH /app/libtorch
ENV LD_LIBRARY_PATH /app/libtorch/lib

RUN mkdir /build
WORKDIR /build
RUN mkdir ./src
COPY ./src ./src
COPY Cargo.toml .
COPY Cargo.lock .
RUN cargo build --release --bin bybit
RUN mv ./target/release/bybit /app/bybit
RUN rm -r /build
WORKDIR /app

CMD ["/app/bybit"]
