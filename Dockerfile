FROM ubuntu:latest

RUN mkdir -p /app/tensors/models/
COPY ./target/release/bybit /app/bybit
COPY ./settings.json /app/settings.json
COPY ./tensors/models/model.safetensors /app/tensors/models/model.safetensors

CMD ["/app/bybit"]
