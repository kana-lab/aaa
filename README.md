# Auto Asset Allocator

This program trains an agent to learn the optimal asset allocation for cryptocurrencies 
using reinforcement learning. It also supports downloading datasets from Binance 
and automated trading on the Bybit platform.

Note that the reinforcement learning algorithm is based on the following paper:

Z.Jiang, J.Liang, "Cryptocurrency Portfolio Management with Deep Reinforcement Learning",
https://arxiv.org/pdf/1612.01277.pdf

### download dataset & preprocess

```shell
cargo run --bin preprocess
```

### model training

```shell
cargo run --bin train
```

### auto-trade on Bybit

```shell
cargo run --bin bybit
```

Please also see `settings.json`.