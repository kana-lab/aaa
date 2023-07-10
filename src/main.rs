fn main() {
    println!(r#"
Auto Asset Allocator.

To download data and preprocess it:
$ cargo run --bin preprocess

To train a model:
$ cargo run --bin train

To auto-trade on Bybit:
$ cargo run --bin bybit

Please also see `settings.json`.
"#);
}
