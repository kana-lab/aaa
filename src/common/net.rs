use tch::{Device, Kind, nn, no_grad, Tensor};
use tch::nn::{BatchNormConfig, ConvConfig, ConvConfigND, init, LinearConfig, ModuleT, OptimizerConfig};
use crate::settings_json_format::Settings;

const KERNEL_SIZE: [i64; 2] = [N_ASSETS as i64, 4];
const DROPOUT_RATE: f64 = 0.7;

#[derive(Debug)]
pub struct Net {
    conv: nn::Conv2D,
    norm2: nn::BatchNorm,
    fc: nn::Linear,
    norm1: nn::BatchNorm,
    output: nn::Linear,

    window_size: i64,
    n_assets: i64,
}

impl Net {
    pub fn new(vs: &nn::Path, settings: &Settings) -> Self {
        let window_size = settings.preprocess.window_size as i64;
        let n_assets = settings.download.asset_names.len() as i64 + 1;

        let conv = nn::conv(
            vs, 1, n_assets, KERNEL_SIZE, Default::default(),
        );
        let fc = nn::linear(
            vs, n_assets * (window_size - KERNEL_SIZE[1] + 1),
            500, Default::default(),
        );
        let output = nn::linear(vs, 500, n_assets, Default::default());
        let norm1 = nn::batch_norm1d(vs, 500, Default::default());
        let norm2 = nn::batch_norm2d(vs, n_assets, Default::default());
        Self { conv, norm2, fc, norm1, output, window_size, n_assets }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, self.n_assets, self.window_size])
            .apply(&self.conv)
            .apply_t(&self.norm2, train)
            // the order of ReLU -> Flatten is more common, according to GPT-3.5
            .relu()
            .flatten(1, -1)
            .apply(&self.fc)
            // NOTICE: the position of BatchNorm is NOT described in the paper
            .apply_t(&self.norm1, train)
            // NOTICE: the position of dropout is NOT described in the paper
            .dropout(DROPOUT_RATE, train)
            .relu()
            .apply(&self.output)
            .softmax(-1, Kind::Float)
    }
}