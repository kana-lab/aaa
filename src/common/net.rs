use tch::{Kind, nn, Tensor};
use crate::settings_json_format::Settings;

const KERNEL_SIZE: i64 = 4;
const DROPOUT_RATE: f64 = 0.7;
const N_LATENTS: i64 = 1500;

#[derive(Debug)]
pub struct Net {
    conv: nn::Conv2D,
    norm2: nn::BatchNorm,
    fc: nn::Linear,
    norm1: nn::BatchNorm,
    asset_ratio_output: nn::Linear,
    probability_output: nn::Linear,

    window_size: i64,
    n_assets: i64,
}

impl Net {
    pub fn new(vs: &nn::Path, settings: &Settings) -> Self {
        let window_size = settings.preprocess.window_size as i64;
        let n_assets = settings.download.asset_names.len() as i64 + 1;

        let conv = nn::conv(
            vs, 1, n_assets, [n_assets, KERNEL_SIZE], Default::default(),
        );
        let fc = nn::linear(
            vs, n_assets * (window_size - KERNEL_SIZE + 1),
            N_LATENTS, Default::default(),
        );
        let asset_ratio_output = nn::linear(vs, N_LATENTS, n_assets, Default::default());
        let probability_output = nn::linear(vs, N_LATENTS, 1, Default::default());
        let norm1 = nn::batch_norm1d(vs, N_LATENTS, Default::default());
        let norm2 = nn::batch_norm2d(vs, n_assets, Default::default());
        Self { conv, norm2, fc, norm1, asset_ratio_output, probability_output, window_size, n_assets }
    }
}

// impl nn::ModuleT for Net {
impl Net {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let latents = xs.view([-1, 1, self.n_assets, self.window_size])
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
            .relu();
        let ratio = latents
            .apply(&self.asset_ratio_output)
            .softmax(-1, Kind::Float);
        let p = latents
            .apply(&self.probability_output)
            .sigmoid();
        let p = p.flatten(0, -1);
        (p, ratio)
    }
}