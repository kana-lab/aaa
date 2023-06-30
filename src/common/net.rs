use tch::{Device, Kind, nn, no_grad, Tensor};
use tch::nn::{BatchNormConfig, ConvConfig, ConvConfigND, init, LinearConfig, ModuleT, OptimizerConfig};

const N_ASSETS: usize = 8;
const WINDOW_SIZE: usize = 50;
const KERNEL_SIZE: [i64; 2] = [N_ASSETS as i64, 4];
const LR: f64 = 1e-6;
// todo: Does total steps described in the paper mean epochs?
const EPOCH: usize = 2500;

#[derive(Debug)]
pub struct Net {
    conv: nn::Conv2D,
    // conv: nn::Linear,
    norm2: nn::BatchNorm,
    fc: nn::Linear,
    norm1: nn::BatchNorm,
    output: nn::Linear,
}

impl Net {
    pub fn new(vs: &nn::Path) -> Self {
        let std_dev = init::Init::Randn { mean: 0., stdev: 0.1 };

        let mut conv_config: ConvConfigND<_> = Default::default();
        // conv_config.ws_init = std_dev.clone();
        let mut linear_config: LinearConfig = Default::default();
        // linear_config.ws_init = std_dev.clone();
        let mut norm_config: BatchNormConfig = Default::default();
        // norm_config.ws_init = std_dev.clone();
        // norm_config.eps = 1e-8;

        let conv = nn::conv(vs, 1, 8, KERNEL_SIZE, conv_config);
        // let conv = nn::linear(vs, (N_ASSETS * WINDOW_SIZE) as i64,376, linear_config.clone());
        let fc = nn::linear(vs, 376, 500, linear_config.clone());
        let output = nn::linear(vs, 500, N_ASSETS as i64, linear_config);
        let norm1 = nn::batch_norm1d(vs, 500, norm_config.clone());
        let norm2 = nn::batch_norm2d(vs, 8, norm_config);
        // let norm2 = nn::batch_norm1d(vs, 376, norm_config.clone());
        Self { conv, norm2, fc, norm1, output }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, N_ASSETS as i64, WINDOW_SIZE as i64])
            // xs.view([-1,(N_ASSETS*WINDOW_SIZE) as i64])
            .apply(&self.conv)
            // .dropout(0.7, train)
            .apply_t(&self.norm2, train)
            // the order of ReLU -> Flatten is more common, according to GPT-3.5
            .relu()
            .flatten(1, -1)
            .apply(&self.fc)
            // NOTICE: the position of BatchNorm is NOT described in the paper
            .apply_t(&self.norm1, train)
            // NOTICE: the position of dropout is NOT described in the paper
            .dropout(0.7, train)
            // .apply_t(&self.norm1, train)
            .relu()
            .apply(&self.output)
            .softmax(-1, Kind::Float)
    }
}