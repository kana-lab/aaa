use std::collections::HashMap;
use std::slice::Iter;
use tch::{Device, Kind, nn, no_grad, Tensor};
use anyhow::Result;
use rand::seq::SliceRandom;
use tch::nn::{BatchNormConfig, ConvConfig, ConvConfigND, init, LinearConfig, ModuleT, OptimizerConfig};

const N_ASSETS: usize = 8;
const WINDOW_SIZE: usize = 50;
const KERNEL_SIZE: [i64; 2] = [N_ASSETS as i64, 4];
const LR: f64 = 1e-5;
// todo: Does total steps described in the paper mean epochs?
const EPOCH: usize = 2500;


fn load_dataset(device: Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let result = Tensor::read_safetensors("../dataset/environment.safetensors")?;
    let mut map: HashMap<String, Tensor> = result.into_iter().collect();
    Ok((
        map.remove("lpm_batches").unwrap().to(device),
        map.remove("pcr_matrices").unwrap().to(device),
        map.remove("test_lpm_batch").unwrap().to(device).requires_grad_(false),
        map.remove("test_pcr_matrix").unwrap().to(device).requires_grad_(false)
    ))
}

#[derive(Debug)]
struct Net {
    conv: nn::Conv2D,
    norm2: nn::BatchNorm,
    fc: nn::Linear,
    norm1: nn::BatchNorm,
    output: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Self {
        let std_dev = init::Init::Randn { mean:0., stdev:0.1 };

        let mut conv_config : ConvConfigND<_> = Default::default();
        // conv_config.ws_init = std_dev.clone();
        let mut linear_config : LinearConfig = Default::default();
        // linear_config.ws_init = std_dev.clone();
        let mut norm_config: BatchNormConfig = Default::default();
        // norm_config.ws_init = std_dev.clone();
        // norm_config.eps = 1e-8;

        let conv = nn::conv(vs, 1, 8, KERNEL_SIZE, conv_config);
        let fc = nn::linear(vs, 376, 500, linear_config.clone());
        let output = nn::linear(vs, 500, N_ASSETS as i64, linear_config);
        let norm1 = nn::batch_norm1d(vs, 500, norm_config.clone());
        let norm2 = nn::batch_norm2d(vs, 8, norm_config);
        Self { conv, norm2, fc, norm1, output }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, N_ASSETS as i64, WINDOW_SIZE as i64])
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

fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    println!("device: {:?}", device);

    let (
        lpm_batches, pcr_matrices, test_lpm_batch, test_pcr_matrix
    ) = load_dataset(device)?;

    let vs = nn::VarStore::new(device);
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, LR)?;
    opt.set_weight_decay(1e-8);

    let mut rng = rand::thread_rng();
    let mut indices: Vec<i64> = (0..lpm_batches.size()[0]).collect();
    let mut growth_data = Vec::new();

    for epoch in 0..EPOCH {
        // let indices: Vec<i64> = Vec::try_from(Tensor::randperm(
        //     lpm_batches.size()[0], (lpm_batches.kind(), lpm_batches.device()),
        // )).unwrap();
        indices.shuffle(&mut rng);

        for &i in &indices {
            let lpm_batch = lpm_batches.get(i);
            let pcr_matrix = pcr_matrices.get(i);
            let mut w = net.forward_t(&lpm_batch, true);
            let dim: &[i64] = &[1];

            let reward = (w * &pcr_matrix)
                .sum_dim_intlist(dim, false, Kind::Float)
                .log()
                .mean(Kind::Float);
            let loss = reward * -1;
            opt.backward_step(&loss);
        }

        if epoch % 30 == 0 {
            let reward = {
                println!("{:?}", test_pcr_matrix.size());
                let mut w = net.forward_t(&test_lpm_batch, false);
                let dim: &[i64] = &[1];
                (&w * &test_pcr_matrix)
                    .sum_dim_intlist(dim, false, Kind::Float)
                    .prod(Kind::Float)
            };
            println!("epoch {}: {}", epoch, reward);
            growth_data.push(f64::try_from(reward).unwrap());
        }
    }

    vs.save("./model.safetensors")?;
    println!("{:?}", growth_data);

    Ok(())
}

#[test]
fn t() {
    // 行列の定義
    let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    // 列ごとの和を計算
    let dim: &[i64] = &[1];
    let sum = matrix.sum_dim_intlist(dim, false, Kind::Int64);

    // 結果の表示
    println!("Column-wise sum: {:?}", sum);
    let v: Vec<f64> = Vec::try_from(&sum).unwrap();
    println!("{:?}", v);
    let t = matrix.get(0);
    // let t: Vec<f64> = Vec::try_from(&t).unwrap();
    println!("{:?}", t);
}

#[test]
fn t2() {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let conv = nn::conv(&vs.root(), 2, 12, [12, 4], Default::default());
    println!("{:?}", conv)
}