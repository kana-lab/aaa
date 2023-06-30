use std::collections::HashMap;
use std::slice::Iter;
use tch::{Device, Kind, nn, no_grad, Tensor};
use anyhow::Result;
use rand::seq::SliceRandom;
use tch::nn::{BatchNormConfig, ConvConfig, ConvConfigND, init, LinearConfig, ModuleT, OptimizerConfig};
use aaa::net::Net;

const N_ASSETS: usize = 8;
const WINDOW_SIZE: usize = 50;
const KERNEL_SIZE: [i64; 2] = [N_ASSETS as i64, 4];
const LR: f64 = 1e-6;
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


fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    println!("device: {:?}", device);

    let (
        lpm_batches, pcr_matrices, test_lpm_batch, test_pcr_matrix
    ) = load_dataset(device)?;

    let vs = nn::VarStore::new(device);
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, LR)?;
    // let mut opt = nn::Sgd::default().build(&vs, LR)?;
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
            opt.zero_grad();
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