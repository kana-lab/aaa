use std::collections::HashMap;
use std::fs;
use std::ops::{Add, Mul};
use std::path::{Path, PathBuf};
use std::slice::Iter;
use rand::rngs::ThreadRng;
use tch::{Device, Kind, nn, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::nn::{ModuleT, OptimizerConfig};
use aaa::net::Net;
use aaa::settings_json_format::Settings;
use aaa::util::{input, input_bool};


struct Env {
    // lpm_batches: Tensor,
    // pcr_matrices: Tensor,
    // validation_lpm: Tensor,
    // validation_pcr: Tensor,
    // test_lpm: Tensor,
    // test_pcr: Tensor,
    train_gpm: Tensor,
    validation_lpm: Tensor,
    validation_pcr: Tensor,
    test_lpm: Tensor,
    test_pcr: Tensor,

    window_size: i64,
    batch_size: i64,
    rng: ThreadRng,
    indices: Vec<i64>,
}

fn load_environment(to: Device, settings: &Settings) -> Env {
    let mut env_path = Path::new(
        &settings.preprocess.environment_files_dir
    ).to_path_buf();
    env_path.push(Path::new(&settings.preprocess.default_environment_file_name));
    if !env_path.exists() {
        eprintln!("could not read environment file `{}`", env_path.to_str().unwrap());
        std::process::exit(1);
    }

    let result = match Tensor::read_safetensors(&env_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to read .safetensors: {}", e);
            std::process::exit(1);
        }
    };

    let err_handler = || {
        eprintln!(".safetensors file is broken.");
        std::process::exit(1);
    };

    let mut map: HashMap<String, Tensor> = result.into_iter().collect();
    let train_gpm = map.remove("train_gpm").unwrap_or_else(err_handler).to(to);
    let window_size = settings.preprocess.window_size as i64;
    let batch_size = settings.preprocess.batch_size as i64;
    let n_batch = (train_gpm.size()[1] - window_size) / batch_size;
    Env {
        // lpm_batches: map.remove("lpm_batches").unwrap_or_else(err_handler).to(to),
        // pcr_matrices: map.remove("pcr_matrices").unwrap_or_else(err_handler).to(to),
        // validation_lpm: map.remove("validation_lpm").unwrap_or_else(err_handler).to(to).detach(),
        // validation_pcr: map.remove("validation_pcr").unwrap_or_else(err_handler).to(to).detach(),
        // test_lpm: map.remove("test_lpm").unwrap_or_else(err_handler).to(to).detach(),
        // test_pcr: map.remove("test_pcr").unwrap_or_else(err_handler).to(to).detach(),
        train_gpm,
        validation_lpm: map.remove("validation_lpm").unwrap_or_else(err_handler).to(to),
        validation_pcr: map.remove("validation_pcr").unwrap_or_else(err_handler).to(to),
        test_lpm: map.remove("test_lpm").unwrap_or_else(err_handler).to(to),
        test_pcr: map.remove("test_pcr").unwrap_or_else(err_handler).to(to),
        window_size,
        batch_size,
        rng: thread_rng(),
        indices: (0..n_batch).collect(),
    }
}

impl Env {
    fn total_lpm(&self) -> i64 {
        self.train_gpm.size()[1] - self.window_size
    }

    fn make_lpm_pcr(&self, lpm_idx: i64) -> (Tensor, Tensor) {
        let i = self.window_size - 1 + lpm_idx;
        let window = self.train_gpm.narrow(
            1, i + 1 - self.window_size, self.window_size,
        );
        let current_price = self.train_gpm.narrow(1, i, 1);
        let next_price = self.train_gpm.narrow(1, i + 1, 1);
        let y = (&next_price / &current_price).flatten(0, -1);
        (window / current_price, y)
    }

    fn iter_train(&mut self) -> TrainLPMIter {
        self.indices.shuffle(&mut self.rng);
        TrainLPMIter {
            env: self,
            batch_indices: self.indices.clone(),
        }
    }
}

struct TrainLPMIter<'a> {
    env: &'a Env,
    batch_indices: Vec<i64>,
}

impl<'a> Iterator for TrainLPMIter<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let mut lpm_vec = vec![];
        let mut pcr_vec = vec![];

        let batch_idx = match self.batch_indices.pop() {
            None => { return None; }
            Some(idx) => idx
        };

        for i in (self.env.batch_size * batch_idx)..(self.env.batch_size * (batch_idx + 1)) {
            let (lpm, pcr) = self.env.make_lpm_pcr(i);
            lpm_vec.push(lpm);
            pcr_vec.push(pcr);
        }

        Some((Tensor::stack(&lpm_vec, 0), Tensor::stack(&pcr_vec, 0)))
    }
}

fn get_model_path(settings: &Settings) -> PathBuf {
    let mut model_file_path = Path::new(
        &settings.train.model_files_dir
    ).to_path_buf();

    // check if the model directory exists
    if !model_file_path.exists() {
        println!("the directory to save the model file does NOT exist.");
        let b = input_bool("would you like to create?", true);
        if b {
            if let Err(err) = fs::create_dir_all(&model_file_path) {
                eprintln!("Failed to create the model directory: {}", err);
                std::process::exit(1);
            } else {
                println!("created");
            }
        } else {
            println!("\nabort.");
            std::process::exit(0);
        }
    }

    // check if model.safetensors exists
    // if exists, rename or abort
    model_file_path.push(
        Path::new(&settings.train.default_model_file_name)
    );
    if model_file_path.exists() {
        println!("model file does already exist: {}", model_file_path.to_str().unwrap());
        let c = input_bool("do you want to continue anyway?", true);
        if !c {
            println!("\nabort.");
            std::process::exit(0);
        }

        println!("please specify the name of the environment file (just Enter to overwrite).");
        let s = input("[name] ");
        if !s.is_empty() {
            model_file_path.pop();
            model_file_path.push(Path::new(&s));
        }
    }

    model_file_path
}


fn main() {
    let device = Device::cuda_if_available();
    println!("device: {:?}", device);

    let settings = match Settings::load() {
        Ok(settings) => settings,
        Err(e) => {
            eprintln!("could not read settings.json: {}", e);
            std::process::exit(1);
        }
    };

    let model_file_path = get_model_path(&settings);

    let mut env = load_environment(device, &settings);
    let test_lpm = env.test_lpm.copy();
    let validation_lpm = env.validation_lpm.copy();
    let test_pcr = env.test_pcr.copy();
    let validation_pcr = env.validation_pcr.copy();

    // let mut rng = rand::thread_rng();
    // let mut indices: Vec<i64> = (0..env.lpm_batches.size()[0]).collect();

    let get_actual_reward = |net: &Net, test: bool| {
        let lpm = if test { &test_lpm } else { &validation_lpm };
        let pcr = if test { &test_pcr } else { &validation_pcr };
        let (p, w) = net.forward_t(lpm, false);

        let timing_idx = p.gt(0.5).nonzero().flatten(0, -1);
        let pcr = Tensor::cat(&[
            &Tensor::ones([1, pcr.size()[1]], (Kind::Float, device)), pcr
        ], 0);
        let pcr_cumprod = pcr.cumprod(0, Kind::Float);
        let w = w.index_select(0, &timing_idx);
        let pcr = pcr_cumprod.index_select(0, &timing_idx);
        let pcr = Tensor::cat(&[
            pcr.narrow(0, 1, pcr.size()[0] - 1),
            pcr_cumprod.narrow(0, pcr_cumprod.size()[0] - 1, 1)
        ], 0) / pcr;

        let dim: &[i64] = &[1];
        let r = (&w * pcr)
            .sum_dim_intlist(dim, false, Kind::Float)
            .prod(Kind::Float);
        f64::try_from(r).unwrap()
    };

    let mut vs_buf = Vec::new();
    for n_model in 0..1 {
        println!("\n[model {}]", n_model);

        let vs = nn::VarStore::new(device);
        let net = Net::new(&vs.root(), &settings);
        let mut opt = match nn::Adam::default().build(&vs, settings.train.learning_rate) {
            Ok(ok) => ok,
            Err(e) => {
                eprintln!("Failed to construct optimizer: {}", e);
                std::process::exit(1);
            }
        };
        opt.set_weight_decay(1e-8);

        // let mut growth_data = Vec::new();
        for epoch in 0..settings.train.epoch {
            // let indices: Vec<i64> = Vec::try_from(Tensor::randperm(
            //     lpm_batches.size()[0], (lpm_batches.kind(), lpm_batches.device()),
            // )).unwrap();
            // indices.shuffle(&mut rng);

            for (lpm_batch, pcr_matrix) in env.iter_train() {
                opt.zero_grad();
                // let lpm_batch = env.lpm_batches.get(i);
                // let pcr_matrix = env.pcr_matrices.get(i);
                let (p, w) = net.forward_t(&lpm_batch, true);
                let dim: &[i64] = &[1];

                let asset_changes = (w * &pcr_matrix)
                    .sum_dim_intlist(dim, false, Kind::Float);
                let reward = (asset_changes * &p - &p + 1.)
                    .log()
                    .mean(Kind::Float);
                let loss = reward * -1;
                opt.backward_step(&loss);
            }

            if epoch % 30 == 0 {
                let reward = get_actual_reward(&net, true);
                let reward2 = get_actual_reward(&net, false);
                println!("epoch {}: {} {}", epoch, reward, reward2);
                // growth_data.push(f64::try_from(reward).unwrap());
            }
        }

        let reward = get_actual_reward(&net, true);
        let reward2 = get_actual_reward(&net, false);
        println!("epoch (last): {} {}", reward, reward2);

        vs_buf.push((reward, reward2, vs));

        // println!("{:?}", growth_data);
    }

    let (score, validation, vs) = {
        let mut max_score = vs_buf[0].0;
        let mut max_index = 0;
        for i in 0..vs_buf.len() {
            if max_score < vs_buf[i].0 {
                max_score = vs_buf[i].0;
                max_index = i;
            }
        }
        vs_buf.remove(max_index)
    };

    println!("\nbiggest score: {}", score);
    println!("its validation score: {}", validation);
    println!("this model will be saved.\n");

    match vs.save(model_file_path) {
        Ok(()) => {
            println!("Successfully saved.");
        }
        Err(e) => {
            eprintln!("Failed to save the model: {}", e);
        }
    };
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