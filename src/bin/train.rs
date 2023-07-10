use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::slice::Iter;
use tch::{Device, Kind, nn, Tensor};
use rand::seq::SliceRandom;
use tch::nn::{ModuleT, OptimizerConfig};
use aaa::net::Net;
use aaa::settings_json_format::Settings;
use aaa::util::{input, input_bool};


struct Env {
    lpm_batches: Tensor,
    pcr_matrices: Tensor,
    validation_lpm: Tensor,
    validation_pcr: Tensor,
    test_lpm: Tensor,
    test_pcr: Tensor,
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
    Env {
        lpm_batches: map.remove("lpm_batches").unwrap_or_else(err_handler).to(to),
        pcr_matrices: map.remove("pcr_matrices").unwrap_or_else(err_handler).to(to),
        validation_lpm: map.remove("validation_lpm").unwrap_or_else(err_handler).to(to).detach(),
        validation_pcr: map.remove("validation_pcr").unwrap_or_else(err_handler).to(to).detach(),
        test_lpm: map.remove("test_lpm").unwrap_or_else(err_handler).to(to).detach(),
        test_pcr: map.remove("test_pcr").unwrap_or_else(err_handler).to(to).detach(),
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

    let env = load_environment(device, &settings);

    let mut rng = rand::thread_rng();
    let mut indices: Vec<i64> = (0..env.lpm_batches.size()[0]).collect();

    let get_actual_reward = |net: &Net, test: bool| {
        let lpm = if test { &env.test_lpm } else { &env.validation_lpm };
        let pcr = if test { &env.test_pcr } else { &env.validation_pcr };
        let w = net.forward_t(lpm, false);
        let dim: &[i64] = &[1];
        let r = (&w * pcr)
            .sum_dim_intlist(dim, false, Kind::Float)
            .prod(Kind::Float);
        f64::try_from(r).unwrap()
    };

    let mut vs_buf = Vec::new();
    for n_model in 0..16 {
        println!("\n[model {}]", n_model);

        let vs = nn::VarStore::new(device);
        let net = Net::new(&vs.root());
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
            indices.shuffle(&mut rng);

            for &i in &indices {
                opt.zero_grad();
                let lpm_batch = env.lpm_batches.get(i);
                let pcr_matrix = env.pcr_matrices.get(i);
                let w = net.forward_t(&lpm_batch, true);
                let dim: &[i64] = &[1];

                let reward = (w * &pcr_matrix)
                    .sum_dim_intlist(dim, false, Kind::Float)
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