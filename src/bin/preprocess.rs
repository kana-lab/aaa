use std::{fs, io, thread};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use csv::ReaderBuilder;
use tch::{Kind, Tensor};
use aaa::util::{input, input_bool};
use aaa::settings_json_format::Settings;

fn main() {
    // load settings.json
    let settings = match Settings::load() {
        Ok(settings) => settings,
        Err(e) => {
            eprintln!("could not read settings.json: {}", e);
            std::process::exit(1);
        }
    };

    let mut env_file_path = Path::new(
        &settings.process.environment_files_dir
    ).to_path_buf();

    // check if the environment directory exists
    if !env_file_path.exists() {
        println!("the directory to save the environment file does NOT exist.");
        let b = input_bool("would you like to create?", true);
        if b {
            if let Err(err) = fs::create_dir_all(&env_file_path) {
                eprintln!("Failed to create the environment directory: {}", err);
                std::process::exit(1);
            } else {
                println!("created.");
            }
        } else {
            println!("\nabort.");
            return;
        }
    }

    // check if env.safetensors exists
    // if exists, rename or abort
    env_file_path.push(
        Path::new(&settings.process.default_environment_file_name)
    );
    if env_file_path.exists() {
        println!("environment file does already exist: {}", env_file_path.to_str().unwrap());
        let c = input_bool("do you want to continue anyway?", true);
        if !c {
            println!("\nabort.");
            return;
        }

        println!("please specify the name of the environment file (just Enter to overwrite).");
        let s = input("[name] ");
        if !s.is_empty() {
            env_file_path.pop();
            env_file_path.push(Path::new(&s));
        }
    }

    // check if dataset directory does exist
    // if exists, let user select re-download or not
    // otherwise, download or abort
    let dataset_path = Path::new(&settings.download.dataset_directory_path);
    let download = if dataset_path.exists() {
        println!("dataset does already exist at {}", dataset_path.to_str().unwrap());
        let b = input_bool("would you like to re-download?", false);
        if b {
            println!("\nnew dataset will be downloaded.");
            print!("removing the directory...");

            // remove the existing dataset directory
            if let Err(err) = fs::remove_dir_all(dataset_path) {
                eprintln!("\nFailed to delete the directory: {}", err);
                std::process::exit(1);
            } else {
                println!("done");
            }
        } else {
            println!("\nexisting dataset will be used.")
        }
        b
    } else {
        println!("dataset does NOT exist.");
        let b = input_bool("would you like to download?", true);
        if b {
            println!("\nnew dataset will be downloaded.");
        } else {
            println!("\nabort.");
            return;
        }
        b
    };

    if download {
        // (re-)create the dataset directory
        if let Err(err) = fs::create_dir_all(dataset_path) {
            eprintln!("Failed to create dataset directory: {}", err);
            std::process::exit(1);
        }

        println!("launching downloader...");
        let status = exec_download_script(&settings, &dataset_path);
        if !status.success() {
            eprintln!("the download script returned the non-zero value: {}", status);
            std::process::exit(1);
        } else {
            println!("done");
        }
    }

    let tensors = make_environment(&settings);
    if let Err(err) = Tensor::write_safetensors(&tensors, &env_file_path) {
        eprintln!("Failed to write environment tensors to .safetensors: {}", err);
        std::process::exit(1);
    }

    println!("created the environment tensors at {}", env_file_path.to_str().unwrap());
}

fn exec_download_script(settings: &Settings, run_dir: &Path) -> std::process::ExitStatus {
    // cast the download script path to the absolute path
    let dl_script = Path::new(&settings.download.download_script_path).canonicalize();
    let dl_script = match dl_script {
        Ok(ok) => ok,
        Err(err) => {
            eprintln!("Failed to get the absolute path of the download script: {}", err);
            std::process::exit(1)
        }
    };

    // build the execution command
    let proc = std::process::Command::new("/usr/bin/env")
        .arg("bash")
        .arg(dl_script)
        .env("ASSET_NAMES", settings.download.asset_names.join(":"))
        .env("NUM_MONTHS", settings.download.num_months.to_string())
        .current_dir(run_dir)
        .stdout(std::process::Stdio::piped())
        .spawn();

    if let Err(err) = proc {
        eprintln!("Failed to execute the download script: {}", err);
        std::process::exit(1);
    }

    // transport the output of the script to our stdout
    let mut proc = proc.unwrap();
    if let Some(stdout) = proc.stdout.take() {
        thread::spawn(move || {
            let reader = io::BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    println!("{}", line);
                }
            }
        });
    }

    let status = proc.wait();
    match status {
        Err(err) => {
            eprintln!("Failed to get status code from download script: {}", err);
            std::process::exit(1);
        }
        Ok(ok) => {
            return ok;
        }
    }
}


fn make_environment(settings: &Settings) -> Vec<(&str, Tensor)> {
    let mut dataset_path = Path::new(
        &settings.download.dataset_directory_path
    ).to_path_buf();

    // global price matrix
    let mut gpm = Vec::new();

    let mut check_sum: Option<usize> = None;
    let mut asset = 0;
    loop {
        dataset_path.push(asset.to_string());
        if !dataset_path.exists() { break; }

        let mut check_sum_for_this_asset = 0;
        let mut i = 0;

        loop {
            dataset_path.push(format!("{}.csv", i.to_string()));
            if !dataset_path.exists() {
                dataset_path.pop();
                break;
            }

            println!("extracting from {} ... ", dataset_path.to_str().unwrap());

            // read csv file

            let file = std::fs::File::open(&dataset_path).unwrap();
            let mut csv_reader = ReaderBuilder::new()
                .has_headers(settings.process.csv_header_exists)
                .from_reader(file);

            for result in csv_reader.records() {
                let record = result.unwrap();

                // on Binance, record format is provided here:
                //     https://www.binance.com/en/landing/data
                // we use close price of K-Line - Spot data
                let close_price = record[settings.process.price_index_in_csv_record]
                    .parse::<f64>()
                    .unwrap();
                gpm.push(close_price);
                check_sum_for_this_asset += 1;
            }

            dataset_path.pop();
            i += 1;
        }

        println!(
            "fetched all data from directory {} ({}entry)\n",
            asset, check_sum_for_this_asset,
        );

        // verify that the total number of CSV records is the same for each asset
        if check_sum.is_none() {
            check_sum = Some(check_sum_for_this_asset);
        } else {
            assert_eq!(check_sum_for_this_asset, check_sum.unwrap());
        }

        dataset_path.pop();
        asset += 1;
    }

    // make global price matrix Tensor
    let gpm = Tensor::cat(&[
        // add the price of the riskless asset (which is always 1)
        Tensor::from_slice(&vec![1.; check_sum.unwrap()]),
        Tensor::from_slice(&gpm)
    ], 0).reshape(&[(asset + 1) as i64, check_sum.unwrap() as i64]);

    // TEST: confirmed the gpm was correctly constructed
    // println!("{}", gpm);

    // generate local price matrices and price change rates
    let (lpm, pcr) = make_local_price_matrix(gpm, settings.process.window_size);

    // calculate sizes to divide dataset into training/validation/test data
    let base = settings.process.training_data_ratio
        + settings.process.validation_data_ratio
        + settings.process.test_data_ratio;
    let training_data_ratio = settings.process.training_data_ratio / base;
    let validation_data_ratio = settings.process.validation_data_ratio / base;
    let training_data_size = (lpm.size()[0] as f64 * training_data_ratio) as i64;
    let validation_data_size = (lpm.size()[0] as f64 * validation_data_ratio) as i64;
    let test_data_size = lpm.size()[0] as i64 - training_data_size - validation_data_size;

    // make mini-batches and price change matrices
    let (lpm_batches, pcr_matrices) = make_mini_batches(
        lpm.narrow(0, 0, training_data_size),
        pcr.narrow(0, 0, training_data_size),
        settings.process.window_size,
    );
    // println!("{}\n{}", lpm_batches, pcr_matrices);

    // divide dataset into training/validation/test data
    let validation_lpm = lpm.narrow(
        0, training_data_size, validation_data_size,
    );
    let validation_pcr = pcr.narrow(
        0, training_data_size, validation_data_size,
    );
    let test_lpm = lpm.narrow(
        0, training_data_size + validation_data_size, test_data_size,
    );
    let test_pcr = pcr.narrow(
        0, training_data_size + validation_data_size, test_data_size,
    );

    vec![
        // this type conversion is IMPORTANT
        ("lpm_batches", lpm_batches.totype(Kind::Float)),
        ("pcr_matrices", pcr_matrices.totype(Kind::Float)),
        ("validation_lpm", validation_lpm.totype(Kind::Float)),
        ("validation_pcr", validation_pcr.totype(Kind::Float)),
        ("test_lpm", test_lpm.totype(Kind::Float)),
        ("test_pcr", test_pcr.totype(Kind::Float)),
    ]
}

// returns local price matrix and price change vectors
fn make_local_price_matrix(gpm: Tensor, window_size: usize) -> (Tensor, Tensor) {
    let n_columns = gpm.size()[1] as usize;
    let mut lpm = Vec::new();
    let mut price_change_rate_t = Vec::new();

    for i in (window_size - 1)..(n_columns - 1) {
        let window = gpm.narrow(
            1, (i + 1 - window_size) as i64, window_size as i64,
        );
        let current_price = gpm.narrow(1, i as i64, 1);
        let next_price = gpm.narrow(1, i as i64 + 1, 1);
        let y = (&next_price / &current_price).flatten(0, -1);
        lpm.push(window / current_price);
        price_change_rate_t.push(y);
    }

    (Tensor::stack(&lpm, 0), Tensor::stack(&price_change_rate_t, 0))
}

#[test]
fn test_make_local_price_matrix() {
    let gpm = Tensor::from_slice(&[
        1., 1., 1., 1., 1., 2., 3., 4., 5., 6., 2., 5., 6., 8., 10.
    ]).reshape(&[3, 5]);
    let (lpm, pcr) = make_local_price_matrix(gpm, 3);
    println!("{}", lpm);
    println!("{}", pcr);
}

// returns an array of mini-batches and corresponding array of price change vectors
fn make_mini_batches(lpm: Tensor, pcr: Tensor, batch_size: usize) -> (Tensor, Tensor) {
    let batch_size = batch_size as i64;
    let (data_len, m, n) = (lpm.size()[0], lpm.size()[1], lpm.size()[2]);
    let adjusted_size = data_len - data_len % batch_size;
    let n_batch = adjusted_size / batch_size;

    let lpm = lpm.narrow(0, 0, adjusted_size);
    let pcr = pcr.narrow(0, 0, adjusted_size);
    (lpm.view([n_batch, batch_size, m, n]),
     pcr.view([n_batch, batch_size, m]))
}

#[test]
fn test_make_mini_batches() {
    let v = &(0..36).collect::<Vec<_>>();
    let lpm = Tensor::from_slice(&v).view([9, 2, 2]);

    let r = &(0..18).collect::<Vec<_>>();
    let pcr = Tensor::from_slice(&r).view([9, 2]);

    println!("{}\n{}", lpm, pcr);

    let (lpm_batch, pcr_batch) = make_mini_batches(lpm, pcr, 4);
    println!("{}\n{}", lpm_batch, pcr_batch);
}
