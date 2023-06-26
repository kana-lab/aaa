use std::path::Path;
use log::error;
use aaa::common::{input_bool, Settings};

fn main() {
    let settings = match Settings::load() {
        Ok(settings) => settings,
        Err(e) => {
            error!("could not read settings.json.");
            error!("{}", e);
            return;
        }
    };

    // todo: environment.safetensors check & rename

    let dataset_path = Path::new(&settings.download.dataset_directory_path);
    let download = if dataset_path.exists() {
        println!("dataset does already exist.");
        let b = input_bool("would you like to re-download?", false);
        if b {
            println!("\nnew dataset will be downloaded.");
            // todo: rm; mkdir
        } else {
            println!("\nexisting dataset will be used.")
        }
        b
    } else {
        println!("dataset does NOT exist.");
        let b = input_bool("would you like to download?", true);
        if b {
            println!("\nnew dataset will be downloaded.");
            // todo: mkdir
        } else {
            println!("\nabort.");
            return;
        }
        b
    };
}