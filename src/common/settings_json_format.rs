use std::fs::File;
use serde::Deserialize;
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct DownloadSettings {
    pub download_script_path: String,
    pub dataset_directory_path: String,
    pub asset_names: Vec<String>,
    pub num_months: usize,
}

#[derive(Debug, Deserialize)]
pub struct ProcessSettings {
    pub price_index_in_csv_record: usize,
    pub csv_header_exists: bool,
    pub environment_files_dir: String,
    pub default_environment_file_name: String,
    pub training_data_ratio: f64,
    pub validation_data_ratio: f64,
    pub test_data_ratio: f64,
    pub window_size: usize,
    pub batch_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct TrainSettings {
    pub learning_rate: f64,
    pub epoch: usize,
    pub model_files_dir: String,
    pub default_model_file_name: String,
}

#[derive(Debug, Deserialize)]
pub struct BybitSettings {
    pub asset_names: Vec<String>,
    pub trade_interval_sec: usize,
    pub api_key: String,
    pub api_secret: String,
}

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub download: DownloadSettings,
    pub process: ProcessSettings,
    pub train: TrainSettings,
    pub bybit: BybitSettings,
}

impl Settings {
    pub fn load() -> Result<Self> {
        let f = File::open("settings.json")?;
        let deserialized: Self = serde_json::from_reader(f)?;
        Ok(deserialized)
    }
}


#[test]
fn load_test() {
    let settings = Settings::load().unwrap();
    println!("{:?}", settings);
}