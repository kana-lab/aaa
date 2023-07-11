use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;
use futures::future;
use ring::hmac;
use serde::Deserialize;
use serde_json::json;
use tch::nn::{ModuleT, VarStore};
use tch::{Device, Kind, Tensor};
use tokio::time;
use aaa::net::Net;
use aaa::settings_json_format::Settings;
use crate::RequestError::{BybitError, JsonError, NetworkError};

// const BASE_URL: &str = "https://api-testnet.bybit.com";
const BASE_URL: &str = "https://api.bybit.com";
const MARKET_URL: &str = "https://api.bybit.com";

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    retCode: i32,
    retMsg: String,
    result: T,
    time: u64,
}

#[derive(Debug, Deserialize)]
struct KLineV5 {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
struct PlaceOrderResultV5 {
    orderId: Option<String>,
    orderLinkId: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AssetInfoV5 {
    spot: AssetInfoInnerV5,
}

#[derive(Debug, Deserialize)]
struct AssetInfoInnerV5 {
    status: String,
    assets: Vec<IndividualAssetInfo>,
}

#[derive(Debug, Deserialize)]
struct IndividualAssetInfo {
    coin: String,
    frozen: String,
    free: String,
    withdraw: String,
}

#[derive(Debug)]
enum RequestError {
    NetworkError(reqwest::Error),
    JsonError(serde_json::Error),
    BybitError(i32, String),
}

fn gen_signature(data: &str, settings: &Settings) -> (String, String) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string();

    let key = hmac::Key::new(hmac::HMAC_SHA256, settings.bybit.api_secret.as_bytes());
    let sign = hmac::sign(
        &key,
        format!("{}{}{}", timestamp, &settings.bybit.api_key, data).as_bytes(),
    );

    (timestamp, hex::encode(sign.as_ref()))
}

// qty should be given as USDT when BUY, otherwise as BTC etc.
async fn place_spot_order(
    symbol: String, qty: f64, settings: &Settings,
) -> Result<(), RequestError> {
    let url = format!("{}/v5/order/create", BASE_URL);

    let data = json!({
        "category": "spot",
        "symbol": symbol,
        "side": if qty > 0. {"Buy"} else {"Sell"},
        "orderType": "Market",
        "qty": qty.abs().to_string()
    });

    let (timestamp, sign) = gen_signature(&data.to_string(), settings);

    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("X-BAPI-SIGN", sign)
        .header("X-BAPI-API-KEY", &settings.bybit.api_key)
        .header("X-BAPI-TIMESTAMP", timestamp)
        .json(&data)
        .send()
        .await;

    let body = match response {
        Err(e) => { return Err(NetworkError(e)); }
        Ok(ok) => ok
    };

    println!("Status: {}", body.status());
    let body = body.text().await;
    let order = match body {
        Err(e) => { return Err(NetworkError(e)); }
        Ok(ok) => ok
    };

    println!("Response body:\n{}", order);
    let order = serde_json::from_str::<BybitResponse<PlaceOrderResultV5>>(&order);
    if let Err(e) = order {
        return Err(JsonError(e));
    }

    Ok(())
}

async fn get_asset_info(settings: &Settings) -> Result<AssetInfoV5, RequestError> {
    let url = format!("{}/v5/asset/transfer/query-asset-info?accountType=SPOT", BASE_URL);
    let (timestamp, sign) = gen_signature("accountType=SPOT", settings);

    let response = reqwest::Client::new()
        .get(url)
        .header("X-BAPI-SIGN", sign)
        .header("X-BAPI-API-KEY", &settings.bybit.api_key)
        .header("X-BAPI-TIMESTAMP", timestamp)
        .send()
        .await;
    let response = match response {
        Err(e) => { return Err(NetworkError(e)); }
        Ok(ok) => ok
    };

    println!("Status: {}", response.status());
    let body = response.text().await;
    let body = match body {
        Err(e) => { return Err(NetworkError(e)); }
        Ok(ok) => ok
    };

    println!("Response body:\n{}", body);
    let asset_info = serde_json::from_str::<BybitResponse<AssetInfoV5>>(&body);
    match asset_info {
        Err(e) => Err(JsonError(e)),
        Ok(info) => Ok(info.result)
    }
}

async fn get_asset_vec(settings: &Settings) -> Result<Tensor, RequestError> {
    let a_info = get_asset_info(settings).await?;
    let mut dict = HashMap::new();
    for (idx, asset) in settings.bybit.asset_names.iter().enumerate() {
        dict.insert(asset.clone(), idx);
    }

    let mut v = vec![0.; settings.bybit.asset_names.len()];
    for asset in a_info.spot.assets {
        v[*dict.get(&asset.coin).unwrap()] = f64::from_str(&asset.free).unwrap();
    }

    let t = Tensor::from_slice(&v);
    Ok(t.detach().totype(Kind::Float))
}

async fn get_spot_price(symbol: &str, window_size: usize) -> Result<KLineV5, RequestError> {
    let url = format!(
        "{}/v5/market/kline?category=spot&symbol={}&interval=30&limit={}",
        MARKET_URL, symbol, window_size
    );

    let response = reqwest::get(url).await;
    let response = match response {
        Err(e) => { return Err(NetworkError(e)); }
        Ok(ok) => ok
    };

    let body = response.text().await;
    let body = match body {
        Err(e) => { return Err(NetworkError(e)); }
        Ok(ok) => ok
    };

    let parsed = serde_json::from_str::<BybitResponse<KLineV5>>(&body);
    match parsed {
        Err(e) => Err(JsonError(e)),
        Ok(body) => {
            if body.retCode != 0 {
                Err(BybitError(body.retCode, body.retMsg))
            } else {
                Ok(body.result)
            }
        }
    }
}

async fn get_price_mat(settings: &Settings) -> Result<(Tensor, Tensor), RequestError> {
    let mut tasks = Vec::new();
    for asset in &settings.bybit.asset_names[1..] {
        let window_size = settings.preprocess.window_size;
        let symbol = format!("{}USDT", asset);
        let task = async move {
            let p = get_spot_price(&symbol, window_size).await;
            let p = match p {
                Ok(ok) => ok,
                Err(e) => { return Err(e); }
            };
            let current_price = f64::from_str(&p.list[0][4]).unwrap();

            let mut history = vec![];
            for price in p.list.iter().rev() {
                history.push(
                    f64::from_str(&price[4]).unwrap() / current_price
                );
            }
            // println!("{} {:?}", asset, history);
            Ok((current_price, Tensor::from_slice(&history)))
        };
        tasks.push(tokio::spawn(task));
    }

    let result = future::join_all(tasks.into_iter()).await;

    let mut v = vec![1.];
    let mut lpm = vec![Tensor::from_slice(&vec![1.; settings.preprocess.window_size])];
    for r in result {
        let (current_price, history) = r.unwrap()?;
        v.push(current_price);
        lpm.push(history);
    }

    let t = Tensor::from_slice(&v);
    Ok((
        Tensor::stack(&lpm, 0).detach().totype(Kind::Float),
        t.detach().totype(Kind::Float)
    ))
}

fn round_qty(qty: f64, asset_name: &str) -> Option<f64> {
    fn round_by_floor(val: f64, pow: f64) -> f64 {
        f64::floor(val * pow) / pow
    }

    if qty > 0. {
        match asset_name {
            "XRP" => if qty < 1. { None } else { Some(round_by_floor(qty, 1e6)) },
            "BTC" => if qty < 1. { None } else { Some(round_by_floor(qty, 1e8)) }
            "ETH" => if qty < 1. { None } else { Some(round_by_floor(qty, 1e7)) },
            "BNB" => if qty < 1. { None } else { Some(round_by_floor(qty, 1e9)) },
            "DOGE" => if qty < 1. { None } else { Some(round_by_floor(qty, 1e6)) },
            "ADA" => if qty < 1. { None } else { Some(round_by_floor(qty, 1e6)) },
            "TRX" => if qty < 1. { None } else { Some(round_by_floor(qty, 1e7)) },
            _ => unreachable!()
        }
    } else {
        let qty_abs = qty.abs();
        match asset_name {
            "XRP" => if qty_abs < 2.63 { None } else { Some(-round_by_floor(qty_abs, 1e2)) },
            "BTC" => if qty_abs < 0.000048 { None } else { Some(-round_by_floor(qty_abs, 1e6)) },
            "ETH" => if qty_abs < 0.00062 { None } else { Some(-round_by_floor(qty_abs, 1e5)) },
            "BNB" => if qty_abs < 0.00262 { None } else { Some(-round_by_floor(qty_abs, 1e5)) },
            "DOGE" => if qty_abs < 6.9 { None } else { Some(-round_by_floor(qty_abs, 1e1)) },
            "ADA" => if qty_abs < 1.86 { None } else { Some(-round_by_floor(qty_abs, 1e2)) },
            "TRX" => if qty_abs < 14.14 { None } else { Some(-round_by_floor(qty_abs, 1e2)) },
            _ => unreachable!()
        }
    }
}

async fn one_step(net: &Net, device: Device, settings: &Settings) -> Result<(), RequestError> {
    let (asset, price_mat) = tokio::join!(
        get_asset_vec(settings), get_price_mat(settings)
    );
    let asset = asset?;
    let (lpm, price) = price_mat?;
    let (lpm, price) = (lpm.to(device), price.to(device));
    println!("asset: {}", asset);
    println!("price: {}", price);

    let total_usdt = asset.dot(&price);
    let w = net.forward_t(&lpm, false);
    let asset_new = &w * &total_usdt;
    println!("w:{}\ntotal_usdt:{}\nasset_new{}", w, total_usdt, asset_new);
    let diff = (&w * &total_usdt - &asset * &price).flatten(0, 1);

    let diff = Vec::<f64>::try_from(diff).unwrap();
    let price = Vec::<f64>::try_from(price).unwrap();
    println!("diff: {:?}", diff);

    let mut sell_orders = Vec::new();
    for (i, asset_name) in settings.bybit.asset_names.iter().enumerate() {
        if i == 0 || diff[i] > 0. { continue; }

        println!("asset_name: {}", asset_name);
        let qty = diff[i] / price[i];
        let qty = round_qty(qty, asset_name);
        println!("qty: {:?} {}", qty, asset_name);
        if qty.is_none() { continue; }
        let qty = qty.unwrap();
        sell_orders.push(place_spot_order(
            format!("{}USDT", asset_name), qty, settings,
        ));
    }
    let r = future::join_all(sell_orders).await;

    let mut buy_orders = Vec::new();
    for (i, asset_name) in settings.bybit.asset_names.iter().enumerate() {
        if i == 0 || diff[i] <= 0. { continue; }

        println!("asset_name: {}", asset_name);
        let qty = round_qty(diff[i], asset_name);
        println!("qty: {:?} USDT", qty);
        if qty.is_none() { continue; }
        let qty = qty.unwrap();
        buy_orders.push(place_spot_order(
            format!("{}USDT", asset_name), qty, settings,
        ));
    }
    let r = future::join_all(buy_orders).await;

    Ok(())
}

#[tokio::main]
async fn main() {
    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);

    let settings = match Settings::load() {
        Ok(ok) => ok,
        Err(e) => {
            eprintln!("could not read settings.json: {}", e);
            std::process::exit(1);
        }
    };

    let mut vs = VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root(), &settings);

    let mut env_file_path = Path::new(
        &settings.train.model_files_dir
    ).to_path_buf();
    env_file_path.push(Path::new(
        &settings.train.default_model_file_name
    ));
    if let Err(e) = vs.load(env_file_path) {
        eprintln!("Failed to load model file: {}", e);
        std::process::exit(1);
    }

    let interval = Duration::from_secs(1800);
    let mut interval_timer = time::interval(interval);
    loop {
        interval_timer.tick().await;
        one_step(&net, device, &settings).await;
    }
}