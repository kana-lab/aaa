use std::collections::HashMap;
use std::io::Write;
use std::str::FromStr;
use std::time::Duration;
use ring::{digest, hmac};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::json;
use tch::nn::{ModuleT, VarStore};
use tch::{Device, Kind, Tensor};
use tokio::time;
use aaa::net::Net;

const BASE_URL: &str = "https://api-testnet.bybit.com";
const BASE_URL_REAL: &str = "https://api.bybit.com";
const API_KEY: &str = "iplEKKUVGfVa9MmgII";
const API_SECRET: &[u8] = b"giEcmdoorNOcBsmr1wIdwwxLfwD60ZwGsKvw";
const ASSETS: [&str; 8] = [
    "USDT", "ADA", "BNB", "BTC", "DOGE",
    "ETH", "TRX", "XRP"
];
const WINDOW_SIZE: usize = 50;

#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    retCode: i32,
    retMsg: String,
    result: T,
    time: u64,
}

#[derive(Debug, Deserialize)]
struct LastTradedPriceV3 {
    symbol: String,
    price: String,
}

impl LastTradedPriceV3 {
    fn price(self: &Self) -> f64 {
        self.price.parse::<f64>().unwrap()
    }
}

#[derive(Debug, Deserialize)]
struct KLineV5 {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

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

// todo: error handling
// async fn get_spot_price(symbol: &str) -> Result<BybitResponse<LastTradedPriceV3>, reqwest::Error> {
//     let url = format!(
//         "{}/spot/v3/public/quote/ticker/price?symbol={}", BASE_URL, symbol
//     );
//     let response = reqwest::get(url).await?;
//     println!("Status: {}", response.status());
//     let body = response.text().await?;
//     println!("Body:\n{}", body);
//     Ok(serde_json::from_str::<BybitResponse<LastTradedPriceV3>>(&body).unwrap())
// }

async fn get_spot_price(symbol: &str) -> Result<BybitResponse<KLineV5>, reqwest::Error> {
    let url = format!(
        "{}/v5/market/kline?category=spot&symbol={}&interval=30&limit={}",
        BASE_URL_REAL, symbol, WINDOW_SIZE
    );
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    // println!("Body:\n{}", body);
    Ok(serde_json::from_str(&body).unwrap())
}

fn gen_signature(data: &str) -> (String, String) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string();

    let key = hmac::Key::new(hmac::HMAC_SHA256, API_SECRET);
    let sign = hmac::sign(
        &key,
        format!("{}{}{}", timestamp, API_KEY, data).as_bytes(),
    );

    (timestamp, hex::encode(sign.as_ref()))
}

// todo: error handling
// todo: consider using reqwest::blocking
// qty should be given as USDT when BUY, otherwise as BTC etc.
async fn place_spot_order(
    symbol: &str, qty: f64, decimals: usize,
) -> Result<BybitResponse<PlaceOrderResultV5>, reqwest::Error> {
    let url = format!("{}/v5/order/create", BASE_URL);

    if qty.abs() <= 0.0001 {
        return Ok(BybitResponse {
            retCode: 0,
            retMsg: "".to_string(),
            result: PlaceOrderResultV5 { orderId: None, orderLinkId: None },
            time: 0,
        });
    }

    let data = json!({
        "category": "spot",
        "symbol": symbol.to_string(),
        "side": if qty > 0. {"Buy"} else {"Sell"},
        "orderType": "Market",
        "qty": qty.abs().to_string()
    });

    let (timestamp, sign) = gen_signature(&data.to_string());

    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("X-BAPI-SIGN", sign)
        .header("X-BAPI-API-KEY", API_KEY)
        .header("X-BAPI-TIMESTAMP", timestamp)
        .json(&data)
        .send()
        .await?;

    println!("Status: {}", response.status());
    let body = response.text().await?;
    println!("Response body:\n{}", body);
    Ok(serde_json::from_str(&body).unwrap())
}

async fn get_asset_info() -> Result<BybitResponse<AssetInfoV5>, reqwest::Error> {
    let url = format!("{}/v5/asset/transfer/query-asset-info?accountType=SPOT", BASE_URL);
    let (timestamp, sign) = gen_signature("accountType=SPOT");

    let response = reqwest::Client::new()
        .get(url)
        .header("X-BAPI-SIGN", sign)
        .header("X-BAPI-API-KEY", API_KEY)
        .header("X-BAPI-TIMESTAMP", timestamp)
        .send()
        .await?;

    println!("Status: {}", response.status());
    let body = response.text().await?;
    println!("Response body:\n{}", body);
    Ok(serde_json::from_str(&body).unwrap())
}

async fn get_asset_vec() -> Tensor {
    let a_info = get_asset_info().await.unwrap();
    let mut dict = HashMap::new();
    for (idx, &asset) in ASSETS.iter().enumerate() {
        dict.insert(asset.to_string(), idx);
    }

    let mut v = vec![0.; ASSETS.len()];
    for asset in a_info.result.spot.assets {
        v[*dict.get(&asset.coin).unwrap()] = f64::from_str(&asset.free).unwrap();
    }

    let t = Tensor::from_slice(&v);
    t.detach().totype(Kind::Float)
}

async fn get_price_mat() -> (Tensor, Tensor) {
    let mut v = vec![1.];
    let mut lpm = vec![Tensor::from_slice(&vec![1.; WINDOW_SIZE])];
    for &asset in &ASSETS[1..] {
        let p = get_spot_price(&format!("{}USDT", asset)).await.unwrap();
        let current_price = f64::from_str(&p.result.list[0][4]).unwrap();
        v.push(current_price);

        let mut history = vec![];
        for price in p.result.list.iter().rev() {
            history.push(
                f64::from_str(&price[4]).unwrap() / current_price
            );
        }
        lpm.push(Tensor::from_slice(&history));
        // println!("{} {:?}", asset, history);
    }

    let t = Tensor::from_slice(&v);
    (
        Tensor::stack(&lpm, 0).detach().totype(Kind::Float),
        t.detach().totype(Kind::Float)
    )
}

async fn one_step(net: &Net, device: Device) {
    let asset = get_asset_vec().await.to(device);
    println!("asset: {}", asset);
    let (lpm, price) = get_price_mat().await;
    let (lpm, price) = (lpm.to(device), price.to(device));
    println!("price: {}", price);

    let total_usdt = asset.dot(&price);
    let w = net.forward_t(&lpm, false);
    let asset_new = &w * &total_usdt;
    println!("w:{}\ntotal_usdt:{}\nasset_new{}", w, total_usdt, asset_new);
    let diff = (&w * &total_usdt - &asset * &price).flatten(0, 1);
    println!("diff: {}", diff);
    for (i, &asset_name) in ASSETS.iter().enumerate() {
        if i == 0 || asset_name == "DOGE" { continue; }

        println!("asset_name: {}", asset_name);
        let mut qty = f64::try_from(diff.get(i as i64)).unwrap();
        qty *= 0.995;
        if qty < 0. {
            qty /= f64::try_from(price.get(i as i64)).unwrap();
            let di = if asset_name == "BTC" { 1e4 } else if asset_name == "ADA" { 1e1 } else { 1e2 };
            qty = (qty * di).round() / di;
        } else {
            qty = qty.round();
        }
        println!("raw_qty: {}", qty);
        place_spot_order(
            &format!("{}USDT", asset_name),
            qty,
            if qty < 0. { 4 } else { 0 },
        ).await.unwrap();
    }
}

#[tokio::main]
async fn main() {
    // let a = get_spot_price("BTCUSDT").await.unwrap();
    // println!("{:?}", a);
    // let a = place_spot_order("BNBUSDT", 98., 4).await.unwrap();
    // println!("{:?}", a);
    // let a = get_asset_info().await.unwrap();
    // let a = get_price_mat().await;

    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);

    let mut vs = VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root());
    vs.load("./model.safetensors").unwrap();

    let interval = Duration::from_secs(1800);
    let mut interval_timer = time::interval(interval);
    loop {
        interval_timer.tick().await;
        one_step(&net, device).await;
    }
}