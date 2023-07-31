use tch::{Device, Kind, Tensor};

fn main() {
    println!(r#"
Auto Asset Allocator.

To download data and preprocess it:
$ cargo run --bin preprocess

To train a model:
$ cargo run --bin train

To auto-trade on Bybit:
$ cargo run --bin bybit

Please also see `settings.json`.
"#);
}

#[test]
fn t(){
    let v = vec![1,2,3,4,5,6];
    let t = Tensor::from_slice(&v);
    println!("{:?}", t);
    let a = t.split_sizes([1,5], 0);
    println!("{:?} {:?}", a[0], a[1]);
    let three = Tensor::full([6],3, (Kind::Int, Device::Cpu));
    let vv = t.gt_tensor(&three).totype(Kind::Float);
    let vv = t.gt(2).nonzero().flatten(0, -1);
    let vvv = t.index_select(0, &vv);
    println!("{:?}", vvv);
    println!("{:?}", vvv.cumprod(0, Kind::Int));

    let v = vec![1,2,3,4];
    let t = Tensor::from_slice(&v).reshape(&[2,2]);
    let t = t.cumprod(0,Kind::Int);
    println!("{:?}", Vec::<Vec<f64>>::try_from(t).unwrap());
}

#[test]
fn t2(){
    let v = Tensor::from_slice(&[1,2,3]);
    let w = Tensor::from_slice(&[4,5,6]);
    let u = Tensor::stack(&[v,w],0);
    println!("{:?}", u);
}