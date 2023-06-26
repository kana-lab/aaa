use std::io;
use std::io::{Read, Write};

pub fn input(msg: &str) -> String {
    let stdin = io::stdin();
    let mut input = String::new();
    print!("{}", msg);
    io::stdout().flush().expect("Failed to flush stdout.");
    stdin.read_line(&mut input)
        .expect("Failed to read input.");
    input.trim().to_string()
}

pub fn input_bool(msg: &str, default: bool) -> bool {
    let mut stdin = io::stdin();
    let mut input = String::new();
    let annotation = if default { "(Y/n)" } else { "(y/N)" };
    loop {
        print!("{} {} ", msg, annotation);
        io::stdout().flush().expect("Failed to flush stdout.");

        input.clear();
        let n = stdin.read_line(&mut input)
            .expect("Failed to read input.");
        if n == 1 {
            return default;
        }

        let c = input.trim().chars().nth(0).unwrap();
        if c == 'y' || c == 'Y' {
            return true;
        } else if c == 'n' || c == 'N' {
            return false;
        }
    }
}

#[test]
fn input_test() {
    let s = input("");
    println!("result: {}", s);
}

#[test]
fn input_bool_test() {
    let c = input_bool("hello?", true);
    println!("result: {}", c);
}