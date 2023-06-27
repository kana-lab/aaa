use std::io;
use std::io::{Read, Write};

pub fn input(msg: &str) -> String {
    let stdin = io::stdin();
    let mut input = String::new();

    print!("{}", msg);
    if let Err(e) = io::stdout().flush() {
        eprintln!("Failed to flush stdout: {}", e);
        std::process::exit(1);
    }

    if let Err(e) = stdin.read_line(&mut input) {
        eprintln!("Failed to read input: {}", e);
        std::process::exit(1);
    }
    input.trim().to_string()
}

pub fn input_bool(msg: &str, default: bool) -> bool {
    let mut stdin = io::stdin();
    let mut input = String::new();
    let annotation = if default { "(Y/n)" } else { "(y/N)" };
    loop {
        print!("{} {} ", msg, annotation);
        if let Err(e) = io::stdout().flush() {
            eprintln!("Failed to flush stdout: {}", e);
            std::process::exit(1);
        }

        input.clear();
        let n = stdin.read_line(&mut input);
        let n = if let Err(e) = n {
            eprintln!("Failed to read input: {}", e);
            std::process::exit(1);
        } else {
            n.unwrap()
        };

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