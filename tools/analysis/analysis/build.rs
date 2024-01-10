#![feature(iter_intersperse)]

use std::{env, fs, path::Path};

use glob::glob;
use regex::Regex;

fn main() -> Result<(), String> {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let src_dir = env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let src_path = Path::new(&src_dir).join("src");

    let pattern = src_path
        .join("**/*.rs")
        .to_str()
        .ok_or("Could not convert source path to string".to_string())?
        .to_string();

    let rx = Regex::new(r"(?s)\banalysis!\s*[({\[]\s*name\s*:\s*(?P<name>.*?),").unwrap();

    let mut funcs = Vec::new();

    for entry in glob(&pattern).expect("Could not glob") {
        let path = entry.map_err(|_e| "could not glob".to_string())?;
        let rel_path = path
            .strip_prefix(src_path.clone())
            .map_err(|_e| "Could not rebase path".to_string())?;

        let content = fs::read_to_string(path.clone()).expect("Could not read file");

        let mod_: String = rel_path
            .components()
            .map(|x| {
                Path::new(&x)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .intersperse("::".to_string())
            .collect();

        funcs.extend(
            rx.captures_iter(&content)
                .map(|c| "crate::".to_owned() + &mod_ + "::" + &c["name"].trim()),
        );
    }

    let dest_path = Path::new(&out_dir).join("analyses_list.rs");
    // let content = format!("build_analyses_descriptors!{{ {} }}", funcs.join(", "));
    let content = format!("build_analyses_descriptors!{{ {} }}", funcs.join(", "));
    // println!("generated: {}", content);
    fs::write(dest_path, content).unwrap();
    Ok(())
}
