//! Regenerates the C header from the exported `extern "C"` items.

use std::{env, path::PathBuf};

fn main() {
    let config = cbindgen::Config::from_file("cbindgen.toml").expect("cbindgen.toml parses");
    let bindings = cbindgen::Builder::new()
        .with_crate(env::var("CARGO_MANIFEST_DIR").expect("manifest dir is set"))
        .with_config(config)
        .generate()
        .expect("bindings generate");
    let header = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir is set"))
        .join("include")
        .join("flare.h");
    bindings.write_to_file(header);
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/c_abi.rs");
}
