extern crate bindgen;
extern crate pkg_config;

use std::{env, path::PathBuf};

/// Tries to use system gtsam and emits necessary build script instructions.
/// Credits to https://github.com/rust-lang/git2-rs/blob/master/libgit2-sys/build.rs
fn try_system_gtsam() -> Result<pkg_config::Library, pkg_config::Error> {
    let mut cfg = pkg_config::Config::new();
    match cfg.exactly_version("4.2").probe("gtsam") {
        Ok(lib) => {
            for include in &lib.include_paths {
                println!("cargo:root={}", include.display());
            }
            Ok(lib)
        }
        Err(e) => {
            // println!("cargo:warning=failed to probe system gtsam: {e}");
            Err(e)
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=gtsam/gtsam_imu.cpp");
    println!("cargo:rerun-if-changed=gtsam/gtsam_imu.hpp");
    println!("cargo:rerun-if-changed=gtsam/CMakeLists.txt");

    let bindings = bindgen::Builder::default()
        .header("gtsam/gtsam_imu.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    cmake::build("gtsam");

    match try_system_gtsam() {
        Ok(_) => {}
        Err(_) => {
            println!(
                "cargo:rustc-link-search=native={}/build/gtsam/gtsam",
                out_path.display()
            );
        }
    }

    println!(
        "cargo:rustc-link-search=native={}/build",
        out_path.display()
    );
    println!("cargo:rustc-link-lib=static=gtsam_imu");
    println!("cargo:rustc-link-lib=gtsam");
    println!("cargo:rustc-link-lib=stdc++");
}
