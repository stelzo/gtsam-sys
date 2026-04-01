extern crate bindgen;
extern crate pkg_config;

use std::{env, path::PathBuf};

/// Tries to use system gtsam and emits necessary build script instructions.
/// Credits to https://github.com/rust-lang/git2-rs/blob/master/libgit2-sys/build.rs
fn try_system_gtsam() -> Result<pkg_config::Library, pkg_config::Error> {
    let mut cfg = pkg_config::Config::new();
    match cfg.atleast_version("4.2").probe("gtsam") {
        Ok(lib) => {
            for include in &lib.include_paths {
                println!("cargo:root={}", include.display());
            }
            Ok(lib)
        }
        Err(e) => Err(e),
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PURE_RUST_NAVIGATION");

    if env::var_os("CARGO_FEATURE_PURE_RUST_NAVIGATION").is_some() {
        return;
    }

    println!("cargo:rerun-if-changed=gtsam/gtsam_imu.cpp");
    println!("cargo:rerun-if-changed=gtsam/gtsam_imu.hpp");
    println!("cargo:rerun-if-changed=gtsam/CMakeLists.txt");

    let bindings = bindgen::Builder::default()
        .header("gtsam/gtsam_imu.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");

    let using_system_gtsam = try_system_gtsam().is_ok();

    let mut cmake_cfg = cmake::Config::new("gtsam");
    if using_system_gtsam {
        cmake_cfg.build_target("gtsam_imu");
    } else {
        cmake_cfg.build_target("gtsam_sys");
    }
    let cmake_out = cmake_cfg.build();

    if !using_system_gtsam {
        println!(
            "cargo:rustc-link-search=native={}/build/gtsam/gtsam",
            cmake_out.display()
        );
    }

    println!("cargo:rustc-link-search=native={}/build", cmake_out.display());
    if using_system_gtsam {
        println!("cargo:rustc-link-lib=gtsam_imu");
    } else {
        println!("cargo:rustc-link-lib=gtsam_sys");
    }
    println!("cargo:rustc-link-lib=gtsam");
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
