extern crate bindgen;

use std::{env, path::PathBuf};

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

    println!(
        "cargo:rustc-link-search=native={}/build",
        out_path.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/gtsam/gtsam",
        out_path.display()
    );
    println!("cargo:rustc-link-lib=static=gtsam_imu");
    println!("cargo:rustc-link-lib=gtsam");
    println!("cargo:rustc-link-lib=stdc++");
}
