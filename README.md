# gtsam-rs Workspace (in migration)

This repository is transitioning from a single `gtsam-sys` FFI crate to a pure-Rust GTSAM workspace.

Current state:
- `gtsam-sys` remains available and defaults to pure-Rust navigation (no C++ build in default configuration).
- A pure-Rust navigation backend is available behind `gtsam-sys` feature `pure-rust-navigation`.
- A C++ oracle path remains available for differential checks by disabling default features.
- New crates exist for math, inference, linear, nonlinear, navigation, slam, discrete, compat, and top-level facade `gtsam-rs`.

## Crates

- `gtsam-rs`: top-level Rust-native API facade
- `gtsam-rs-math`: Lie groups/manifolds and geometry primitives (`Rot3`, `Pose3`, etc.)
- `gtsam-rs-inference`: `Key`, `Symbol`, values, factor graph containers
- `gtsam-rs-linear`: Gaussian factor graph abstractions
- `gtsam-rs-nonlinear`: nonlinear factor graph + optimizer scaffolding
- `gtsam-rs-navigation`: pure-Rust IMU/navigation scaffolding
- `gtsam-rs-slam`: SLAM factor placeholders
- `gtsam-rs-discrete`: discrete graph placeholder
- `gtsam-rs-compat`: compatibility naming layer
- `gtsam-sys`: legacy bridge + migration wrapper

## Build examples

Default Rust-only backend:

```bash
cargo check -p gtsam-sys
```

Pure-Rust IMU backend tests through `gtsam-sys`:

```bash
cargo test -p gtsam-sys --features pure-rust-navigation pure_rust_backend
```

C++ oracle differential test:

```bash
cargo test -p gtsam-sys --no-default-features navigation_matches_cpp_oracle_on_static_dataset_window
```

Full workspace:

```bash
cargo check --workspace
```

Run linear benchmark gate (ignored by default):

```bash
cargo test -p gtsam-rs-linear benchmark_gate_sparse_vs_dense_chain -- --ignored
```

## License

Licensed under either of:
- Apache License, Version 2.0
- MIT license

at your option.
