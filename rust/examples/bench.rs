// Rough end-to-end benchmark: parse + execute the vector_add kernel repeatedly.
// Run: cargo run --release --example bench
use ktir_cpu::dtypes::DType;
use ktir_cpu::interpreter::{execute_function, Arg};
use ktir_cpu::ir::Scalar;
use ktir_cpu::parser::parse_module;
use std::time::Instant;

fn main() {
    let src = include_str!("../../examples/triton-ktir/vector_add_ktir.mlir");
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();

    let iters = 2000;

    // Time parse alone.
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(parse_module(src).unwrap());
    }
    let parse_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // Time parse + execute (the full pipeline, 32-core grid).
    let t1 = Instant::now();
    for _ in 0..iters {
        let module = parse_module(src).unwrap();
        let args = [
            ("x_ptr", Arg::Tensor { data: x.clone(), shape: vec![n], dtype: DType::F16 }),
            ("y_ptr", Arg::Tensor { data: y.clone(), shape: vec![n], dtype: DType::F16 }),
            ("output_ptr", Arg::Tensor { data: vec![0.0; n], shape: vec![n], dtype: DType::F16 }),
            ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(128))),
        ];
        let out = execute_function(&module, "add_kernel", &args).unwrap();
        std::hint::black_box(&out);
    }
    let full_ns = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!("iters            : {iters}");
    println!("parse only       : {:.1} us/iter", parse_ns / 1000.0);
    println!("parse + execute  : {:.1} us/iter", full_ns / 1000.0);
    println!("execute only     : {:.1} us/iter", (full_ns - parse_ns) / 1000.0);
}
