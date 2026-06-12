// End-to-end execution: parse a real KTIR kernel and run it through the full
// driver (HBM marshalling -> multi-core execution -> read-back), checking the
// computed output. This is the parity checkpoint — the interpreter actually
// runs a kernel and produces correct tensor results.

use ktir_cpu::dtypes::DType;
use ktir_cpu::interpreter::{execute_function, Arg, Output};
use ktir_cpu::parser::parse_module;

#[test]
fn vector_add_executes_end_to_end() {
    let src = include_str!("../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).expect("parse vector_add");

    // 32 cores x BLOCK_SIZE=128 = 4096 elements, matching the kernel's views.
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let out = vec![0.0f32; n];

    let args = [
        ("x_ptr", Arg::Tensor { data: x.clone(), shape: vec![n], dtype: DType::F16 }),
        ("y_ptr", Arg::Tensor { data: y.clone(), shape: vec![n], dtype: DType::F16 }),
        ("output_ptr", Arg::Tensor { data: out, shape: vec![n], dtype: DType::F16 }),
        ("BLOCK_SIZE", Arg::Scalar(ktir_cpu::ir::Scalar::I64(128))),
    ];

    let outputs = execute_function(&module, "add_kernel", &args).expect("run add_kernel");
    let Output { data, .. } = outputs.get("output_ptr").expect("output_ptr present");

    let expected: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_eq!(data.len(), n);
    assert_eq!(*data, expected, "elementwise x + y mismatch");
}
