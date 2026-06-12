// End-to-end execution: parse a real KTIR kernel and run it through the full
// driver (HBM marshalling -> multi-core execution -> read-back), checking the
// computed output. This is the parity checkpoint — the interpreter actually
// runs a kernel and produces correct tensor results.

use ktir_cpu::dtypes::DType;
use ktir_cpu::interpreter::{execute_function, execute_function_with_latency, Arg, Output};
use ktir_cpu::latency::HardwareConfig;
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

#[test]
fn vector_add_latency_report_is_populated() {
    let src = include_str!("../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).expect("parse vector_add");
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let args = [
        ("x_ptr", Arg::Tensor { data: x, shape: vec![n], dtype: DType::F16 }),
        ("y_ptr", Arg::Tensor { data: y, shape: vec![n], dtype: DType::F16 }),
        ("output_ptr", Arg::Tensor { data: vec![0.0; n], shape: vec![n], dtype: DType::F16 }),
        ("BLOCK_SIZE", Arg::Scalar(ktir_cpu::ir::Scalar::I64(128))),
    ];

    let (outputs, report) =
        execute_function_with_latency(&module, "add_kernel", &args, HardwareConfig::default())
            .expect("run with latency");

    // Correctness is unaffected by tracking.
    assert_eq!(outputs.get("output_ptr").unwrap().data.len(), n);

    // The kernel does 2 HBM loads + 1 HBM store per core across 32 cores, plus
    // an addf — so the report must show real memory and compute cost.
    assert!(report.kernel_cycles() > 0.0, "expected non-zero kernel cycles");
    let summary = report.per_core_summary();
    assert_eq!(summary.len(), 32, "one row per core");
    let mem: f64 = summary.iter().map(|c| c.memory_cycles).sum();
    let compute: f64 = summary.iter().map(|c| c.compute_cycles).sum();
    assert!(mem > 0.0, "expected non-zero memory cycles, got {mem}");
    assert!(compute > 0.0, "expected non-zero compute cycles, got {compute}");
}
