use ktir_cpu::parser::parse_module;
use ktir_cpu::interpreter::{execute_function, Arg};
use ktir_cpu::dtypes::DType;
#[test]
fn probe_ring() {
    let src = include_str!("../../examples/ktir/ring_reduce.mlir");
    let m = parse_module(src).unwrap();
    let r = execute_function(&m, "ring_reduce", &[
        ("%in_ptr", Arg::Tensor{data: vec![1.0;512], shape: vec![4,128], dtype: DType::F16}),
        ("%out_ptr", Arg::Tensor{data: vec![0.0;128], shape: vec![1,128], dtype: DType::F16}),
    ]);
    println!("RESULT: {:?}", r.map(|o| o.get("%out_ptr").map(|t| t.data[0..3].to_vec())));
}
