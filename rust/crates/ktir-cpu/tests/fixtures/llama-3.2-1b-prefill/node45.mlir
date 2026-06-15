module {
  func.func @ktir_prefill_llama_3_2_1b_n45(%t225_ptr: index, %t36_ptr: index, %t226_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t225_ptr, sizes: [32, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>
    %acc1 = ktdp.construct_access_tile %view0[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<32x2048xindex>
    %v2 = ktdp.load %acc1 : !ktdp.access_tile<32x2048xindex> -> tensor<32x2048xf16>
    %xf3 = arith.extf %v2 : tensor<32x2048xf16> to tensor<32x2048xf32>
    %x24 = arith.mulf %xf3, %xf3 : tensor<32x2048xf32>
    %zero5 = arith.constant 0.0 : f32
    %sinit6 = tensor.splat %zero5 : tensor<32xf32>
    %ssum7 = linalg.reduce { arith.addf }
      ins(%x24 : tensor<32x2048xf32>)
      outs(%sinit6 : tensor<32xf32>)
      dimensions = [1]
    %dt8 = arith.constant 2048.0 : f32
    %dts9 = tensor.splat %dt8 : tensor<32xf32>
    %mean10 = arith.divf %ssum7, %dts9 : tensor<32xf32>
    %epsc11 = arith.constant 0.00001 : f32
    %epst12 = tensor.splat %epsc11 : tensor<32xf32>
    %meps13 = arith.addf %mean10, %epst12 : tensor<32xf32>
    %rms14 = math.sqrt %meps13 : tensor<32xf32>
    %one15 = arith.constant 1.0 : f32
    %onet16 = tensor.splat %one15 : tensor<32xf32>
    %inv17 = arith.divf %onet16, %rms14 : tensor<32xf32>
    %invE18 = arith.truncf %inv17 : tensor<32xf32> to tensor<32xf16>
    %invinit19 = tensor.empty() : tensor<32x2048xf16>
    %invb20 = linalg.broadcast ins(%invE18 : tensor<32xf16>) outs(%invinit19 : tensor<32x2048xf16>) dimensions = [1]
    %xs21 = arith.mulf %v2, %invb20 : tensor<32x2048xf16>
    %view22 = ktdp.construct_memory_view %t36_ptr, sizes: [2048], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 2047 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2048xf16>
    %acc23 = ktdp.construct_access_tile %view22[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 2047 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<2048xf16> -> !ktdp.access_tile<2048xindex>
    %v24 = ktdp.load %acc23 : !ktdp.access_tile<2048xindex> -> tensor<2048xf16>
    %ginit25 = tensor.empty() : tensor<32x2048xf16>
    %gb26 = linalg.broadcast ins(%v24 : tensor<2048xf16>) outs(%ginit25 : tensor<32x2048xf16>) dimensions = [0]
    %y27 = arith.mulf %xs21, %gb26 : tensor<32x2048xf16>
    %view28 = ktdp.construct_memory_view %t226_ptr, sizes: [32, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>
    %acc29 = ktdp.construct_access_tile %view28[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<32x2048xindex>
    ktdp.store %y27, %acc29 : tensor<32x2048xf16>, !ktdp.access_tile<32x2048xindex>
    return
  }
}
