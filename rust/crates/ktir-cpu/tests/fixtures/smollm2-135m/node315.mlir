module {
  func.func @ktir_decode_smollm2_135m_n315(%t649_ptr: index, %t234_ptr: index, %t650_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t649_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %acc1 = ktdp.construct_access_tile %view0[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    %v2 = ktdp.load %acc1 : !ktdp.access_tile<1x576xindex> -> tensor<1x576xf16>
    %xf3 = arith.extf %v2 : tensor<1x576xf16> to tensor<1x576xf32>
    %x24 = arith.mulf %xf3, %xf3 : tensor<1x576xf32>
    %zero5 = arith.constant 0.0 : f32
    %sinit6 = tensor.splat %zero5 : tensor<1xf32>
    %ssum7 = linalg.reduce { arith.addf }
      ins(%x24 : tensor<1x576xf32>)
      outs(%sinit6 : tensor<1xf32>)
      dimensions = [1]
    %dt8 = arith.constant 576.0 : f32
    %dts9 = tensor.splat %dt8 : tensor<1xf32>
    %mean10 = arith.divf %ssum7, %dts9 : tensor<1xf32>
    %epsc11 = arith.constant 0.00001 : f32
    %epst12 = tensor.splat %epsc11 : tensor<1xf32>
    %meps13 = arith.addf %mean10, %epst12 : tensor<1xf32>
    %rms14 = math.sqrt %meps13 : tensor<1xf32>
    %one15 = arith.constant 1.0 : f32
    %onet16 = tensor.splat %one15 : tensor<1xf32>
    %inv17 = arith.divf %onet16, %rms14 : tensor<1xf32>
    %invE18 = arith.truncf %inv17 : tensor<1xf32> to tensor<1xf16>
    %invinit19 = tensor.empty() : tensor<1x576xf16>
    %invb20 = linalg.broadcast ins(%invE18 : tensor<1xf16>) outs(%invinit19 : tensor<1x576xf16>) dimensions = [1]
    %xs21 = arith.mulf %v2, %invb20 : tensor<1x576xf16>
    %view22 = ktdp.construct_memory_view %t234_ptr, sizes: [576], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 575 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<576xf16>
    %acc23 = ktdp.construct_access_tile %view22[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 575 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<576xf16> -> !ktdp.access_tile<576xindex>
    %v24 = ktdp.load %acc23 : !ktdp.access_tile<576xindex> -> tensor<576xf16>
    %ginit25 = tensor.empty() : tensor<1x576xf16>
    %gb26 = linalg.broadcast ins(%v24 : tensor<576xf16>) outs(%ginit25 : tensor<1x576xf16>) dimensions = [0]
    %y27 = arith.mulf %xs21, %gb26 : tensor<1x576xf16>
    %view28 = ktdp.construct_memory_view %t650_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %acc29 = ktdp.construct_access_tile %view28[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    ktdp.store %y27, %acc29 : tensor<1x576xf16>, !ktdp.access_tile<1x576xindex>
    return
  }
}
