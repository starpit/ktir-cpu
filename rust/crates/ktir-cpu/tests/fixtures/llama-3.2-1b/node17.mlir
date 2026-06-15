module {
  func.func @ktir_decode_llama_3_2_1b_n17(%t196_ptr: index, %t16_ptr: index, %t198_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t196_ptr, sizes: [1, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x2048xf16>
    %view1 = ktdp.construct_memory_view %t16_ptr, sizes: [512, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<512x2048xf16>
    %view2 = ktdp.construct_memory_view %t198_ptr, sizes: [1, 512], strides: [512, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x512xf16>
    %K3 = arith.constant 2048 : index
    %KB4 = arith.constant 128 : index
    %azero5 = arith.constant dense<0.0> : tensor<1x512xf16>
    %mm6 = scf.for %k7 = %c0 to %K3 step %KB4 iter_args(%accit8 = %azero5) -> (tensor<1x512xf16>) {
    %acc9 = ktdp.construct_access_tile %view0[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x128xindex>
    %v10 = ktdp.load %acc9 : !ktdp.access_tile<1x128xindex> -> tensor<1x128xf16>
    %acc11 = ktdp.construct_access_tile %view1[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<512x2048xf16> -> !ktdp.access_tile<512x128xindex>
    %v12 = ktdp.load %acc11 : !ktdp.access_tile<512x128xindex> -> tensor<512x128xf16>
    %cinit13 = arith.constant dense<0.0> : tensor<1x512xf16>
    %part14 = linalg.matmul_transpose_b ins(%v10, %v12 : tensor<1x128xf16>, tensor<512x128xf16>) outs(%cinit13 : tensor<1x512xf16>) -> tensor<1x512xf16>
    %accnext15 = arith.addf %accit8, %part14 : tensor<1x512xf16>
    scf.yield %accnext15 : tensor<1x512xf16>
    }
    %acc16 = ktdp.construct_access_tile %view2[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x512xf16> -> !ktdp.access_tile<1x512xindex>
    ktdp.store %mm6, %acc16 : tensor<1x512xf16>, !ktdp.access_tile<1x512xindex>
    return
  }
}
