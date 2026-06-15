module {
  func.func @ktir_decode_llama_3_2_1b_n127(%t307_ptr: index, %t97_ptr: index, %t308_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t307_ptr, sizes: [1, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x2048xf16>
    %view1 = ktdp.construct_memory_view %t97_ptr, sizes: [2048, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2047 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2048x2048xf16>
    %view2 = ktdp.construct_memory_view %t308_ptr, sizes: [1, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x2048xf16>
    %K3 = arith.constant 2048 : index
    %KB4 = arith.constant 32 : index
    %azero5 = arith.constant dense<0.0> : tensor<1x2048xf16>
    %mm6 = scf.for %k7 = %c0 to %K3 step %KB4 iter_args(%accit8 = %azero5) -> (tensor<1x2048xf16>) {
    %acc9 = ktdp.construct_access_tile %view0[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x32xindex>
    %v10 = ktdp.load %acc9 : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf16>
    %acc11 = ktdp.construct_access_tile %view1[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2047 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<2048x2048xf16> -> !ktdp.access_tile<2048x32xindex>
    %v12 = ktdp.load %acc11 : !ktdp.access_tile<2048x32xindex> -> tensor<2048x32xf16>
    %cinit13 = arith.constant dense<0.0> : tensor<1x2048xf16>
    %part14 = linalg.matmul_transpose_b ins(%v10, %v12 : tensor<1x32xf16>, tensor<2048x32xf16>) outs(%cinit13 : tensor<1x2048xf16>) -> tensor<1x2048xf16>
    %accnext15 = arith.addf %accit8, %part14 : tensor<1x2048xf16>
    scf.yield %accnext15 : tensor<1x2048xf16>
    }
    %acc16 = ktdp.construct_access_tile %view2[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x2048xindex>
    ktdp.store %mm6, %acc16 : tensor<1x2048xf16>, !ktdp.access_tile<1x2048xindex>
    return
  }
}
