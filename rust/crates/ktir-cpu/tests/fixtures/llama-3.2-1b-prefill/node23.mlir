module {
  func.func @ktir_prefill_llama_3_2_1b_n23(%t203_ptr: index, %t195_ptr: index, %t204_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t203_ptr, sizes: [32, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>
    %acc1 = ktdp.construct_access_tile %view0[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<32x2048xindex>
    %v2 = ktdp.load %acc1 : !ktdp.access_tile<32x2048xindex> -> tensor<32x2048xf16>
    %view3 = ktdp.construct_memory_view %t195_ptr, sizes: [32, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>
    %acc4 = ktdp.construct_access_tile %view3[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<32x2048xindex>
    %v5 = ktdp.load %acc4 : !ktdp.access_tile<32x2048xindex> -> tensor<32x2048xf16>
    %add6 = arith.addf %v2, %v5 : tensor<32x2048xf16>
    %view7 = ktdp.construct_memory_view %t204_ptr, sizes: [32, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>
    %acc8 = ktdp.construct_access_tile %view7[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<32x2048xindex>
    ktdp.store %add6, %acc8 : tensor<32x2048xf16>, !ktdp.access_tile<32x2048xindex>
    return
  }
}
