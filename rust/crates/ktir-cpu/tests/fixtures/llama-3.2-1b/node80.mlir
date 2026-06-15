module {
  func.func @ktir_decode_llama_3_2_1b_n80(%t258_ptr: index, %t261_ptr: index, %t5_ptr: index, %t6_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %half0 = arith.constant 32 : index
    %view1 = ktdp.construct_memory_view %t258_ptr, sizes: [8, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x64xf16>
    %view2 = ktdp.construct_memory_view %t261_ptr, sizes: [8, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x64xf16>
    %acc3 = ktdp.construct_access_tile %view1[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8x64xf16> -> !ktdp.access_tile<8x32xindex>
    %v4 = ktdp.load %acc3 : !ktdp.access_tile<8x32xindex> -> tensor<8x32xf16>
    %acc5 = ktdp.construct_access_tile %view1[%c0, %half0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8x64xf16> -> !ktdp.access_tile<8x32xindex>
    %v6 = ktdp.load %acc5 : !ktdp.access_tile<8x32xindex> -> tensor<8x32xf16>
    %view7 = ktdp.construct_memory_view %t5_ptr, sizes: [64], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64xf16>
    %acc8 = ktdp.construct_access_tile %view7[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<64xf16> -> !ktdp.access_tile<32xindex>
    %v9 = ktdp.load %acc8 : !ktdp.access_tile<32xindex> -> tensor<32xf16>
    %cbi10 = tensor.empty() : tensor<8x32xf16>
    %cosb11 = linalg.broadcast ins(%v9 : tensor<32xf16>) outs(%cbi10 : tensor<8x32xf16>) dimensions = [0]
    %view12 = ktdp.construct_memory_view %t6_ptr, sizes: [64], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64xf16>
    %acc13 = ktdp.construct_access_tile %view12[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<64xf16> -> !ktdp.access_tile<32xindex>
    %v14 = ktdp.load %acc13 : !ktdp.access_tile<32xindex> -> tensor<32xf16>
    %cbi15 = tensor.empty() : tensor<8x32xf16>
    %sinb16 = linalg.broadcast ins(%v14 : tensor<32xf16>) outs(%cbi15 : tensor<8x32xf16>) dimensions = [0]
    %a117 = arith.mulf %v4, %cosb11 : tensor<8x32xf16>
    %a218 = arith.mulf %v6, %sinb16 : tensor<8x32xf16>
    %of19 = arith.subf %a117, %a218 : tensor<8x32xf16>
    %b120 = arith.mulf %v4, %sinb16 : tensor<8x32xf16>
    %b221 = arith.mulf %v6, %cosb11 : tensor<8x32xf16>
    %os22 = arith.addf %b120, %b221 : tensor<8x32xf16>
    %acc23 = ktdp.construct_access_tile %view2[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8x64xf16> -> !ktdp.access_tile<8x32xindex>
    ktdp.store %of19, %acc23 : tensor<8x32xf16>, !ktdp.access_tile<8x32xindex>
    %acc24 = ktdp.construct_access_tile %view2[%c0, %half0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8x64xf16> -> !ktdp.access_tile<8x32xindex>
    ktdp.store %os22, %acc24 : tensor<8x32xf16>, !ktdp.access_tile<8x32xindex>
    return
  }
}
