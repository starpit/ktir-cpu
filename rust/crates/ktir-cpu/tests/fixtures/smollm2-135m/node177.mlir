module {
  func.func @ktir_decode_smollm2_135m_n177(%t510_ptr: index, %t511_ptr: index, %t512_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t510_ptr, sizes: [1, 1536], strides: [1536, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x1536xf16>
    %acc1 = ktdp.construct_access_tile %view0[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x1536xf16> -> !ktdp.access_tile<1x1536xindex>
    %v2 = ktdp.load %acc1 : !ktdp.access_tile<1x1536xindex> -> tensor<1x1536xf16>
    %view3 = ktdp.construct_memory_view %t511_ptr, sizes: [1, 1536], strides: [1536, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x1536xf16>
    %acc4 = ktdp.construct_access_tile %view3[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x1536xf16> -> !ktdp.access_tile<1x1536xindex>
    %v5 = ktdp.load %acc4 : !ktdp.access_tile<1x1536xindex> -> tensor<1x1536xf16>
    %neg6 = arith.negf %v2 : tensor<1x1536xf16>
    %e7 = math.exp %neg6 : tensor<1x1536xf16>
    %one8 = arith.constant 1.0 : f16
    %onet9 = tensor.splat %one8 : tensor<1x1536xf16>
    %denom10 = arith.addf %onet9, %e7 : tensor<1x1536xf16>
    %silu11 = arith.divf %v2, %denom10 : tensor<1x1536xf16>
    %y12 = arith.mulf %silu11, %v5 : tensor<1x1536xf16>
    %view13 = ktdp.construct_memory_view %t512_ptr, sizes: [1, 1536], strides: [1536, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x1536xf16>
    %acc14 = ktdp.construct_access_tile %view13[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x1536xf16> -> !ktdp.access_tile<1x1536xindex>
    ktdp.store %y12, %acc14 : tensor<1x1536xf16>, !ktdp.access_tile<1x1536xindex>
    return
  }
}
