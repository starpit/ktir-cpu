module {
  func.func @ktir_decode_smollm2_135m_n194(%t528_ptr: index, %t523_ptr: index, %t529_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t528_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %acc1 = ktdp.construct_access_tile %view0[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    %v2 = ktdp.load %acc1 : !ktdp.access_tile<1x576xindex> -> tensor<1x576xf16>
    %view3 = ktdp.construct_memory_view %t523_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %acc4 = ktdp.construct_access_tile %view3[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    %v5 = ktdp.load %acc4 : !ktdp.access_tile<1x576xindex> -> tensor<1x576xf16>
    %add6 = arith.addf %v2, %v5 : tensor<1x576xf16>
    %view7 = ktdp.construct_memory_view %t529_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %acc8 = ktdp.construct_access_tile %view7[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    ktdp.store %add6, %acc8 : tensor<1x576xf16>, !ktdp.access_tile<1x576xindex>
    return
  }
}
