module {
  func.func @ktir_decode_smollm2_135m_n223(%t557_ptr: index, %t167_ptr: index, %t558_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t557_ptr, sizes: [1, 1536], strides: [1536, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x1536xf16>
    %view1 = ktdp.construct_memory_view %t167_ptr, sizes: [576, 1536], strides: [1536, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<576x1536xf16>
    %view2 = ktdp.construct_memory_view %t558_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %K3 = arith.constant 1536 : index
    %KB4 = arith.constant 128 : index
    %azero5 = arith.constant dense<0.0> : tensor<1x576xf16>
    %mm6 = scf.for %k7 = %c0 to %K3 step %KB4 iter_args(%accit8 = %azero5) -> (tensor<1x576xf16>) {
    %acc9 = ktdp.construct_access_tile %view0[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x1536xf16> -> !ktdp.access_tile<1x128xindex>
    %v10 = ktdp.load %acc9 : !ktdp.access_tile<1x128xindex> -> tensor<1x128xf16>
    %acc11 = ktdp.construct_access_tile %view1[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<576x1536xf16> -> !ktdp.access_tile<576x128xindex>
    %v12 = ktdp.load %acc11 : !ktdp.access_tile<576x128xindex> -> tensor<576x128xf16>
    %cinit13 = arith.constant dense<0.0> : tensor<1x576xf16>
    %part14 = linalg.matmul_transpose_b ins(%v10, %v12 : tensor<1x128xf16>, tensor<576x128xf16>) outs(%cinit13 : tensor<1x576xf16>) -> tensor<1x576xf16>
    %accnext15 = arith.addf %accit8, %part14 : tensor<1x576xf16>
    scf.yield %accnext15 : tensor<1x576xf16>
    }
    %acc16 = ktdp.construct_access_tile %view2[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    ktdp.store %mm6, %acc16 : tensor<1x576xf16>, !ktdp.access_tile<1x576xindex>
    return
  }
}
