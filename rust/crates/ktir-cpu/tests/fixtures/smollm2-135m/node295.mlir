module {
  func.func @ktir_decode_smollm2_135m_n295(%t629_ptr: index, %t220_ptr: index, %t630_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t629_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %view1 = ktdp.construct_memory_view %t220_ptr, sizes: [1536, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1535 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1536x576xf16>
    %view2 = ktdp.construct_memory_view %t630_ptr, sizes: [1, 1536], strides: [1536, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x1536xf16>
    %K3 = arith.constant 576 : index
    %KB4 = arith.constant 64 : index
    %azero5 = arith.constant dense<0.0> : tensor<1x1536xf16>
    %mm6 = scf.for %k7 = %c0 to %K3 step %KB4 iter_args(%accit8 = %azero5) -> (tensor<1x1536xf16>) {
    %acc9 = ktdp.construct_access_tile %view0[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x64xindex>
    %v10 = ktdp.load %acc9 : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %acc11 = ktdp.construct_access_tile %view1[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1535 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1536x576xf16> -> !ktdp.access_tile<1536x64xindex>
    %v12 = ktdp.load %acc11 : !ktdp.access_tile<1536x64xindex> -> tensor<1536x64xf16>
    %cinit13 = arith.constant dense<0.0> : tensor<1x1536xf16>
    %part14 = linalg.matmul_transpose_b ins(%v10, %v12 : tensor<1x64xf16>, tensor<1536x64xf16>) outs(%cinit13 : tensor<1x1536xf16>) -> tensor<1x1536xf16>
    %accnext15 = arith.addf %accit8, %part14 : tensor<1x1536xf16>
    scf.yield %accnext15 : tensor<1x1536xf16>
    }
    %acc16 = ktdp.construct_access_tile %view2[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x1536xf16> -> !ktdp.access_tile<1x1536xindex>
    ktdp.store %mm6, %acc16 : tensor<1x1536xf16>, !ktdp.access_tile<1x1536xindex>
    return
  }
}
