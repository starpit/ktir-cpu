module {
  func.func @ktir_prefill_smollm2_135m_n76(%t410_ptr: index, %t59_ptr: index, %t411_ptr: index) attributes {grid = [32, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t410_ptr, sizes: [32, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x576xf16>
    %view1 = ktdp.construct_memory_view %t59_ptr, sizes: [576, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<576x576xf16>
    %view2 = ktdp.construct_memory_view %t411_ptr, sizes: [32, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x576xf16>
    %pid3 = ktdp.get_compute_tile_id : index
    %K4 = arith.constant 576 : index
    %KB5 = arith.constant 64 : index
    %azero6 = arith.constant dense<0.0> : tensor<1x576xf16>
    %mm7 = scf.for %k8 = %c0 to %K4 step %KB5 iter_args(%accit9 = %azero6) -> (tensor<1x576xf16>) {
    %acc10 = ktdp.construct_access_tile %view0[%pid3, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x576xf16> -> !ktdp.access_tile<1x64xindex>
    %v11 = ktdp.load %acc10 : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %acc12 = ktdp.construct_access_tile %view1[%c0, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<576x576xf16> -> !ktdp.access_tile<576x64xindex>
    %v13 = ktdp.load %acc12 : !ktdp.access_tile<576x64xindex> -> tensor<576x64xf16>
    %cinit14 = arith.constant dense<0.0> : tensor<1x576xf16>
    %part15 = linalg.matmul_transpose_b ins(%v11, %v13 : tensor<1x64xf16>, tensor<576x64xf16>) outs(%cinit14 : tensor<1x576xf16>) -> tensor<1x576xf16>
    %accnext16 = arith.addf %accit9, %part15 : tensor<1x576xf16>
    scf.yield %accnext16 : tensor<1x576xf16>
    }
    %acc17 = ktdp.construct_access_tile %view2[%pid3, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x576xf16> -> !ktdp.access_tile<1x576xindex>
    ktdp.store %mm7, %acc17 : tensor<1x576xf16>, !ktdp.access_tile<1x576xindex>
    return
  }
}
