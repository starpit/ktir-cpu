module {
  func.func @ktir_prefill_smollm2_135m_n430(%t764_ptr: index, %t319_ptr: index, %t765_ptr: index) attributes {grid = [32, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t764_ptr, sizes: [32, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x576xf16>
    %view1 = ktdp.construct_memory_view %t319_ptr, sizes: [1536, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1535 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1536x576xf16>
    %view2 = ktdp.construct_memory_view %t765_ptr, sizes: [32, 1536], strides: [1536, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x1536xf16>
    %pid3 = ktdp.get_compute_tile_id : index
    %K4 = arith.constant 576 : index
    %KB5 = arith.constant 64 : index
    %azero6 = arith.constant dense<0.0> : tensor<1x1536xf16>
    %mm7 = scf.for %k8 = %c0 to %K4 step %KB5 iter_args(%accit9 = %azero6) -> (tensor<1x1536xf16>) {
    %acc10 = ktdp.construct_access_tile %view0[%pid3, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x576xf16> -> !ktdp.access_tile<1x64xindex>
    %v11 = ktdp.load %acc10 : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %acc12 = ktdp.construct_access_tile %view1[%c0, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1535 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1536x576xf16> -> !ktdp.access_tile<1536x64xindex>
    %v13 = ktdp.load %acc12 : !ktdp.access_tile<1536x64xindex> -> tensor<1536x64xf16>
    %cinit14 = arith.constant dense<0.0> : tensor<1x1536xf16>
    %part15 = linalg.matmul_transpose_b ins(%v11, %v13 : tensor<1x64xf16>, tensor<1536x64xf16>) outs(%cinit14 : tensor<1x1536xf16>) -> tensor<1x1536xf16>
    %accnext16 = arith.addf %accit9, %part15 : tensor<1x1536xf16>
    scf.yield %accnext16 : tensor<1x1536xf16>
    }
    %acc17 = ktdp.construct_access_tile %view2[%pid3, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1535 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x1536xf16> -> !ktdp.access_tile<1x1536xindex>
    ktdp.store %mm7, %acc17 : tensor<1x1536xf16>, !ktdp.access_tile<1x1536xindex>
    return
  }
}
