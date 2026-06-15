module {
  func.func @ktir_prefill_smollm2_135m_n451(%t785_ptr: index, %t334_ptr: index, %t786_ptr: index) attributes {grid = [32, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t785_ptr, sizes: [32, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x576xf16>
    %view1 = ktdp.construct_memory_view %t334_ptr, sizes: [49152, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 49151 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<49152x576xf16>
    %view2 = ktdp.construct_memory_view %t786_ptr, sizes: [32, 49152], strides: [49152, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 49151 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x49152xf16>
    %pid3 = ktdp.get_compute_tile_id : index
    %K4 = arith.constant 576 : index
    %KB5 = arith.constant 4 : index
    %azero6 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm7 = scf.for %k8 = %c0 to %K4 step %KB5 iter_args(%accit9 = %azero6) -> (tensor<1x16384xf16>) {
    %acc10 = ktdp.construct_access_tile %view0[%pid3, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x576xf16> -> !ktdp.access_tile<1x4xindex>
    %v11 = ktdp.load %acc10 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc12 = ktdp.construct_access_tile %view1[%c0, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<49152x576xf16> -> !ktdp.access_tile<16384x4xindex>
    %v13 = ktdp.load %acc12 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit14 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part15 = linalg.matmul_transpose_b ins(%v11, %v13 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit14 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext16 = arith.addf %accit9, %part15 : tensor<1x16384xf16>
    scf.yield %accnext16 : tensor<1x16384xf16>
    }
    %acc17 = ktdp.construct_access_tile %view2[%pid3, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x49152xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm7, %acc17 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff18 = arith.constant 16384 : index
    %K19 = arith.constant 576 : index
    %KB20 = arith.constant 4 : index
    %azero21 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm22 = scf.for %k23 = %c0 to %K19 step %KB20 iter_args(%accit24 = %azero21) -> (tensor<1x16384xf16>) {
    %acc25 = ktdp.construct_access_tile %view0[%pid3, %k23] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x576xf16> -> !ktdp.access_tile<1x4xindex>
    %v26 = ktdp.load %acc25 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc27 = ktdp.construct_access_tile %view1[%noff18, %k23] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<49152x576xf16> -> !ktdp.access_tile<16384x4xindex>
    %v28 = ktdp.load %acc27 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit29 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part30 = linalg.matmul_transpose_b ins(%v26, %v28 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit29 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext31 = arith.addf %accit24, %part30 : tensor<1x16384xf16>
    scf.yield %accnext31 : tensor<1x16384xf16>
    }
    %acc32 = ktdp.construct_access_tile %view2[%pid3, %noff18] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x49152xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm22, %acc32 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff33 = arith.constant 32768 : index
    %K34 = arith.constant 576 : index
    %KB35 = arith.constant 4 : index
    %azero36 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm37 = scf.for %k38 = %c0 to %K34 step %KB35 iter_args(%accit39 = %azero36) -> (tensor<1x16384xf16>) {
    %acc40 = ktdp.construct_access_tile %view0[%pid3, %k38] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x576xf16> -> !ktdp.access_tile<1x4xindex>
    %v41 = ktdp.load %acc40 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc42 = ktdp.construct_access_tile %view1[%noff33, %k38] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<49152x576xf16> -> !ktdp.access_tile<16384x4xindex>
    %v43 = ktdp.load %acc42 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit44 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part45 = linalg.matmul_transpose_b ins(%v41, %v43 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit44 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext46 = arith.addf %accit39, %part45 : tensor<1x16384xf16>
    scf.yield %accnext46 : tensor<1x16384xf16>
    }
    %acc47 = ktdp.construct_access_tile %view2[%pid3, %noff33] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x49152xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm37, %acc47 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    return
  }
}
