module {
  func.func @ktir_prefill_llama_3_2_1b_n241(%t421_ptr: index, %t180_ptr: index, %t422_ptr: index) attributes {grid = [32, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t421_ptr, sizes: [32, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>
    %view1 = ktdp.construct_memory_view %t180_ptr, sizes: [128256, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 128255 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128256x2048xf16>
    %view2 = ktdp.construct_memory_view %t422_ptr, sizes: [32, 128256], strides: [128256, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 128255 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x128256xf16>
    %pid3 = ktdp.get_compute_tile_id : index
    %K4 = arith.constant 2048 : index
    %KB5 = arith.constant 4 : index
    %azero6 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm7 = scf.for %k8 = %c0 to %K4 step %KB5 iter_args(%accit9 = %azero6) -> (tensor<1x16384xf16>) {
    %acc10 = ktdp.construct_access_tile %view0[%pid3, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v11 = ktdp.load %acc10 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc12 = ktdp.construct_access_tile %view1[%c0, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v13 = ktdp.load %acc12 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit14 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part15 = linalg.matmul_transpose_b ins(%v11, %v13 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit14 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext16 = arith.addf %accit9, %part15 : tensor<1x16384xf16>
    scf.yield %accnext16 : tensor<1x16384xf16>
    }
    %acc17 = ktdp.construct_access_tile %view2[%pid3, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm7, %acc17 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff18 = arith.constant 16384 : index
    %K19 = arith.constant 2048 : index
    %KB20 = arith.constant 4 : index
    %azero21 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm22 = scf.for %k23 = %c0 to %K19 step %KB20 iter_args(%accit24 = %azero21) -> (tensor<1x16384xf16>) {
    %acc25 = ktdp.construct_access_tile %view0[%pid3, %k23] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v26 = ktdp.load %acc25 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc27 = ktdp.construct_access_tile %view1[%noff18, %k23] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v28 = ktdp.load %acc27 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit29 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part30 = linalg.matmul_transpose_b ins(%v26, %v28 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit29 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext31 = arith.addf %accit24, %part30 : tensor<1x16384xf16>
    scf.yield %accnext31 : tensor<1x16384xf16>
    }
    %acc32 = ktdp.construct_access_tile %view2[%pid3, %noff18] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm22, %acc32 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff33 = arith.constant 32768 : index
    %K34 = arith.constant 2048 : index
    %KB35 = arith.constant 4 : index
    %azero36 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm37 = scf.for %k38 = %c0 to %K34 step %KB35 iter_args(%accit39 = %azero36) -> (tensor<1x16384xf16>) {
    %acc40 = ktdp.construct_access_tile %view0[%pid3, %k38] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v41 = ktdp.load %acc40 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc42 = ktdp.construct_access_tile %view1[%noff33, %k38] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v43 = ktdp.load %acc42 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit44 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part45 = linalg.matmul_transpose_b ins(%v41, %v43 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit44 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext46 = arith.addf %accit39, %part45 : tensor<1x16384xf16>
    scf.yield %accnext46 : tensor<1x16384xf16>
    }
    %acc47 = ktdp.construct_access_tile %view2[%pid3, %noff33] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm37, %acc47 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff48 = arith.constant 49152 : index
    %K49 = arith.constant 2048 : index
    %KB50 = arith.constant 4 : index
    %azero51 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm52 = scf.for %k53 = %c0 to %K49 step %KB50 iter_args(%accit54 = %azero51) -> (tensor<1x16384xf16>) {
    %acc55 = ktdp.construct_access_tile %view0[%pid3, %k53] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v56 = ktdp.load %acc55 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc57 = ktdp.construct_access_tile %view1[%noff48, %k53] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v58 = ktdp.load %acc57 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit59 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part60 = linalg.matmul_transpose_b ins(%v56, %v58 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit59 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext61 = arith.addf %accit54, %part60 : tensor<1x16384xf16>
    scf.yield %accnext61 : tensor<1x16384xf16>
    }
    %acc62 = ktdp.construct_access_tile %view2[%pid3, %noff48] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm52, %acc62 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff63 = arith.constant 65536 : index
    %K64 = arith.constant 2048 : index
    %KB65 = arith.constant 4 : index
    %azero66 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm67 = scf.for %k68 = %c0 to %K64 step %KB65 iter_args(%accit69 = %azero66) -> (tensor<1x16384xf16>) {
    %acc70 = ktdp.construct_access_tile %view0[%pid3, %k68] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v71 = ktdp.load %acc70 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc72 = ktdp.construct_access_tile %view1[%noff63, %k68] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v73 = ktdp.load %acc72 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit74 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part75 = linalg.matmul_transpose_b ins(%v71, %v73 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit74 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext76 = arith.addf %accit69, %part75 : tensor<1x16384xf16>
    scf.yield %accnext76 : tensor<1x16384xf16>
    }
    %acc77 = ktdp.construct_access_tile %view2[%pid3, %noff63] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm67, %acc77 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff78 = arith.constant 81920 : index
    %K79 = arith.constant 2048 : index
    %KB80 = arith.constant 4 : index
    %azero81 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm82 = scf.for %k83 = %c0 to %K79 step %KB80 iter_args(%accit84 = %azero81) -> (tensor<1x16384xf16>) {
    %acc85 = ktdp.construct_access_tile %view0[%pid3, %k83] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v86 = ktdp.load %acc85 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc87 = ktdp.construct_access_tile %view1[%noff78, %k83] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v88 = ktdp.load %acc87 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit89 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part90 = linalg.matmul_transpose_b ins(%v86, %v88 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit89 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext91 = arith.addf %accit84, %part90 : tensor<1x16384xf16>
    scf.yield %accnext91 : tensor<1x16384xf16>
    }
    %acc92 = ktdp.construct_access_tile %view2[%pid3, %noff78] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm82, %acc92 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff93 = arith.constant 98304 : index
    %K94 = arith.constant 2048 : index
    %KB95 = arith.constant 4 : index
    %azero96 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm97 = scf.for %k98 = %c0 to %K94 step %KB95 iter_args(%accit99 = %azero96) -> (tensor<1x16384xf16>) {
    %acc100 = ktdp.construct_access_tile %view0[%pid3, %k98] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v101 = ktdp.load %acc100 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc102 = ktdp.construct_access_tile %view1[%noff93, %k98] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v103 = ktdp.load %acc102 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit104 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part105 = linalg.matmul_transpose_b ins(%v101, %v103 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit104 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext106 = arith.addf %accit99, %part105 : tensor<1x16384xf16>
    scf.yield %accnext106 : tensor<1x16384xf16>
    }
    %acc107 = ktdp.construct_access_tile %view2[%pid3, %noff93] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm97, %acc107 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff108 = arith.constant 114688 : index
    %K109 = arith.constant 2048 : index
    %KB110 = arith.constant 4 : index
    %azero111 = arith.constant dense<0.0> : tensor<1x13568xf16>
    %mm112 = scf.for %k113 = %c0 to %K109 step %KB110 iter_args(%accit114 = %azero111) -> (tensor<1x13568xf16>) {
    %acc115 = ktdp.construct_access_tile %view0[%pid3, %k113] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v116 = ktdp.load %acc115 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc117 = ktdp.construct_access_tile %view1[%noff108, %k113] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 13567 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<13568x4xindex>
    %v118 = ktdp.load %acc117 : !ktdp.access_tile<13568x4xindex> -> tensor<13568x4xf16>
    %cinit119 = arith.constant dense<0.0> : tensor<1x13568xf16>
    %part120 = linalg.matmul_transpose_b ins(%v116, %v118 : tensor<1x4xf16>, tensor<13568x4xf16>) outs(%cinit119 : tensor<1x13568xf16>) -> tensor<1x13568xf16>
    %accnext121 = arith.addf %accit114, %part120 : tensor<1x13568xf16>
    scf.yield %accnext121 : tensor<1x13568xf16>
    }
    %acc122 = ktdp.construct_access_tile %view2[%pid3, %noff108] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 13567 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x128256xf16> -> !ktdp.access_tile<1x13568xindex>
    ktdp.store %mm112, %acc122 : tensor<1x13568xf16>, !ktdp.access_tile<1x13568xindex>
    return
  }
}
