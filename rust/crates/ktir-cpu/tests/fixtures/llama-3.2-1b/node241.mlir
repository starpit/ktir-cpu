module {
  func.func @ktir_decode_llama_3_2_1b_n241(%t421_ptr: index, %t180_ptr: index, %t422_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t421_ptr, sizes: [1, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x2048xf16>
    %view1 = ktdp.construct_memory_view %t180_ptr, sizes: [128256, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 128255 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128256x2048xf16>
    %view2 = ktdp.construct_memory_view %t422_ptr, sizes: [1, 128256], strides: [128256, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 128255 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x128256xf16>
    %K3 = arith.constant 2048 : index
    %KB4 = arith.constant 4 : index
    %azero5 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm6 = scf.for %k7 = %c0 to %K3 step %KB4 iter_args(%accit8 = %azero5) -> (tensor<1x16384xf16>) {
    %acc9 = ktdp.construct_access_tile %view0[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v10 = ktdp.load %acc9 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc11 = ktdp.construct_access_tile %view1[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v12 = ktdp.load %acc11 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit13 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part14 = linalg.matmul_transpose_b ins(%v10, %v12 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit13 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext15 = arith.addf %accit8, %part14 : tensor<1x16384xf16>
    scf.yield %accnext15 : tensor<1x16384xf16>
    }
    %acc16 = ktdp.construct_access_tile %view2[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm6, %acc16 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff17 = arith.constant 16384 : index
    %K18 = arith.constant 2048 : index
    %KB19 = arith.constant 4 : index
    %azero20 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm21 = scf.for %k22 = %c0 to %K18 step %KB19 iter_args(%accit23 = %azero20) -> (tensor<1x16384xf16>) {
    %acc24 = ktdp.construct_access_tile %view0[%c0, %k22] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v25 = ktdp.load %acc24 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc26 = ktdp.construct_access_tile %view1[%noff17, %k22] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v27 = ktdp.load %acc26 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit28 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part29 = linalg.matmul_transpose_b ins(%v25, %v27 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit28 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext30 = arith.addf %accit23, %part29 : tensor<1x16384xf16>
    scf.yield %accnext30 : tensor<1x16384xf16>
    }
    %acc31 = ktdp.construct_access_tile %view2[%c0, %noff17] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm21, %acc31 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff32 = arith.constant 32768 : index
    %K33 = arith.constant 2048 : index
    %KB34 = arith.constant 4 : index
    %azero35 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm36 = scf.for %k37 = %c0 to %K33 step %KB34 iter_args(%accit38 = %azero35) -> (tensor<1x16384xf16>) {
    %acc39 = ktdp.construct_access_tile %view0[%c0, %k37] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v40 = ktdp.load %acc39 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc41 = ktdp.construct_access_tile %view1[%noff32, %k37] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v42 = ktdp.load %acc41 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit43 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part44 = linalg.matmul_transpose_b ins(%v40, %v42 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit43 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext45 = arith.addf %accit38, %part44 : tensor<1x16384xf16>
    scf.yield %accnext45 : tensor<1x16384xf16>
    }
    %acc46 = ktdp.construct_access_tile %view2[%c0, %noff32] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm36, %acc46 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff47 = arith.constant 49152 : index
    %K48 = arith.constant 2048 : index
    %KB49 = arith.constant 4 : index
    %azero50 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm51 = scf.for %k52 = %c0 to %K48 step %KB49 iter_args(%accit53 = %azero50) -> (tensor<1x16384xf16>) {
    %acc54 = ktdp.construct_access_tile %view0[%c0, %k52] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v55 = ktdp.load %acc54 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc56 = ktdp.construct_access_tile %view1[%noff47, %k52] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v57 = ktdp.load %acc56 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit58 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part59 = linalg.matmul_transpose_b ins(%v55, %v57 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit58 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext60 = arith.addf %accit53, %part59 : tensor<1x16384xf16>
    scf.yield %accnext60 : tensor<1x16384xf16>
    }
    %acc61 = ktdp.construct_access_tile %view2[%c0, %noff47] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm51, %acc61 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff62 = arith.constant 65536 : index
    %K63 = arith.constant 2048 : index
    %KB64 = arith.constant 4 : index
    %azero65 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm66 = scf.for %k67 = %c0 to %K63 step %KB64 iter_args(%accit68 = %azero65) -> (tensor<1x16384xf16>) {
    %acc69 = ktdp.construct_access_tile %view0[%c0, %k67] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v70 = ktdp.load %acc69 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc71 = ktdp.construct_access_tile %view1[%noff62, %k67] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v72 = ktdp.load %acc71 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit73 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part74 = linalg.matmul_transpose_b ins(%v70, %v72 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit73 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext75 = arith.addf %accit68, %part74 : tensor<1x16384xf16>
    scf.yield %accnext75 : tensor<1x16384xf16>
    }
    %acc76 = ktdp.construct_access_tile %view2[%c0, %noff62] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm66, %acc76 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff77 = arith.constant 81920 : index
    %K78 = arith.constant 2048 : index
    %KB79 = arith.constant 4 : index
    %azero80 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm81 = scf.for %k82 = %c0 to %K78 step %KB79 iter_args(%accit83 = %azero80) -> (tensor<1x16384xf16>) {
    %acc84 = ktdp.construct_access_tile %view0[%c0, %k82] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v85 = ktdp.load %acc84 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc86 = ktdp.construct_access_tile %view1[%noff77, %k82] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v87 = ktdp.load %acc86 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit88 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part89 = linalg.matmul_transpose_b ins(%v85, %v87 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit88 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext90 = arith.addf %accit83, %part89 : tensor<1x16384xf16>
    scf.yield %accnext90 : tensor<1x16384xf16>
    }
    %acc91 = ktdp.construct_access_tile %view2[%c0, %noff77] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm81, %acc91 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff92 = arith.constant 98304 : index
    %K93 = arith.constant 2048 : index
    %KB94 = arith.constant 4 : index
    %azero95 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %mm96 = scf.for %k97 = %c0 to %K93 step %KB94 iter_args(%accit98 = %azero95) -> (tensor<1x16384xf16>) {
    %acc99 = ktdp.construct_access_tile %view0[%c0, %k97] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v100 = ktdp.load %acc99 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc101 = ktdp.construct_access_tile %view1[%noff92, %k97] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 16383 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<16384x4xindex>
    %v102 = ktdp.load %acc101 : !ktdp.access_tile<16384x4xindex> -> tensor<16384x4xf16>
    %cinit103 = arith.constant dense<0.0> : tensor<1x16384xf16>
    %part104 = linalg.matmul_transpose_b ins(%v100, %v102 : tensor<1x4xf16>, tensor<16384x4xf16>) outs(%cinit103 : tensor<1x16384xf16>) -> tensor<1x16384xf16>
    %accnext105 = arith.addf %accit98, %part104 : tensor<1x16384xf16>
    scf.yield %accnext105 : tensor<1x16384xf16>
    }
    %acc106 = ktdp.construct_access_tile %view2[%c0, %noff92] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 16383 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x16384xindex>
    ktdp.store %mm96, %acc106 : tensor<1x16384xf16>, !ktdp.access_tile<1x16384xindex>
    %noff107 = arith.constant 114688 : index
    %K108 = arith.constant 2048 : index
    %KB109 = arith.constant 4 : index
    %azero110 = arith.constant dense<0.0> : tensor<1x13568xf16>
    %mm111 = scf.for %k112 = %c0 to %K108 step %KB109 iter_args(%accit113 = %azero110) -> (tensor<1x13568xf16>) {
    %acc114 = ktdp.construct_access_tile %view0[%c0, %k112] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x2048xf16> -> !ktdp.access_tile<1x4xindex>
    %v115 = ktdp.load %acc114 : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>
    %acc116 = ktdp.construct_access_tile %view1[%noff107, %k112] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 13567 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<128256x2048xf16> -> !ktdp.access_tile<13568x4xindex>
    %v117 = ktdp.load %acc116 : !ktdp.access_tile<13568x4xindex> -> tensor<13568x4xf16>
    %cinit118 = arith.constant dense<0.0> : tensor<1x13568xf16>
    %part119 = linalg.matmul_transpose_b ins(%v115, %v117 : tensor<1x4xf16>, tensor<13568x4xf16>) outs(%cinit118 : tensor<1x13568xf16>) -> tensor<1x13568xf16>
    %accnext120 = arith.addf %accit113, %part119 : tensor<1x13568xf16>
    scf.yield %accnext120 : tensor<1x13568xf16>
    }
    %acc121 = ktdp.construct_access_tile %view2[%c0, %noff107] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 13567 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128256xf16> -> !ktdp.access_tile<1x13568xindex>
    ktdp.store %mm111, %acc121 : tensor<1x13568xf16>, !ktdp.access_tile<1x13568xindex>
    return
  }
}
