module {
  func.func @ktir_decode_smollm2_135m_n377(%t710_ptr: index, %t280_ptr: index, %t712_ptr: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t710_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %view1 = ktdp.construct_memory_view %t280_ptr, sizes: [192, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 + 575 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<192x576xf16>
    %view2 = ktdp.construct_memory_view %t712_ptr, sizes: [1, 192], strides: [192, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 191 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x192xf16>
    %K3 = arith.constant 576 : index
    %KB4 = arith.constant 64 : index
    %azero5 = arith.constant dense<0.0> : tensor<1x192xf16>
    %mm6 = scf.for %k7 = %c0 to %K3 step %KB4 iter_args(%accit8 = %azero5) -> (tensor<1x192xf16>) {
    %acc9 = ktdp.construct_access_tile %view0[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x64xindex>
    %v10 = ktdp.load %acc9 : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %acc11 = ktdp.construct_access_tile %view1[%c0, %k7] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<192x576xf16> -> !ktdp.access_tile<192x64xindex>
    %v12 = ktdp.load %acc11 : !ktdp.access_tile<192x64xindex> -> tensor<192x64xf16>
    %cinit13 = arith.constant dense<0.0> : tensor<1x192xf16>
    %part14 = linalg.matmul_transpose_b ins(%v10, %v12 : tensor<1x64xf16>, tensor<192x64xf16>) outs(%cinit13 : tensor<1x192xf16>) -> tensor<1x192xf16>
    %accnext15 = arith.addf %accit8, %part14 : tensor<1x192xf16>
    scf.yield %accnext15 : tensor<1x192xf16>
    }
    %acc16 = ktdp.construct_access_tile %view2[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 191 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x192xf16> -> !ktdp.access_tile<1x192xindex>
    ktdp.store %mm6, %acc16 : tensor<1x192xf16>, !ktdp.access_tile<1x192xindex>
    return
  }
}
