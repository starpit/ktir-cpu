module {
  func.func @ktir_prefill_llama_3_2_1b_n146(%t325_ptr: index, %t111_ptr: index, %t327_ptr: index) attributes {grid = [32, 1]} {
    %c0 = arith.constant 0 : index
    %view0 = ktdp.construct_memory_view %t325_ptr, sizes: [32, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>
    %view1 = ktdp.construct_memory_view %t111_ptr, sizes: [8192, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 8191 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8192x2048xf16>
    %view2 = ktdp.construct_memory_view %t327_ptr, sizes: [32, 8192], strides: [8192, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 8191 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x8192xf16>
    %pid3 = ktdp.get_compute_tile_id : index
    %K4 = arith.constant 2048 : index
    %KB5 = arith.constant 8 : index
    %azero6 = arith.constant dense<0.0> : tensor<1x8192xf16>
    %mm7 = scf.for %k8 = %c0 to %K4 step %KB5 iter_args(%accit9 = %azero6) -> (tensor<1x8192xf16>) {
    %acc10 = ktdp.construct_access_tile %view0[%pid3, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x2048xf16> -> !ktdp.access_tile<1x8xindex>
    %v11 = ktdp.load %acc10 : !ktdp.access_tile<1x8xindex> -> tensor<1x8xf16>
    %acc12 = ktdp.construct_access_tile %view1[%c0, %k8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 8191 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8192x2048xf16> -> !ktdp.access_tile<8192x8xindex>
    %v13 = ktdp.load %acc12 : !ktdp.access_tile<8192x8xindex> -> tensor<8192x8xf16>
    %cinit14 = arith.constant dense<0.0> : tensor<1x8192xf16>
    %part15 = linalg.matmul_transpose_b ins(%v11, %v13 : tensor<1x8xf16>, tensor<8192x8xf16>) outs(%cinit14 : tensor<1x8192xf16>) -> tensor<1x8192xf16>
    %accnext16 = arith.addf %accit9, %part15 : tensor<1x8192xf16>
    scf.yield %accnext16 : tensor<1x8192xf16>
    }
    %acc17 = ktdp.construct_access_tile %view2[%pid3, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 8191 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x8192xf16> -> !ktdp.access_tile<1x8192xindex>
    ktdp.store %mm7, %acc17 : tensor<1x8192xf16>, !ktdp.access_tile<1x8192xindex>
    return
  }
}
