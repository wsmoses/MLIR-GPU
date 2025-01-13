//polygeist-opt --linalg-debufferize debufferize.mlir 

#map16 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d1 + d3, d0 + d2)>
#map18 = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map22 = affine_map<(d0, d1) -> (d1, d0)>

module @in_place_add{
    func.func @in_place_add(%value: f32) {
      %c0 = arith.constant 0 : index
      %buffer = memref.alloca() : memref<128xf32> 
      linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%buffer : memref<128xf32>) 
        outs(%buffer : memref<128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %sum = arith.addf %in, %value : f32
        linalg.yield %sum : f32
      }
      return
    }
}

module @conv_2 {
 func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
   %c0_i32 = arith.constant 0 : i32
   %0 = memref.alloca() : memref<515x67xi32>
   %1 = memref.alloca() : memref<4x4xi32>
   %2 = memref.alloca() : memref<512x64xi32>
   linalg.generic {indexing_maps = [#map17, #map18, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : memref<515x67xi32>, memref<4x4xi32>) outs(%2 : memref<512x64xi32>) {
   ^bb0(%in: i32, %in_0: i32, %out: i32):
     %3 = arith.muli %in, %in_0 : i32
     %4 = arith.addi %out, %3 : i32
     linalg.yield %4 : i32
   }
   return %c0_i32 : i32
 }
}

module @harris_score_with_gradient_extra_kernel {
    memref.global "private" @_ZL8coeffs_1 : memref<5x5xi32> = dense<1>
    memref.global "private" @_ZL8coeffs_y : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
    memref.global "private" @_ZL8coeffs_x : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c4_i32 = arith.constant 4 : i32
      %c0_i32 = arith.constant 0 : i32
      %alloca = memref.alloca() : memref<512x512xi32>
      %alloca_0 = memref.alloca() : memref<512x512xi32>
      %alloca_1 = memref.alloca() : memref<512x512xi32>
      %alloca_2 = memref.alloca() : memref<516x516xi32>
      %alloca_3 = memref.alloca() : memref<516x516xi32>
      %alloca_4 = memref.alloca() : memref<518x518xi32>
      %score    = memref.alloca() : memref<512x512xi32>
      %0 = memref.get_global @_ZL8coeffs_x : memref<3x3xi32>
      %1 = memref.get_global @_ZL8coeffs_y : memref<3x3xi32>
      %2 = memref.get_global @_ZL8coeffs_1 : memref<5x5xi32>
      // 2nd variant
      // %0 = memref.alloca() : memref<3x3xi32>
      // %1 = memref.alloca() : memref<3x3xi32>
      // %2 = memref.alloca() : memref<5x5xi32>
      linalg.generic {indexing_maps = [#map17, #map18, #map18, #map19, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%alloca_4, %0, %1 : memref<518x518xi32>, memref<3x3xi32>, memref<3x3xi32>) outs(%alloca_2, %alloca_3 : memref<516x516xi32>, memref<516x516xi32>) {
      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32):
        %4 = arith.muli %in, %in_5 : i32
        %5 = arith.addi %out_7, %4 : i32
        %6 = arith.muli %in, %in_6 : i32
        %7 = arith.addi %out, %6 : i32
        linalg.yield %7, %5 : i32, i32
      }
      linalg.generic {indexing_maps = [#map17, #map17, #map18, #map19, #map19, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%alloca_3, %alloca_2, %2 : memref<516x516xi32>, memref<516x516xi32>, memref<5x5xi32>) outs(%alloca, %alloca_0, %alloca_1 : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32>) {
      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32, %out_8: i32):
        %4 = arith.muli %in, %in : i32
        %5 = arith.muli %4, %in_6 : i32
        %6 = arith.addi %out_8, %5 : i32
        %7 = arith.muli %in_5, %in_5 : i32
        %8 = arith.muli %7, %in_6 : i32
        %9 = arith.addi %out_7, %8 : i32
        %10 = arith.muli %in, %in_5 : i32
        %11 = arith.muli %10, %in_6 : i32
        %12 = arith.addi %out, %11 : i32
        linalg.yield %12, %9, %6 : i32, i32, i32
      }
      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel"]} ins(%alloca_1, %alloca_0, %alloca : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32>) outs(%score : memref<512x512xi32>) {
      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32):
        %4 = arith.muli %in, %in_5 : i32
        %5 = arith.muli %in_6, %in_6 : i32
        %6 = arith.subi %4, %5 : i32
        %7 = arith.addi %in, %in_5 : i32
        %8 = arith.muli %7, %c4_i32 : i32
        %9 = arith.muli %8, %7 : i32
        %10 = arith.subi %6, %9 : i32
        linalg.yield %10 : i32
      }
      return %c0_i32 : i32
    }
  }