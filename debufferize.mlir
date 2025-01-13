//polygeist-opt --linalg-debufferize debufferize.mlir 

#map16 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d1 + d3, d0 + d2)>
#map18 = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>

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