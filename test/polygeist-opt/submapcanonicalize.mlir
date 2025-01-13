// RUN: polygeist-opt -canonicalize %s | FileCheck %s
#map = affine_map<(d0)[s0, s1] -> (d0 * s0, d0 * s1)>
module @submap_to_load__store{
  func.func private @use(i32)
  func.func @f(%arg0: memref<?x?xi32>, %arg1 : index, %arg2 : index, %arg3 : index) {

    %submap = "polygeist.submap"(%arg0, %arg1, %arg2) <{map = #map}> : (memref<?x?xi32>, index, index) -> memref<?xi32>

    affine.for %arg4 = 0 to 10 {
        %l = affine.load %submap[5 + %arg4 + symbol(%arg3)] : memref<?xi32>
        func.call @use(%l) : (i32) -> ()
        affine.yield
    }
    return
  }

  func.func @g(%arg0: memref<?x?xi32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : i32) {
    %submap = "polygeist.submap"(%arg0, %arg1, %arg2) <{map = #map}> : (memref<?x?xi32>, index, index) -> memref<?xi32>
    affine.for %arg5 = 0 to 10 {
        affine.store %arg4, %submap[5 + %arg5 + symbol(%arg3)] : memref<?xi32>
        affine.yield
    }
    return
  }
}


// CHECK:   func.func @f(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index) {
// CHECK-NEXT:     affine.for %arg4 = 0 to 10 {
// CHECK-NEXT:       %0 = affine.load %arg0[(%arg4 + symbol(%arg3) + 5) * symbol(%arg1), (%arg4 + symbol(%arg3) + 5) * symbol(%arg2)] : memref<?x?xi32>
// CHECK-NEXT:       func.call @use(%0) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @g(%arg0: memref<?x?xi32>, %arg1: index, %arg2: index, %arg3: index, %arg4: i32) {
// CHECK-NEXT:     affine.for %arg5 = 0 to 10 {
// CHECK-NEXT:       affine.store %arg4, %arg0[(%arg5 + symbol(%arg3) + 5) * symbol(%arg1), (%arg5 + symbol(%arg3) + 5) * symbol(%arg2)] : memref<?x?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#map19 = affine_map<(d0, d1, d2, d3) -> (d1 + d3, d0 + d2)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @conv_1 {
  memref.global @out : memref<512x64xi32> = uninitialized
  memref.global @filter : memref<4x4xi32> = uninitialized
  memref.global @im : memref<515x67xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @im : memref<515x67xi32>
    %1 = memref.get_global @filter : memref<4x4xi32>
    %2 = memref.get_global @out : memref<512x64xi32>
    %3 = "polygeist.submap"(%0, %c4, %c4, %c64, %c512) <{map = #map19}> : (memref<515x67xi32>, index, index, index, index) -> memref<4x4x64x512xi32>
    %ssmap = "polygeist.submap"(%3, %c4, %c4, %c64, %c512) <{map = #map22}> : (memref<4x4x64x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %4 = "polygeist.submap"(%1, %c4, %c4, %c64, %c512) <{map = #map20}> : (memref<4x4xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %5 = "polygeist.submap"(%2, %c4, %c4, %c64, %c512) <{map = #map21}> : (memref<512x64xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    linalg.generic {indexing_maps = [#map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%ssmap, %4 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%5 : memref<?x?x?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %6 = arith.muli %in, %in_0 : i32
      %7 = arith.addi %out, %6 : i32
      linalg.yield %7 : i32
    }
    return %c0_i32 : i32
  }
}