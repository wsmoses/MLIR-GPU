// RUN: polygeist-opt -canonicalize %s | FileCheck %s
#map = affine_map<(d0)[s0, s1] -> (d0 * s0, d0 * s1)>
module {
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