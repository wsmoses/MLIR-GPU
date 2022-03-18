// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func private @print()
  func @main() {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%arg2) = (%c0) to (%c5) step (%c1) {
      %0 = llvm.alloca %c1_i64 x i8 : (i64) -> !llvm.ptr<i8>
      scf.parallel (%arg3) = (%c0) to (%c2) step (%c1) {
        %4 = scf.while (%arg4 = %c1_i8) : (i8) -> i8 {
          %6 = arith.cmpi ne, %arg4, %c0_i8 : i8
          scf.condition(%6) %arg4 : i8
        } do {
        ^bb0(%arg4: i8):  // no predecessors
          llvm.store %c0_i8, %0 : !llvm.ptr<i8>
          "polygeist.barrier"(%arg3) : (index) -> ()
          scf.yield %c0_i8 : i8
        }
        %5 = arith.cmpi ne, %4, %c0_i8 : i8
        scf.if %5 {
          call @print() : () -> ()
        }
        scf.yield
      }
      scf.yield
    }
    return
  }
  func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<?xf32>, %len : index, %f : f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
        affine.parallel (%arg15, %arg16) = (0, 0) to (16, 16) {
            scf.for %arg17 = %c0 to %len step %c1 {
              affine.store %f, %arg0[%arg15] : memref<?xf32>
              "polygeist.barrier"(%arg15, %arg16, %c0) : (index, index, index) -> ()
            }
        }
    return 
  }
}


// CHECK:   func @main() {
// CHECK-NEXT:     %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT:     %c1_i8 = arith.constant 1 : i8
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c5 = arith.constant 5 : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     scf.parallel (%arg0) = (%c0) to (%c5) step (%c1) {
// CHECK-NEXT:       %0 = llvm.alloca %c1_i64 x i8 : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %1 = memref.alloc(%c2) : memref<?xi8>
// CHECK-NEXT:       %2 = memref.alloc(%c2) : memref<?xi8>
// CHECK-NEXT:       scf.parallel (%arg1) = (%c0) to (%c2) step (%c1) {
// CHECK-NEXT:         %3 = "polygeist.subindex"(%2, %arg1) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:         memref.store %c1_i8, %3[] : memref<i8>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.while : () -> () {
// CHECK-NEXT:         %3 = memref.alloca() : memref<i1>
// CHECK-NEXT:         scf.parallel (%arg1) = (%c0) to (%c2) step (%c1) {
// CHECK-NEXT:           %5 = "polygeist.subindex"(%2, %arg1) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:           %6 = memref.load %5[] : memref<i8>
// CHECK-NEXT:           %7 = arith.cmpi ne, %6, %c0_i8 : i8
// CHECK-NEXT:           %8 = arith.cmpi eq, %c0, %arg1 : index
// CHECK-NEXT:           scf.if %8 {
// CHECK-NEXT:             memref.store %7, %3[] : memref<i1>
// CHECK-NEXT:           }
// CHECK-NEXT:           %9 = "polygeist.subindex"(%1, %arg1) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:           memref.store %6, %9[] : memref<i8>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         %4 = memref.load %3[] : memref<i1>
// CHECK-NEXT:         scf.condition(%4)
// CHECK-NEXT:       } do {
// CHECK-NEXT:         scf.parallel (%arg1) = (%c0) to (%c2) step (%c1) {
// CHECK-NEXT:           llvm.store %c0_i8, %0 : !llvm.ptr<i8>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.parallel (%arg1) = (%c0) to (%c2) step (%c1) {
// CHECK-NEXT:           %3 = "polygeist.subindex"(%2, %arg1) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:           memref.store %c0_i8, %3[] : memref<i8>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.parallel (%arg1) = (%c0) to (%c2) step (%c1) {
// CHECK-NEXT:         %3 = "polygeist.subindex"(%1, %arg1) : (memref<?xi8>, index) -> memref<i8>
// CHECK-NEXT:         %4 = memref.load %3[] : memref<i8>
// CHECK-NEXT:         %5 = arith.cmpi ne, %4, %c0_i8 : i8
// CHECK-NEXT:         scf.if %5 {
// CHECK-NEXT:           call @print() : () -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       memref.dealloc %1 : memref<?xi8>
// CHECK-NEXT:       memref.dealloc %2 : memref<?xi8>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) {
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     scf.for %arg3 = %c0 to %arg1 step %c1 {
// CHECK-NEXT:       affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) {
// CHECK-NEXT:         affine.store %arg2, %arg0[%arg4] : memref<?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
