// RUN: polygeist-opt --mem2reg --split-input-file %s | FileCheck %s

module {
  func private @overwrite(%a : memref<i1>)
  func private @use(%a : i1)
  
  func @infLoop1(%c : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %false = arith.constant false
    %6 = memref.alloca() : memref<i1>
    memref.store %true, %6[] : memref<i1>
        scf.execute_region {
          br ^bb1(%c0_i32 : i32)
        ^bb1(%28 : i32):  // 2 preds: ^bb0, ^bb2
          %29 = arith.cmpi slt, %28, %c2_i32 : i32
          %30 = memref.load %6[] : memref<i1>
          %33 = arith.andi %29, %30 : i1
          cond_br %33, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          scf.if %c {
              %45 = arith.cmpi eq, %28, %c1_i32 : i32
              scf.if %45 {
                memref.store %false, %6[] : memref<i1>
              }
          }
              %44 = memref.load %6[] : memref<i1>
          call @use(%44) : (i1) -> ()
          %42 = arith.addi %28, %c1_i32 : i32
          br ^bb1(%42 : i32)
        ^bb3:  // pred: ^bb1
          scf.yield
        }
    return
  }

// CHECK:   func @infLoop1
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %0 = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %true, %0[] : memref<i1>
// CHECK-NEXT:     scf.execute_region {
// CHECK-NEXT:       br ^bb1(%c0_i32 : i32)
// CHECK:     ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:       %2 = arith.cmpi slt, %1, %c2_i32 : i32
// CHECK-NEXT:       %3 = memref.load %0[] : memref<i1>
// CHECK-NEXT:       %4 = arith.andi %2, %3 : i1
// CHECK-NEXT:       cond_br %4, ^bb2, ^bb3
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       %5 = scf.if %arg0 -> (i1) {
// CHECK-NEXT:         %7 = arith.cmpi eq, %1, %c1_i32 : i32
// CHECK-NEXT:         %8 = scf.if %7 -> (i1) {
// CHECK-NEXT:           memref.store %false, %0[] : memref<i1>
// CHECK-NEXT:           scf.yield %false : i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %3 : i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %8 : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %3 : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       call @use(%5) : (i1) -> ()
// CHECK-NEXT:       %6 = arith.addi %1, %c1_i32 : i32
// CHECK-NEXT:       br ^bb1(%6 : i32)
// CHECK:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

  func @infLoop2(%c : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %false = arith.constant false
    %6 = memref.alloca() : memref<i1>
    memref.store %true, %6[] : memref<i1>
        scf.execute_region {
          br ^bb1(%c0_i32 : i32)
        ^bb1(%28 : i32):  // 2 preds: ^bb0, ^bb2
          %29 = arith.cmpi slt, %28, %c2_i32 : i32
          %30 = memref.load %6[] : memref<i1>
          %33 = arith.andi %29, %30 : i1
          cond_br %33, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          scf.if %true {
              %45 = arith.cmpi eq, %28, %c1_i32 : i32
              scf.if %45 {
                call @overwrite(%6) : (memref<i1>) -> ()
              }
          }
          %44 = memref.load %6[] : memref<i1>
          call @use(%44) : (i1) -> ()
          %42 = arith.addi %28, %c1_i32 : i32
          br ^bb1(%42 : i32)
        ^bb3:  // pred: ^bb1
          scf.yield
        }
    return
  }

// CHECK:   func @infLoop2
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %0 = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %true, %0[] : memref<i1>
// CHECK-NEXT:     scf.execute_region {
// CHECK-NEXT:       br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:     ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:       %2 = arith.cmpi slt, %1, %c2_i32 : i32
// CHECK-NEXT:       %3 = memref.load %0[] : memref<i1>
// CHECK-NEXT:       %4 = arith.andi %2, %3 : i1
// CHECK-NEXT:       cond_br %4, ^bb2, ^bb3
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       scf.if %true {
// CHECK-NEXT:         %7 = arith.cmpi eq, %1, %c1_i32 : i32
// CHECK-NEXT:         scf.if %7 {
// CHECK-NEXT:           call @overwrite(%0) : (memref<i1>) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       %5 = memref.load %0[] : memref<i1>
// CHECK-NEXT:       call @use(%5) : (i1) -> ()
// CHECK-NEXT:       %6 = arith.addi %1, %c1_i32 : i32
// CHECK-NEXT:       br ^bb1(%6 : i32)
// CHECK-NEXT:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
}
