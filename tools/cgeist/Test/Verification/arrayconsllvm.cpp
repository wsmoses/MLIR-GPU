// RUN: cgeist %s  --no-inline --function=* -S | FileCheck %s

struct AIntDivider {
    AIntDivider() : divisor(3) {}
    unsigned int divisor;
    double v;
};

void kern() {
    AIntDivider sizes_[25];
}

// CHECK-LABEL:   func.func @_Z4kernv()  
// CHECK-DAG:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 25 : index
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.alloca() : memref<25x!llvm.struct<(i32, f64)>>
// CHECK:           scf.for %[[VAL_4:[A-Za-z0-9_]*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]] {
// CHECK:             %[[VAL_5:[A-Za-z0-9_]*]] = "polygeist.subindex"(%[[VAL_3]], %[[VAL_4]]) : (memref<25x!llvm.struct<(i32, f64)>>, index) -> memref<?x!llvm.struct<(i32, f64)>>
// CHECK:             func.call @_ZN11AIntDividerC1Ev(%[[VAL_5]]) : (memref<?x!llvm.struct<(i32, f64)>>) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN11AIntDividerC1Ev(
// CHECK-SAME:                                    %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(i32, f64)>>)  
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_2]] : i32, !llvm.ptr
// CHECK:           return
// CHECK:         }

