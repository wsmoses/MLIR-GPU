// RUN: cgeist %s  --no-inline --function=* -S | FileCheck %s

struct AIntDivider {
    AIntDivider() : divisor(3) {}
    unsigned int divisor;
};

struct Meta {
    AIntDivider sizes_[25];
    double x;
};

void kern() {
    Meta m;
}

// CHECK-LABEL:   func.func @_Z4kernv()  
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.cast %[[VAL_0]] : memref<1x!llvm.struct<(array<25 x struct<(i32)>>, f64)>> to memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>
// CHECK:           call @_ZN4MetaC1Ev(%[[VAL_1]]) : (memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN4MetaC1Ev(
// CHECK-SAME:                            %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>)  
// CHECK-DAG:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.constant 25 : index
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(array<25 x struct<(i32)>>, f64)>>) -> !llvm.ptr
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<25x1xi32>
// CHECK:           scf.for %[[VAL_6:[A-Za-z0-9_]*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_1]] {
// CHECK:             %[[VAL_7:[A-Za-z0-9_]*]] = "polygeist.subindex"(%[[VAL_5]], %[[VAL_6]]) : (memref<25x1xi32>, index) -> memref<?x1xi32>
// CHECK:             func.call @_ZN11AIntDividerC1Ev(%[[VAL_7]]) : (memref<?x1xi32>) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN11AIntDividerC1Ev(
// CHECK-SAME:                                    %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x1xi32>)  
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 3 : i32
// CHECK:           affine.store %[[VAL_1]], %[[VAL_0]][0, 0] : memref<?x1xi32>
// CHECK:           return
// CHECK:         }

