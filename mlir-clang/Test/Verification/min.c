// RUN: mlir-clang %s --function=min -S | FileCheck %s

int min(int a, int b) {
    if (a < b) return a;
    return b;
}

// CHECK:   func @min(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = llvm.mlir.undef : i32
// CHECK-NEXT:     %1 = cmpi slt, %arg0, %arg1 : i32
// CHECK-NEXT:     %2 = cmpi sge, %arg0, %arg1 : i32
// CHECK-NEXT:     %3 = select %1, %arg0, %0 : i32
// CHECK-NEXT:     %4 = select %2, %arg1, %3 : i32
// CHECK-NEXT:     return %4 : i32
// CHECK-NEXT:   }
