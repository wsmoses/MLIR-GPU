// RUN: cgeist %s --no-inline --function=* -S | FileCheck %s

extern "C" {

double kernel_deriche(int x, float y) {
    ([&y,x]() {
        y *= x;
    })();
    return y;
}

}
// CHECK-LABEL:   func.func @kernel_deriche(
// CHECK-SAME:                              %[[VAL_0:[A-Za-z0-9_]*]]: i32,
// CHECK-SAME:                              %[[VAL_1:[A-Za-z0-9_]*]]: f32) -> f64  
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.cast %[[VAL_2]] : memref<1x!llvm.struct<(memref<?xf32>, i32)>> to memref<?x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = memref.alloca() : memref<1xf32>
// CHECK:           affine.store %[[VAL_1]], %[[VAL_5]][0] : memref<1xf32>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = memref.cast %[[VAL_5]] : memref<1xf32> to memref<?xf32>
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_4]]) : (memref<1x!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_6]], %[[VAL_7]] : memref<?xf32>, !llvm.ptr
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_7]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xf32>, i32)>
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_8]] : i32, !llvm.ptr
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = affine.load %[[VAL_4]][0] : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK:           affine.store %[[VAL_9]], %[[VAL_2]][0] : memref<1x!llvm.struct<(memref<?xf32>, i32)>>
// CHECK:           call @_ZZ14kernel_dericheENK3$_0clEv(%[[VAL_3]]) : (memref<?x!llvm.struct<(memref<?xf32>, i32)>>) -> ()
// CHECK:           %[[VAL_10:[A-Za-z0-9_]*]] = affine.load %[[VAL_5]][0] : memref<1xf32>
// CHECK:           %[[VAL_11:[A-Za-z0-9_]*]] = arith.extf %[[VAL_10]] : f32 to f64
// CHECK:           return %[[VAL_11]] : f64
// CHECK:         }

// CHECK-LABEL:   func.func private @_ZZ14kernel_dericheENK3$_0clEv(
// CHECK-SAME:                                                      %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(memref<?xf32>, i32)>>)  
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> memref<?xf32>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_1]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xf32>, i32)>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = arith.sitofp %[[VAL_4]] : i32 to f32
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = affine.load %[[VAL_2]][0] : memref<?xf32>
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = arith.mulf %[[VAL_6]], %[[VAL_5]] : f32
// CHECK:           affine.store %[[VAL_7]], %[[VAL_2]][0] : memref<?xf32>
// CHECK:           return
// CHECK:         }

