// RUN: cgeist %s --function=* -S | FileCheck %s

class D {
  double a;
  double b;
};

class QStream {
  D device_;
  int id;
};

QStream ilaunch_kernel(QStream x) {
  return x;
}

// CHECK-LABEL:   func.func @_Z14ilaunch_kernel7QStream(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.struct<(struct<(f64, f64)>, i32)>) -> !llvm.struct<(struct<(f64, f64)>, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           %[[VAL_2:.*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           affine.store %[[VAL_0]], %[[VAL_2]][0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           %[[VAL_3:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> f64
// CHECK:           llvm.store %[[VAL_5]], %[[VAL_3]] : f64, !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_4]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> f64
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_3]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_8]] : f64, !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_4]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           llvm.store %[[VAL_10]], %[[VAL_11]] : i32, !llvm.ptr
// CHECK:           %[[VAL_12:.*]] = affine.load %[[VAL_1]][0] : memref<1x!llvm.struct<(struct<(f64, f64)>, i32)>>
// CHECK:           return %[[VAL_12]] : !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN7QStreamC1EOS_(
// CHECK-SAME:                                 %[[VAL_0:.*]]: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_2:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?x!llvm.struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> f64
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_2]] : f64, !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_3]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> f64
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_2]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK:           llvm.store %[[VAL_6]], %[[VAL_7]] : f64, !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_2]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           llvm.store %[[VAL_9]], %[[VAL_10]] : i32, !llvm.ptr
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN1DC1EOS_(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<?x2xf64>,
// CHECK-SAME:                           %[[VAL_1:.*]]: memref<?x2xf64>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_1]][0, 0] : memref<?x2xf64>
// CHECK:           affine.store %[[VAL_2]], %[[VAL_0]][0, 0] : memref<?x2xf64>
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_1]][0, 1] : memref<?x2xf64>
// CHECK:           affine.store %[[VAL_3]], %[[VAL_0]][0, 1] : memref<?x2xf64>
// CHECK:           return
// CHECK:         }