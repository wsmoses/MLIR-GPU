// RUN: cgeist %s --no-inline --function=* -S | FileCheck %s

union S {
	double d;
};

class MyScalar {
 public:
  S v;
  MyScalar(double vv) {
   v.d = vv;
  }
};

void use(double);
void meta() {
	MyScalar alpha_scalar(1.0);
	alpha_scalar = MyScalar(3.0);
	use(alpha_scalar.v.d);
}

// CHECK-LABEL:   func.func @_Z4metav()
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 3.000000e+00 : f64
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.cast %[[VAL_2]] : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = memref.cast %[[VAL_4]] : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = memref.cast %[[VAL_6]] : memref<1x!llvm.struct<(struct<(f64)>)>> to memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK:           call @_ZN8MyScalarC1Ed(%[[VAL_7]], %[[VAL_1]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>, f64) -> ()
// CHECK:           call @_ZN8MyScalarC1Ed(%[[VAL_5]], %[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>, f64) -> ()
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = affine.load %[[VAL_4]][0] : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK:           affine.store %[[VAL_8]], %[[VAL_2]][0] : memref<1x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = call @_ZN8MyScalaraSEOS_(%[[VAL_7]], %[[VAL_3]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>, memref<?x!llvm.struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_10:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_6]]) : (memref<1x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_11:[A-Za-z0-9_]*]] = llvm.load %[[VAL_10]] : !llvm.ptr -> f64
// CHECK:           call @_Z3used(%[[VAL_11]]) : (f64) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN8MyScalarC1Ed(
// CHECK-SAME:                                %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(f64)>)>>,
// CHECK-SAME:                                %[[VAL_1:[A-Za-z0-9_]*]]: f64)
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_2]] : f64, !llvm.ptr
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN8MyScalaraSEOS_(
// CHECK-SAME:                                  %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(f64)>)>>,
// CHECK-SAME:                                  %[[VAL_1:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(f64)>)>>) -> memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(f64)>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?x!llvm.struct<(struct<(f64)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(f64)>>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = call @_ZN1SaSEOS_(%[[VAL_3]], %[[VAL_5]]) : (memref<?x!llvm.struct<(f64)>>, memref<?x!llvm.struct<(f64)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK:           return %[[VAL_0]] : memref<?x!llvm.struct<(struct<(f64)>)>>
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN1SaSEOS_(
// CHECK-SAME:                           %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(f64)>>,
// CHECK-SAME:                           %[[VAL_1:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(f64)>>) -> memref<?x!llvm.struct<(f64)>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(f64)>>) -> !llvm.ptr
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?x!llvm.struct<(f64)>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_3]], %[[VAL_4]], %[[VAL_2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           return %[[VAL_0]] : memref<?x!llvm.struct<(f64)>>
// CHECK:         }

