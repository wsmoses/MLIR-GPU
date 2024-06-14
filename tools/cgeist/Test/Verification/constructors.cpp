// RUN: cgeist  %s --function=* -S | FileCheck %s

class M {
public:
  float x;
  double y;
};

class A {
public:
  A(int x, const M &m) : x(x), m(m) {}
private:
  int x;
  M m;
};

class B : public A {
public:
  using A::A;
};

// CHECK-LABEL:   func.func @_Z4getBiRK1M(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32,
// CHECK-SAME:                            %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.struct<(struct<(i32, struct<(f32, f64)>)>)>
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(i32, struct<(f32, f64)>)>)> : (i64) -> !llvm.ptr
// CHECK:           call @[[B_CTOR:.*]](%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) : (!llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> !llvm.struct<(struct<(i32, struct<(f32, f64)>)>)>
// CHECK:           return %[[VAL_4]] : !llvm.struct<(struct<(i32, struct<(f32, f64)>)>)>
// CHECK:         }
B getB(int x, const M &m) { return {x, m}; }

// CHECK:         func.func @[[B_CTOR]](
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32,
// CHECK-SAME:                          %[[VAL_2:.*]]: !llvm.ptr)
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, struct<(f32, f64)>)>)>
// CHECK:           call @[[A_CTOR:.*]](%[[VAL_3]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }

// CHECK:         func.func @[[A_CTOR]](
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32,
// CHECK-SAME:                          %[[VAL_2:.*]]: !llvm.ptr)
// CHECK:           %[[SIZE:.*]] = arith.constant 16 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, struct<(f32, f64)>)>
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_3]] : i32, !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, struct<(f32, f64)>)>
// CHECK:           "llvm.intr.memcpy"(%[[VAL_4]], %[[VAL_2]], %[[SIZE]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           return
// CHECK:         }
