// RUN: cgeist %s --no-inline --function=* -S | FileCheck %s

class M {
};


struct _Alloc_hider : M
      {
  _Alloc_hider() {}

  char *_M_p; // The actual data.
      };

class A {
    public:
  int x;
  virtual void foo();
  A() : x(3) {}
};

class mbasic_stringbuf : public A {
public:
  _Alloc_hider _M_dataplus;
  mbasic_stringbuf() {}
};

void a() {
    mbasic_stringbuf a;
}

// CHECK-LABEL:   func.func @_Z1av()
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.cast %[[VAL_0]] : memref<1x!llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>> to memref<?x!llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>
// CHECK:           call @_ZN16mbasic_stringbufC1Ev(%[[VAL_1]]) : (memref<?x!llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN16mbasic_stringbufC1Ev(
// CHECK-SAME:                                         %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>)
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?x!llvm.struct<packed (ptr, i32, array<4 x i8>)>>
// CHECK:           call @_ZN1AC1Ev(%[[VAL_2]]) : (memref<?x!llvm.struct<packed (ptr, i32, array<4 x i8>)>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN1AC1Ev(
// CHECK-SAME:                         %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<packed (ptr, i32, array<4 x i8>)>>)
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<packed (ptr, i32, array<4 x i8>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (ptr, i32, array<4 x i8>)>
// CHECK:           llvm.store %[[VAL_1]], %[[VAL_3]] : i32, !llvm.ptr
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN12_Alloc_hiderC1Ev(
// CHECK-SAME:                                     %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(i8)>, memref<?xi8>)>>)
// CHECK:           return
// CHECK:         }

