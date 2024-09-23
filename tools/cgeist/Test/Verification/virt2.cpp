// RUN: cgeist %s --no-inline --function=* -S | FileCheck %s

extern void print(char*);

class Root {
public:
    Root(int y) {
        print("Calling root");
    }
};

class FRoot {
public:
    FRoot() {
        print("Calling froot");
    }
};

class Sub : public Root, public FRoot {
public:
    Sub(int i, double y) : Root(i) {
        print("Calling Sub");
    }
};

void make() {
    Sub s(3, 3.14);
}

// CHECK-LABEL:   func.func @_Z4makev()
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 3.140000e+00 : f64
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.cast %[[VAL_2]] : memref<1x!llvm.struct<(struct<(i8)>, struct<(i8)>)>> to memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>
// CHECK:           call @_ZN3SubC1Eid(%[[VAL_3]], %[[VAL_1]], %[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>, i32, f64) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN3SubC1Eid(
// CHECK-SAME:                            %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>,
// CHECK-SAME:                            %[[VAL_1:[A-Za-z0-9_]*]]: i32,
// CHECK-SAME:                            %[[VAL_2:[A-Za-z0-9_]*]]: f64)
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(i8)>, struct<(i8)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(i8)>>
// CHECK:           call @_ZN4RootC1Ei(%[[VAL_4]], %[[VAL_1]]) : (memref<?x!llvm.struct<(i8)>>, i32) -> ()
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>, struct<(i8)>)>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_5]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(i8)>>
// CHECK:           call @_ZN5FRootC1Ev(%[[VAL_6]]) : (memref<?x!llvm.struct<(i8)>>) -> ()
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_7]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           call @_Z5printPc(%[[VAL_8]]) : (memref<?xi8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN4RootC1Ei(
// CHECK-SAME:                            %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(i8)>>,
// CHECK-SAME:                            %[[VAL_1:[A-Za-z0-9_]*]]: i32)
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           call @_Z5printPc(%[[VAL_3]]) : (memref<?xi8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN5FRootC1Ev(
// CHECK-SAME:                             %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(i8)>>)
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = llvm.mlir.addressof @str2 : !llvm.ptr
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           call @_Z5printPc(%[[VAL_2]]) : (memref<?xi8>) -> ()
// CHECK:           return
// CHECK:         }
