// RUN: cgeist %s --no-inline --function=* -S | FileCheck %s
// RUN: cgeist %s --no-inline --function=* --struct-abi=0 -memref-abi=0 -S | FileCheck %s --check-prefix CHECK-STR


struct A {
  int val1;
  int val2;
};

struct B {
  bool bool1;
};

struct C : B, A {
  int val3;
};

struct D : C {
  int val3;
  bool bool2;
};

C* castAtoC(A *a) {
  // A -> C
  return static_cast<C *>(a);
};

D* castBtoD(B *b) {
  // B -> C -> D
  return static_cast<D *>(b);
}

D* castAtoD(A *b) {
  // A -> C -> D
  return static_cast<D *>(b);
}

int main() {
    C c;
    D d;
    c.val3 = 2;
    d.val3 = 2;
    return castAtoC(&c)->val3 + // expect nonzero offset due to A -> C
           castBtoD(&d)->val3 +
           castAtoD(&d)->val3; // expect nonzero offset due to A -> C
}

// CHECK-LABEL:   func.func @_Z8castAtoCP1A(
// CHECK-NEXT:     polygeist.memref2pointer
// CHECK-NEXT:     llvm.getelementptr {{.*}}[-1]
// CHECK-NEXT:     polygeist.pointer2memref
// CHECK-NEXT:     return
// CHECK-LABEL:   func.func @_Z8castBtoDP1B(
// CHECK-NEXT:     polygeist.memref2pointer
// CHECK-NEXT:     polygeist.pointer2memref
// CHECK-NEXT:     return
// CHECK-LABEL:   func.func @_Z8castAtoDP1A(
// CHECK-NEXT:     polygeist.memref2pointer
// CHECK-NEXT:     llvm.getelementptr {{.*}}[-1]
// CHECK-NEXT:     polygeist.pointer2memref
// CHECK-NEXT:     return
// CHECK-LABEL:    func.func @main()
// CHECK:    call @_Z8castAtoCP1A(
// CHECK:    call @_Z8castBtoDP1B(
// CHECK:    call @_Z8castAtoDP1A(

// TODO revisit after reimplementing opaque ptr mem2reg

// CHECK-STR-LABEL:   func.func @_Z8castAtoCP1A(
// CHECK-STR:     llvm.getelementptr {{.*}}[-4]
// CHECK-STR:     return
// CHECK-STR-LABEL:   func.func @_Z8castBtoDP1B(
// CHECK-STR-NOT:     llvm.getelementptr
// CHECK-STR:     return
