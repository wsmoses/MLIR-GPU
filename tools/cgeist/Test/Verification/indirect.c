// RUN: cgeist %s --no-inline --function=main -S | FileCheck %s
// RUN: cgeist %s --no-inline --function=main -S --emit-llvm | FileCheck %s --check-prefix=LLCHECK

int square(int x) {
  return x*x;
}
int meta(int (*f)(int), int x) {
  return f(x);
}

int printf(const char*, ...);
int main() {
  printf("sq(%d)=%d\n", 3, meta(square, 3));
  return 0;
}

// LLCHECK: define i32 @main()
// LLCHECK:   %[[V1:.+]] = call i32 @meta(ptr @square, i32 3)
// LLCHECK:   %[[V2:.+]] = call i32 (ptr, ...) @printf(ptr @str0, i32 3, i32 %[[V1]])
// LLCHECK:   ret i32 0
// LLCHECK: }

// CHECK-LABEL:   func.func @main() -> i32  
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<11 x i8>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.get_func"() <{name = @square}> : () -> !llvm.ptr
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?x!llvm.func<i32 (i32)>>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = call @meta(%[[VAL_5]], %[[VAL_1]]) : (memref<?x!llvm.func<i32 (i32)>>, i32) -> i32
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = llvm.call @printf(%[[VAL_3]], %[[VAL_1]], %[[VAL_6]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

