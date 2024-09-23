// RUN: cgeist %s --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=* --cuda-lower --cpuify="distribute" -S | FileCheck %s

#include "Inputs/cuda.h"
#include "__clang_cuda_builtin_vars.h"

#define N 20

__device__ void bar(double* w) {
  w[threadIdx.x] = 2.0;
}

__global__ void foo(double * w) {
  bar(w);
}

template<typename T>
void templ(T fn, double *w) {
  fn<<< 1, N >>>(w);
}

void start(double* w) {
  templ(foo, w);
}

// CHECK:   func.func @_Z5startPd(%arg0: memref<?xf64>)
// CHECK-DAG:     %cst = arith.constant 2.000000e+00 : f64
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c20 = arith.constant 20 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     scf.parallel (%arg1) = (%c0) to (%c20) step (%c1) {
// CHECK-NEXT:       memref.store %cst, %arg0[%arg1] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
