// RUN: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | opt -O3 -S | lli - &> %s.out2
// RUN: rm -f %s.exec1
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | opt -O3 -S | lli - > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* seidel-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_seidel_2d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

#pragma scop
  for (t = 0; t <= _PB_TSTEPS - 1; t++)
    for (i = 1; i<= _PB_N - 2; i++)
      for (j = 1; j <= _PB_N - 2; j++)
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/SCALAR_VAL(9.0);
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_seidel_2d (tsteps, n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}

// CHECK: #map = affine_map<()[s0] -> (s0 - 1)>

// CHECK: func @kernel_seidel_2d(%arg0: i32, %arg1: i32, %arg2: memref<2000x2000xf64>) {
// CHECK-NEXT:      %cst = constant 9.000000e+00 : f64
// CHECK-NEXT:      %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:      affine.for %arg3 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg4 = 1 to #map()[%1] {
// CHECK-NEXT:          affine.for %arg5 = 1 to #map()[%1] {
// CHECK-NEXT:            %2 = affine.load %arg2[%arg4 - 1, %arg5 - 1] : memref<2000x2000xf64>
// CHECK-NEXT:            %3 = affine.load %arg2[%arg4 - 1, %arg5] : memref<2000x2000xf64>
// CHECK-NEXT:            %4 = addf %2, %3 : f64
// CHECK-NEXT:            %5 = affine.load %arg2[%arg4 - 1, %arg5 + 1] : memref<2000x2000xf64>
// CHECK-NEXT:            %6 = addf %4, %5 : f64
// CHECK-NEXT:            %7 = affine.load %arg2[%arg4, %arg5 - 1] : memref<2000x2000xf64>
// CHECK-NEXT:            %8 = addf %6, %7 : f64
// CHECK-NEXT:            %9 = affine.load %arg2[%arg4, %arg5] : memref<2000x2000xf64>
// CHECK-NEXT:            %10 = addf %8, %9 : f64
// CHECK-NEXT:            %11 = affine.load %arg2[%arg4, %arg5 + 1] : memref<2000x2000xf64>
// CHECK-NEXT:            %12 = addf %10, %11 : f64
// CHECK-NEXT:            %13 = affine.load %arg2[%arg4 + 1, %arg5 - 1] : memref<2000x2000xf64>
// CHECK-NEXT:            %14 = addf %12, %13 : f64
// CHECK-NEXT:            %15 = affine.load %arg2[%arg4 + 1, %arg5] : memref<2000x2000xf64>
// CHECK-NEXT:            %16 = addf %14, %15 : f64
// CHECK-NEXT:            %17 = affine.load %arg2[%arg4 + 1, %arg5 + 1] : memref<2000x2000xf64>
// CHECK-NEXT:            %18 = addf %16, %17 : f64
// CHECK-NEXT:            %19 = divf %18, %cst : f64
// CHECK-NEXT:            affine.store %19, %arg2[%arg4, %arg5] : memref<2000x2000xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
