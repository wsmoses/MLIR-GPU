// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 -lm && %s.exec1 &> %s.out1
// RUN: cgeist -omit-fp-contract %s %polyverify %stdinclude -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: cgeist -omit-fp-contract %s %polyexec %stdinclude -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 -lm && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm %s.mlir.time %s.clang.time

// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 -lm && %s.exec1 &> %s.out1
// RUN: cgeist -omit-fp-contract %s %polyverify %stdinclude -detect-reduction -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2

// RUN: cgeist -omit-fp-contract %s %stdinclude -O3 -S | FileCheck %s

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE eps = SCALAR_VAL(0.1);


#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(corr),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 - 1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1)[s0] : ((d0 + d1 * s0) mod 20 == 0)>

// CHECK-LABEL:   llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32, alignment = 1 : i64}
// CHECK:         llvm.mlir.global internal constant @str6("\0Aend   dump: %[[VAL_0:.*]]\0A\00") {addr_space = 0 : i32, alignment = 1 : i64}
// CHECK:         llvm.mlir.global internal constant @str5("%[[VAL_1:.*]].2lf \00") {addr_space = 0 : i32, alignment = 1 : i64}
// CHECK:         llvm.mlir.global internal constant @str4("\0A\00") {addr_space = 0 : i32, alignment = 1 : i64}
// CHECK:         llvm.mlir.global internal constant @str3("corr\00") {addr_space = 0 : i32, alignment = 1 : i64}
// CHECK:         llvm.mlir.global internal constant @str2("begin dump: %[[VAL_0]]\00") {addr_space = 0 : i32, alignment = 1 : i64}
// CHECK:         llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
// CHECK:         llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00") {addr_space = 0 : i32, alignment = 1 : i64}
// CHECK:         llvm.mlir.global external @stderr() {addr_space = 0 : i32} : !llvm.ptr
// CHECK:         llvm.mlir.global internal constant @str0("\00") {addr_space = 0 : i32, alignment = 1 : i64}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                    %[[VAL_1:.*]]: memref<?xmemref<?xi8>>) -> i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 1200 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 1440000 : i64
// CHECK:           %[[VAL_7:.*]] = arith.constant 1680000 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 1200 : i32
// CHECK:           %[[VAL_9:.*]] = arith.constant 1400 : i32
// CHECK:           %[[VAL_10:.*]] = memref.alloca() : memref<1xf64>
// CHECK:           %[[VAL_UNDEF:.*]] = llvm.mlir.undef : f64
// CHECK:           affine.store %[[VAL_UNDEF]], %[[VAL_10]][0] : memref<1xf64>
// CHECK:           %[[VAL_11:.*]] = call @polybench_alloc_data(%[[VAL_7]], %[[VAL_2]]) : (i64, i32) -> !llvm.ptr
// CHECK:           %[[VAL_12:.*]] = "polygeist.pointer2memref"(%[[VAL_11]]) : (!llvm.ptr) -> memref<1400x1200xf64>
// CHECK:           %[[VAL_13:.*]] = call @polybench_alloc_data(%[[VAL_6]], %[[VAL_2]]) : (i64, i32) -> !llvm.ptr
// CHECK:           %[[VAL_14:.*]] = "polygeist.pointer2memref"(%[[VAL_13]]) : (!llvm.ptr) -> memref<1200x1200xf64>
// CHECK:           %[[VAL_15:.*]] = call @polybench_alloc_data(%[[VAL_5]], %[[VAL_2]]) : (i64, i32) -> !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = "polygeist.pointer2memref"(%[[VAL_15]]) : (!llvm.ptr) -> memref<1200xf64>
// CHECK:           %[[VAL_17:.*]] = call @polybench_alloc_data(%[[VAL_5]], %[[VAL_2]]) : (i64, i32) -> !llvm.ptr
// CHECK:           %[[VAL_18:.*]] = "polygeist.pointer2memref"(%[[VAL_17]]) : (!llvm.ptr) -> memref<1200xf64>
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_10]] : memref<1xf64> to memref<?xf64>
// CHECK:           %[[VAL_20:.*]] = "polygeist.pointer2memref"(%[[VAL_11]]) : (!llvm.ptr) -> memref<?x1200xf64>
// CHECK:           call @init_array(%[[VAL_8]], %[[VAL_9]], %[[VAL_19]], %[[VAL_20]]) : (i32, i32, memref<?xf64>, memref<?x1200xf64>) -> ()
// CHECK:           %[[VAL_21:.*]] = affine.load %[[VAL_10]][0] : memref<1xf64>
// CHECK:           %[[VAL_22:.*]] = "polygeist.pointer2memref"(%[[VAL_13]]) : (!llvm.ptr) -> memref<?x1200xf64>
// CHECK:           %[[VAL_23:.*]] = "polygeist.pointer2memref"(%[[VAL_15]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK:           %[[VAL_24:.*]] = "polygeist.pointer2memref"(%[[VAL_17]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK:           call @kernel_correlation(%[[VAL_8]], %[[VAL_9]], %[[VAL_21]], %[[VAL_20]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]]) : (i32, i32, f64, memref<?x1200xf64>, memref<?x1200xf64>, memref<?xf64>, memref<?xf64>) -> ()
// CHECK:           %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_4]] : i32
// CHECK:           scf.if %[[VAL_25]] {
// CHECK:             %[[VAL_26:.*]] = affine.load %[[VAL_1]][0] : memref<?xmemref<?xi8>>
// CHECK:             %[[VAL_27:.*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK:             %[[VAL_28:.*]] = "polygeist.pointer2memref"(%[[VAL_27]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_29:.*]] = func.call @strcmp(%[[VAL_26]], %[[VAL_28]]) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK:             %[[VAL_30:.*]] = arith.cmpi eq, %[[VAL_29]], %[[VAL_3]] : i32
// CHECK:             scf.if %[[VAL_30]] {
// CHECK:               func.call @print_array(%[[VAL_8]], %[[VAL_22]]) : (i32, memref<?x1200xf64>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_12]] : memref<1400x1200xf64>
// CHECK:           memref.dealloc %[[VAL_14]] : memref<1200x1200xf64>
// CHECK:           memref.dealloc %[[VAL_16]] : memref<1200xf64>
// CHECK:           memref.dealloc %[[VAL_18]] : memref<1200xf64>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @polybench_alloc_data(
// CHECK-SAME:                                            %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                            %[[VAL_1:.*]]: i32) -> !llvm.ptr
// CHECK:           %[[VAL_EXT:.*]] = arith.extsi %[[VAL_1]] : i32 to i64
// CHECK:           %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_EXT]] : i64
// CHECK:           %[[VAL_3:.*]] = call @malloc(%[[VAL_2]]) : (i64) -> !llvm.ptr
// CHECK:           return %[[VAL_3]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   func.func private @init_array(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                  %[[VAL_2:.*]]: memref<?xf64>,
// CHECK-SAME:                                  %[[VAL_3:.*]]: memref<?x1200xf64>)
// CHECK:           %[[VAL_4:.*]] = arith.constant 1.200000e+03 : f64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1.400000e+03 : f64
// CHECK:           affine.store %[[VAL_5]], %[[VAL_2]][0] : memref<?xf64>
// CHECK:           affine.for %[[VAL_6:.*]] = 0 to 1400 {
// CHECK:             %[[VAL_IC:.*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:             %[[VAL_7:.*]] = arith.sitofp %[[VAL_IC]] : i32 to f64
// CHECK:             affine.for %[[VAL_8:.*]] = 0 to 1200 {
// CHECK:               %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : index to i32
// CHECK:               %[[VAL_10:.*]] = arith.muli %[[VAL_IC]], %[[VAL_9]] : i32
// CHECK:               %[[VAL_11:.*]] = arith.sitofp %[[VAL_10]] : i32 to f64
// CHECK:               %[[VAL_12:.*]] = arith.divf %[[VAL_11]], %[[VAL_4]] : f64
// CHECK:               %[[VAL_13:.*]] = arith.addf %[[VAL_12]], %[[VAL_7]] : f64
// CHECK:               affine.store %[[VAL_13]], %[[VAL_3]]{{\[}}%[[VAL_6]], %[[VAL_8]]] : memref<?x1200xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @kernel_correlation(
// CHECK-SAME:                                          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                          %[[VAL_2:.*]]: f64,
// CHECK-SAME:                                          %[[VAL_3:.*]]: memref<?x1200xf64>, %[[VAL_4:.*]]: memref<?x1200xf64>,
// CHECK-SAME:                                          %[[VAL_5:.*]]: memref<?xf64>, %[[VAL_6:.*]]: memref<?xf64>)
// CHECK:           %[[VAL_7:.*]] = arith.constant 1.000000e-01 : f64
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_IC:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           affine.for %[[VAL_11:.*]] = 0 to %[[VAL_10]] {
// CHECK:             affine.store %[[VAL_8]], %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:             affine.for %[[VAL_12:.*]] = 0 to %[[VAL_IC]] {
// CHECK:               %[[VAL_13:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_12]], %[[VAL_11]]] : memref<?x1200xf64>
// CHECK:               %[[VAL_14:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:               %[[VAL_15:.*]] = arith.addf %[[VAL_14]], %[[VAL_13]] : f64
// CHECK:               affine.store %[[VAL_15]], %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_16:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:             %[[VAL_17:.*]] = arith.divf %[[VAL_16]], %[[VAL_2]] : f64
// CHECK:             affine.store %[[VAL_17]], %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xf64>
// CHECK:           }
// CHECK:           affine.for %[[VAL_18:.*]] = 0 to %[[VAL_10]] {
// CHECK:             affine.store %[[VAL_8]], %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:             affine.for %[[VAL_19:.*]] = 0 to %[[VAL_IC]] {
// CHECK:               %[[VAL_20:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_19]], %[[VAL_18]]] : memref<?x1200xf64>
// CHECK:               %[[VAL_21:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:               %[[VAL_22:.*]] = arith.subf %[[VAL_20]], %[[VAL_21]] : f64
// CHECK:               %[[VAL_23:.*]] = arith.mulf %[[VAL_22]], %[[VAL_22]] : f64
// CHECK:               %[[VAL_24:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:               %[[VAL_25:.*]] = arith.addf %[[VAL_24]], %[[VAL_23]] : f64
// CHECK:               affine.store %[[VAL_25]], %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_26:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:             %[[VAL_27:.*]] = arith.divf %[[VAL_26]], %[[VAL_2]] : f64
// CHECK:             %[[VAL_28:.*]] = math.sqrt %[[VAL_27]] : f64
// CHECK:             %[[VAL_29:.*]] = arith.cmpf ole, %[[VAL_28]], %[[VAL_7]] : f64
// CHECK:             %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_9]], %[[VAL_28]] : f64
// CHECK:             affine.store %[[VAL_30]], %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = math.sqrt %[[VAL_2]] : f64
// CHECK:           affine.for %[[VAL_32:.*]] = 0 to %[[VAL_IC]] {
// CHECK:             affine.for %[[VAL_33:.*]] = 0 to %[[VAL_10]] {
// CHECK:               %[[VAL_34:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_33]]] : memref<?xf64>
// CHECK:               %[[VAL_35:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<?x1200xf64>
// CHECK:               %[[VAL_36:.*]] = arith.subf %[[VAL_35]], %[[VAL_34]] : f64
// CHECK:               affine.store %[[VAL_36]], %[[VAL_3]]{{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<?x1200xf64>
// CHECK:               %[[VAL_37:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_33]]] : memref<?xf64>
// CHECK:               %[[VAL_38:.*]] = arith.mulf %[[VAL_31]], %[[VAL_37]] : f64
// CHECK:               %[[VAL_39:.*]] = arith.divf %[[VAL_36]], %[[VAL_38]] : f64
// CHECK:               affine.store %[[VAL_39]], %[[VAL_3]]{{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<?x1200xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for %[[VAL_40:.*]] = 0 to #[[$ATTR_0]](){{\[}}%[[VAL_10]]] {
// CHECK:             affine.store %[[VAL_9]], %[[VAL_4]]{{\[}}%[[VAL_40]], %[[VAL_40]]] : memref<?x1200xf64>
// CHECK:             affine.for %[[VAL_41:.*]] = #[[$ATTR_1]](%[[VAL_40]]) to %[[VAL_10]] {
// CHECK:               affine.store %[[VAL_8]], %[[VAL_4]]{{\[}}%[[VAL_40]], %[[VAL_41]]] : memref<?x1200xf64>
// CHECK:               affine.for %[[VAL_42:.*]] = 0 to %[[VAL_IC]] {
// CHECK:                 %[[VAL_43:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_42]], %[[VAL_40]]] : memref<?x1200xf64>
// CHECK:                 %[[VAL_44:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_42]], %[[VAL_41]]] : memref<?x1200xf64>
// CHECK:                 %[[VAL_45:.*]] = arith.mulf %[[VAL_43]], %[[VAL_44]] : f64
// CHECK:                 %[[VAL_46:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_40]], %[[VAL_41]]] : memref<?x1200xf64>
// CHECK:                 %[[VAL_47:.*]] = arith.addf %[[VAL_46]], %[[VAL_45]] : f64
// CHECK:                 affine.store %[[VAL_47]], %[[VAL_4]]{{\[}}%[[VAL_40]], %[[VAL_41]]] : memref<?x1200xf64>
// CHECK:               }
// CHECK:               %[[VAL_48:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_40]], %[[VAL_41]]] : memref<?x1200xf64>
// CHECK:               affine.store %[[VAL_48]], %[[VAL_4]]{{\[}}%[[VAL_41]], %[[VAL_40]]] : memref<?x1200xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.store %[[VAL_9]], %[[VAL_4]][symbol(%[[VAL_10]]) - 1, symbol(%[[VAL_10]]) - 1] : memref<?x1200xf64>
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @strcmp(memref<?xi8>, memref<?xi8>) -> i32

// CHECK-LABEL:   func.func private @print_array(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                   %[[VAL_1:.*]]: memref<?x1200xf64>)
// CHECK:           %[[VAL_IC:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.addressof @stderr : !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
// CHECK:           %[[VAL_6:.*]] = llvm.call @fprintf(%[[VAL_3]], %[[VAL_5]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.addressof @str2 : !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_8]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.addressof @str3 : !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
// CHECK:           %[[VAL_12:.*]] = llvm.call @fprintf(%[[VAL_7]], %[[VAL_9]], %[[VAL_11]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.addressof @str5 : !llvm.ptr
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_13]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
// CHECK:           affine.for %[[VAL_15:.*]] = 0 to %[[VAL_IC]] {
// CHECK:             affine.for %[[VAL_16:.*]] = 0 to %[[VAL_IC]] {
// CHECK:               affine.if #[[$ATTR_2]](%[[VAL_16]], %[[VAL_15]]){{\[}}%[[VAL_IC]]] {
// CHECK:                 %[[VAL_17:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK:                 %[[VAL_18:.*]] = llvm.mlir.addressof @str4 : !llvm.ptr
// CHECK:                 %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_18]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK:                 %[[VAL_20:.*]] = llvm.call @fprintf(%[[VAL_17]], %[[VAL_19]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:               }
// CHECK:               %[[VAL_21:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK:               %[[VAL_22:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_15]], %[[VAL_16]]] : memref<?x1200xf64>
// CHECK:               %[[VAL_23:.*]] = llvm.call @fprintf(%[[VAL_21]], %[[VAL_14]], %[[VAL_22]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, f64) -> i32
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.addressof @str6 : !llvm.ptr
// CHECK:           %[[VAL_26:.*]] = llvm.getelementptr inbounds %[[VAL_25]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
// CHECK:           %[[VAL_27:.*]] = llvm.call @fprintf(%[[VAL_24]], %[[VAL_26]], %[[VAL_11]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           %[[VAL_28:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.addressof @str7 : !llvm.ptr
// CHECK:           %[[VAL_30:.*]] = llvm.getelementptr inbounds %[[VAL_29]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
// CHECK:           %[[VAL_31:.*]] = llvm.call @fprintf(%[[VAL_28]], %[[VAL_30]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @malloc(i64) -> !llvm.ptr

// EXEC: {{[0-9]\.[0-9]+}}
