// RUN: polygeist-opt --raise-affine-to-linalg --split-input-file %s | FileCheck %s

module {
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    scf.if %12 {
      affine.for %arg4 = 0 to %17 {
        %ld = affine.load %18[%arg4] : memref<?xf32>
        affine.store %ld, %19[%arg4] : memref<?xf32>
      }
   }
    return
  }


  func.func @main2(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    scf.if %12 {
      affine.for %arg4 = 0 to 17 {
        %ld = affine.load %18[3 * %arg4] : memref<?xf32>
        %ld2 = affine.load %18[0] : memref<?xf32>
        %fadd = arith.addf %ld, %ld2 : f32
        affine.store %fadd, %19[%arg4 + 17] : memref<?xf32>
      }
   }
    return
  }

}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK:   func.func @main(%[[arg0:.+]]: i1, %[[arg1:.+]]: i32, %[[arg2:.+]]: memref<?xf32>, %[[arg3:.+]]: memref<?xf32>) {
// CHECK-NEXT:     %[[c4:.+]] = arith.constant 4 : index
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg1]] : i32 to index
// CHECK-NEXT:     %[[V1:.+]] = arith.muli %[[V0]], %[[c4]] : index
// CHECK-NEXT:     %[[V2:.+]] = arith.divui %[[V1]], %[[c4]] : index
// CHECK-NEXT:     scf.if %[[arg0]] {
// TODO note that presently we do not ensure that the memrefs are sliced to the right size as the space requires
// CHECK-NEXT:        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg2 : memref<?xf32>) outs(%alloca : memref<?xf32>) {
// CHECK-NEXT:        ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:          linalg.yield %in : f32
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:     }

//constant-access
module @constant_access{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %ci324 = arith.constant 4.0 : f32
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld = affine.load %18[%arg4] : memref<?xf32>
      %mul = arith.mulf %ld, %ci324 : f32
      affine.store %mul, %19[%arg4] : memref<?xf32>
    }
    return
  }
}

//constant-mem-access
module @constant_mem_access{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 4 to 17 step 2 {
      %ld = affine.load %18[3*%arg4] : memref<?xf32>
      %ld2 = affine.load %18[%c4] : memref<?xf32>
      %mul = arith.mulf %ld, %ld2 : f32
      affine.store %mul, %19[%arg4] : memref<?xf32>
    }
    return
  }
}

//without-if
module @no_if{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld = affine.load %18[%arg4] : memref<?xf32>
      affine.store %ld, %19[%arg4] : memref<?xf32>
    }
    return
  }
}

//arith.mul
module @arith_mul{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld = affine.load %18[%arg4] : memref<?xf32>
      %mul = arith.mulf %ld, %ld : f32
      affine.store %mul, %19[%arg4] : memref<?xf32>
    }
    return
  }
}

//arith.add
module @arith_add{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>  ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld1 = affine.load %18[%arg4] : memref<?xf32>
      %ld2 = affine.load %20[%arg4] : memref<?xf32>
      %add = arith.addf %ld1, %ld2 : f32
      %mul = arith.mulf %add, %add : f32
      affine.store %mul, %19[%arg4] : memref<?xf32>
    }
    return
  }
}

//Conditional arith
module @cond_arith{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld = affine.load %18[%arg4] : memref<?xf32>
      %if = scf.if %12 -> f32 {
        %mul = arith.mulf %ld, %ld : f32
        scf.yield %mul : f32
      } else {
        scf.yield %ld : f32
      }
      affine.store %if, %19[%arg4] : memref<?xf32>
    }
    return
  }
}

//reduction
module @reduction{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>  ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    %sum_0 = arith.constant 0.0 : f32
    %red = affine.for %arg4 = 0 to 17 step 1 iter_args(%sum_iter = %sum_0) -> f32 {
      %ld1 = affine.load %18[%arg4] : memref<?xf32>
      %sum_next = arith.addf %sum_iter, %ld1 : f32
      affine.yield %sum_next : f32
    }
    affine.store %red, %19[0] : memref<?xf32>
    return
  }
}

//Conditional store-1
module @cond_store_1 {
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld = affine.load %18[%arg4] : memref<?xf32>
      %mul = arith.mulf %ld, %ld : f32
      scf.if %12 {
        affine.store %mul, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}

//Conditional store-2
module @cond_store_2{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld = affine.load %18[%arg4] : memref<?xf32>
      scf.if %12 {
        %mul = arith.mulf %ld, %ld : f32
        affine.store %mul, %19[%arg4] : memref<?xf32>
      } else {
        affine.store %ld, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}

//Parallel for
module @parallel_for{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      %ld = affine.load %18[%arg4] : memref<?xf32>
      %mul = arith.mulf %ld, %ld : f32
      affine.store %mul, %19[%arg4] : memref<?xf32>
    }
    affine.for %arg4 = 0 to 17 {
      %ld1 = affine.load %18[%arg4] : memref<?xf32>
      %ld2 = affine.load %20[%arg4] : memref<?xf32>
      %add = arith.addf %ld1, %ld2 : f32
      %mul = arith.mulf %add, %add : f32
      affine.store %mul, %19[%arg4] : memref<?xf32>
    }
    return
  }
}

//Fors inside for
module @for_within_for{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %21 = arith.muli %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg3 = 0 to 21 {
      affine.for %arg4 = 0 to 17 {
        %ld1 = affine.load %18[%arg3] : memref<?xf32>
        %ld2 = affine.load %20[%arg4] : memref<?xf32>
        %mul = arith.mulf %ld1, %ld2 : f32
        affine.store %mul, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}

//Parallel fors inside for
module @parallel_fors_inside_for {
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg3 = 0 to 17 {
      affine.for %arg4 = 0 to 17 {
        %ld1 = affine.load %18[%arg3] : memref<?xf32>
        %ld2 = affine.load %20[%arg4] : memref<?xf32>
        %mul = arith.mulf %ld1, %ld2 : f32
        affine.store %mul, %19[%arg4] : memref<?xf32>
      }
      affine.for %arg4 = 0 to 17 {
        %ld1 = affine.load %18[%arg3] : memref<?xf32>
        %ld2 = affine.load %20[%arg4] : memref<?xf32>
        %add = arith.addf %ld1, %ld2 : f32
        %mul = arith.mulf %add, %add : f32
        affine.store %mul, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}

//matrix-mul iter arg
module @matmul_1 {
  memref.global @out : memref<32x8xi32> = uninitialized
  memref.global @im2 : memref<8x8xi32> = uninitialized
  memref.global @im1 : memref<32x8xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @im1 : memref<32x8xi32>
    %1 = memref.get_global @im2 : memref<8x8xi32>
    %2 = memref.get_global @out : memref<32x8xi32>
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 8 {
        %3 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %c0_i32) -> (i32) {
          %4 = affine.load %0[%arg0, %arg2] : memref<32x8xi32>
          %5 = affine.load %1[%arg2, %arg1] : memref<8x8xi32>
          %6 = arith.muli %4, %5 : i32
          %7 = arith.addi %arg3, %6 : i32
          affine.yield %7 : i32
        }
        affine.store %3, %2[%arg0, %arg1] : memref<32x8xi32>
      }
    }
    return %c0_i32 : i32
  }
}

//matrix-mul alias issue
module @matmul_2 {
  memref.global @out : memref<128x32xi32> = uninitialized
  memref.global @im2 : memref<64x32xi32> = uninitialized
  memref.global @im1 : memref<128x64xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @im1 : memref<128x64xi32>
    %1 = memref.get_global @im2 : memref<64x32xi32>
    %2 = memref.get_global @out : memref<128x32xi32>
    affine.for %arg0 = 0 to 128 {
      affine.for %arg1 = 0 to 32 {
        affine.for %arg2 = 0 to 64 {
          %3 = affine.load %0[%arg0, %arg2] : memref<128x64xi32>
          %4 = affine.load %1[%arg2, %arg1] : memref<64x32xi32>
          %5 = arith.muli %3, %4 : i32
          %6 = affine.load %2[%arg0, %arg1] : memref<128x32xi32>
          %7 = arith.addi %6, %5 : i32
          affine.store %7, %2[%arg0, %arg1] : memref<128x32xi32>
        }
      }
    }
    return %c0_i32 : i32
  }
}

//conv (with inner loop accumulate)
//How to deal with IR in outer loops as well?
module @conv_1{
  memref.global @out : memref<512x64xi32> = uninitialized
  memref.global @filter : memref<4x4xi32> = uninitialized
  memref.global @im : memref<515x67xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @im : memref<515x67xi32>
    %1 = memref.get_global @filter : memref<4x4xi32>
    %2 = memref.get_global @out : memref<512x64xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 64 {
        %3 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %c0_i32) -> (i32) {
          %4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %arg3) -> (i32) {
            %5 = affine.load %0[%arg0 + %arg2, %arg1 + %arg4] : memref<515x67xi32>
            %6 = affine.load %1[%arg2, %arg4] : memref<4x4xi32>
            %7 = arith.muli %5, %6 : i32
            %8 = arith.addi %arg5, %7 : i32
            affine.yield %8 : i32
          }
          affine.yield %4 : i32
        }
        affine.store %3, %2[%arg0, %arg1] : memref<512x64xi32>
      }
    }
    return %c0_i32 : i32
  }
}

//conv (direct store)
module @conv_2{
  memref.global @out : memref<512x64xi32> = uninitialized
  memref.global @filter : memref<4x4xi32> = uninitialized
  memref.global @im : memref<515x67xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @im : memref<515x67xi32>
    %1 = memref.get_global @out : memref<512x64xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 64 {
        affine.for %arg2 = 0 to 4 {
          affine.for %arg3 = 0 to 4 {
            %2 = affine.load %0[%arg0 + %arg2, %arg1 + %arg3] : memref<515x67xi32>
            %3 = affine.load %1[%arg0, %arg1] : memref<512x64xi32>
            %4 = arith.addi %3, %2 : i32
            affine.store %4, %1[%arg0, %arg1] : memref<512x64xi32>
          }
        }
      }
    }
    return %c0_i32 : i32
  }
}