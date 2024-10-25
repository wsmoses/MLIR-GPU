//// RUN: polygeist-opt --raise-affine-to-linalg --split-input-file %s | FileCheck %s
//
// module {
//   func.func @main0(%12 : i1, %18 : memref<32xf32> ) {
//     %c0 = arith.constant 0 : index
//     %c4 = arith.constant 4 : index
//     %c1 = arith.constant 1 : index
//     %19 = memref.alloca() : memref<32xf32>
//     scf.if %12 {
//       affine.for %arg4 = 0 to 17 {
//         %ld = affine.load %18[%arg4] : memref<32xf32>
//         affine.store %ld, %19[%arg4] : memref<32xf32>
//       }
//    }
//     return
//   }
  
  // func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
  //   %c0 = arith.constant 0 : index
  //   %c4 = arith.constant 4 : index
  //   %c1 = arith.constant 1 : index
  //   %15 = arith.index_cast %14 : i32 to index
  //   %16 = arith.muli %15, %c4 : index
  //   %17 = arith.divui %16, %c4 : index
  //   %19 = memref.alloca(%17) : memref<?xf32>
  //   scf.if %12 {
  //     affine.for %arg4 = 0 to 17 {
  //       %ld = affine.load %18[%arg4] : memref<?xf32>
  //       affine.store %ld, %19[%arg4] : memref<?xf32>
  //     }
  //  }
  //   return
  // }


 //    func.func @main2(%12 : i1, %14 : i32, %18 : memref<?xf32> ) {
 //      %c0 = arith.constant 0 : index
 //      %c4 = arith.constant 4 : index
 //      %c1 = arith.constant 1 : index
 //      %15 = arith.index_cast %14 : i32 to index
 //      %16 = arith.muli %15, %c4 : index
 //      %17 = arith.divui %16, %c4 : index
 //      %19 = memref.alloca(%17) : memref<?xf32>
 //      scf.if %12 {
 //        affine.for %arg4 = 0 to 17 {
 //          %ld = affine.load %18[3 * %arg4] : memref<?xf32>
 //          %ld2 = affine.load %18[0] : memref<?xf32>
 //          %fadd = arith.addf %ld, %ld2 : f32
 //          affine.store %fadd, %19[%arg4 + 17] : memref<?xf32>
 //        }
 //     }
 //      return
 //    }

 // }

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

//TODO: reduction
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

module @reduction_transformed{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>  ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    %sum_0 = arith.constant 0.0 : f32
    %alloca = memref.alloca() : memref<1xf32> 
    affine.store %sum_0, %alloca[0] : memref<1xf32>
    affine.for %arg4 = 0 to 17 step 1 {
      %iter_arg = affine.load %alloca[0] : memref<1xf32>
      %ld1 = affine.load %18[%arg4] : memref<?xf32>
      %sum_next = arith.addf %iter_arg, %ld1 : f32
      affine.store %sum_next, %alloca[0] : memref<1xf32>
      affine.yield
    }
    %red = affine.load %alloca[0] : memref<1xf32>
    affine.store %red, %19[0] : memref<?xf32>
    return
  }
}

module @reduction_transformed_simplified{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>  ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    %sum_0 = arith.constant 0.0 : f32
    affine.store %sum_0, %19[0] : memref<?xf32>
    affine.for %arg4 = 0 to 17 step 1 {
      %iter_arg = affine.load %19[0] : memref<?xf32>
      %ld1 = affine.load %18[%arg4] : memref<?xf32>
      %sum_next = arith.addf %iter_arg, %ld1 : f32
      affine.store %sum_next, %19[0] : memref<?xf32>
      affine.yield
    }
    return
  }
}
//TODO: Conditional store-1
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

//TODO: Conditional store-2
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

// //Parallel for
// module @parallel_for{
//   func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>) {
//     %c0 = arith.constant 0 : index
//     %c4 = arith.constant 4 : index
//     %c1 = arith.constant 1 : index
//     %15 = arith.index_cast %14 : i32 to index
//     %16 = arith.muli %15, %c4 : index
//     %17 = arith.divui %16, %c4 : index
//     %19 = memref.alloca(%17) : memref<?xf32>
//     affine.for %arg4 = 0 to 17 {
//       %ld = affine.load %18[%arg4] : memref<?xf32>
//       %mul = arith.mulf %ld, %ld : f32
//       affine.store %mul, %19[%arg4] : memref<?xf32>
//     }
//     affine.for %arg4 = 0 to 17 {
//       %ld1 = affine.load %18[%arg4] : memref<?xf32>
//       %ld2 = affine.load %20[%arg4] : memref<?xf32>
//       %add = arith.addf %ld1, %ld2 : f32
//       %mul = arith.mulf %add, %add : f32
//       affine.store %mul, %19[%arg4] : memref<?xf32>
//     }
//     return
//   }
// }

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

//Fors inside for
module @for_within_for_2{
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
        %ld1 = affine.load %18[%arg3+2*%arg4] : memref<?xf32>
        %ld2 = affine.load %20[%arg4] : memref<?xf32>
        %mul = arith.mulf %ld1, %ld2 : f32
        affine.store %mul, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}

//Fors inside for
module @for_within_for_3{
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
        %ld1 = affine.load %18[%arg3+2*%arg4] : memref<?xf32>
        %ld2 = affine.load %18[%arg3] : memref<?xf32>
        %ld3 = affine.load %20[%arg4] : memref<?xf32>
        %mul = arith.mulf %ld1, %ld2 : f32
        %mul2 = arith.mulf %mul, %ld3 : f32
        affine.store %mul2, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}

//Fors inside for
module @for_within_for_4{
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
        %ld1 = affine.load %18[%arg4+2*%arg3] : memref<?xf32>
        %ld2 = affine.load %20[%arg4] : memref<?xf32>
        %mul = arith.mulf %ld1, %ld2 : f32
        affine.store %mul, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}

//Fors no-loop dependency
module @for_no_loop_dependency{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>, %23 : memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %21 = arith.muli %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg3 = 0 to 15 {
          %ld1 = affine.load %18[0] : memref<?xf32>
          affine.store %ld1, %19[0] : memref<?xf32>
    }
    return
  }
}
//Fors no-loop dependency
module @for_2_levels_no_loop_dependency{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>, %23 : memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %21 = arith.muli %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg4 = 0 to 17 {
      affine.for %arg3 = 0 to 15 {
            %ld1 = affine.load %18[%arg4] : memref<?xf32>
            affine.store %ld1, %19[%arg4] : memref<?xf32>
      }
    }
    return
  }
}
//Fors inside for
module @for_3_levels_0{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>, %23 : memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %21 = arith.muli %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg3 = 0 to 15 {
      affine.for %arg4 = 0 to 17 {
        affine.for %arg5 = 0 to 21 {
          %ld1 = affine.load %18[%arg3] : memref<?xf32>
          %ld2 = affine.load %20[%arg4] : memref<?xf32>
          %mul = arith.mulf %ld1, %ld2 : f32
          affine.store %mul, %19[%arg5] : memref<?xf32>
        }
      }
    }
    return
  }
}

//Fors inside for
module @for_3_levels_1{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>, %23 : memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.muli %15, %c4 : index
    %17 = arith.divui %16, %c4 : index
    %21 = arith.muli %16, %c4 : index
    %19 = memref.alloca(%17) : memref<?xf32>
    affine.for %arg5 = 0 to 21 {
      affine.for %arg3 = 0 to 21 {
        affine.for %arg4 = 0 to 17 {
          %ld1 = affine.load %18[%arg3] : memref<?xf32>
          %ld2 = affine.load %20[%arg4] : memref<?xf32>
          %mul = arith.mulf %ld1, %ld2 : f32
          affine.store %mul, %19[%arg4] : memref<?xf32>
        }
      }
    }
    return
  }
}

//Fors inside for
module @for_3_levels_2{
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>, %23 : memref<?xf32>) {
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
        affine.for %arg5 = 0 to 21 {
          %ld1 = affine.load %18[%arg3] : memref<?xf32>
          %ld2 = affine.load %20[%arg4] : memref<?xf32>
          %ld3 = affine.load %23[%arg5] : memref<?xf32>
          %mul = arith.mulf %ld1, %ld2 : f32
          %mul2 = arith.mulf %mul, %ld3 : f32
          affine.store %mul2, %19[%arg4] : memref<?xf32>
        }
      }
    }
    return
  }
}

//Fors inside for
module @for_3_levels_3{
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
        affine.for %arg5 = 0 to 21 {
          %ld1 = affine.load %18[%arg3] : memref<?xf32>
          %ld2 = affine.load %20[%arg4] : memref<?xf32>
          %ld3 = affine.load %20[%arg5] : memref<?xf32>
          %mul = arith.mulf %ld1, %ld2 : f32
          %mul2 = arith.mulf %mul, %ld3 : f32
          affine.store %mul2, %19[%arg4] : memref<?xf32>
        }
      }
    }
    return
  }
}

//Fors inside for
module @for_3_levels_4{
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
        affine.for %arg5 = 0 to 21 {
          %ld1 = affine.load %18[%arg3+4*%arg4+3] : memref<?xf32>
          %ld2 = affine.load %20[7*%arg4+%arg5+2] : memref<?xf32>
          %ld3 = affine.load %20[%arg5+2*%arg3] : memref<?xf32>
          %mul = arith.mulf %ld1, %ld2 : f32
          %mul2 = arith.mulf %mul, %ld3 : f32
          affine.store %mul2, %19[%arg4] : memref<?xf32>
        }
      }
    }
    return
  }
}

//Intermediate raising
#map = affine_map<(d0)[s0] -> (s0)>
#map1 = affine_map<(d0) -> (d0)>
module @for_within_for2 {
  func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
    %c17 = arith.constant 17 : index
    %c4 = arith.constant 4 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.muli %0, %c4 : index
    %2 = arith.divui %1, %c4 : index
    %alloca = memref.alloca(%2) : memref<?xf32>
    affine.for %arg4 = 0 to 21 {
      %3 = "polygeist.submap"(%arg2, %arg4, %c17) <{map = #map}> : (memref<?xf32>, index, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%arg3, %c17) <{map = #map1}> : (memref<?xf32>, index) -> memref<?xf32>
      %5 = "polygeist.submap"(%alloca, %c17) <{map = #map1}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%3, %4 : memref<?xf32>, memref<?xf32>) outs(%5 : memref<?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      }
    }
    return
  }
}

// //Parallel fors inside for
// module @parallel_fors_inside_for {
//   func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %20 : memref<?xf32>) {
//     %c0 = arith.constant 0 : index
//     %c4 = arith.constant 4 : index
//     %c1 = arith.constant 1 : index
//     %15 = arith.index_cast %14 : i32 to index
//     %16 = arith.muli %15, %c4 : index
//     %17 = arith.divui %16, %c4 : index
//     %19 = memref.alloca(%17) : memref<?xf32>
//     affine.for %arg3 = 0 to 17 {
//       affine.for %arg4 = 0 to 17 {
//         %ld1 = affine.load %18[%arg3] : memref<?xf32>
//         %ld2 = affine.load %20[%arg4] : memref<?xf32>
//         %mul = arith.mulf %ld1, %ld2 : f32
//         affine.store %mul, %19[%arg4] : memref<?xf32>
//       }
//       affine.for %arg4 = 0 to 17 {
//         %ld1 = affine.load %18[%arg3] : memref<?xf32>
//         %ld2 = affine.load %20[%arg4] : memref<?xf32>
//         %add = arith.addf %ld1, %ld2 : f32
//         %mul = arith.mulf %add, %add : f32
//         affine.store %mul, %19[%arg4] : memref<?xf32>
//       }
//     }
//     return
//   }
// }

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

//matrix-mul extra load-store variant
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

module @conv_1_reduction_test{
  memref.global @out : memref<512x64xi32> = uninitialized
  memref.global @filter : memref<4x4xi32> = uninitialized
  memref.global @im : memref<515x67xi32> = uninitialized
  func.func @main(%arg0 : index, %arg1 : index) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @im : memref<515x67xi32>
    %1 = memref.get_global @filter : memref<4x4xi32>
    %2 = memref.get_global @out : memref<512x64xi32>
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
    return %c0_i32 : i32
  }
}

//conv (direct store)
 module @conv_2 {
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
         affine.for %arg2 = 0 to 4 {
           affine.for %arg3 = 0 to 4 {
             %3 = affine.load %0[%arg0 + %arg2, %arg1 + %arg3] : memref<515x67xi32>
             %4 = affine.load %1[%arg2, %arg3] : memref<4x4xi32>
             %5 = arith.muli %3, %4 : i32
             %6 = affine.load %2[%arg0, %arg1] : memref<512x64xi32>
             %7 = arith.addi %6, %5 : i32
             affine.store %7, %2[%arg0, %arg1] : memref<512x64xi32>
           }
         }
       }
     }
     return %c0_i32 : i32
   }
 }

//box_filter (direct store)
 module @box_filter {
   memref.global @out : memref<512x64xi32> = uninitialized
   memref.global @filter : memref<4x4xi32> = uninitialized
   memref.global @im : memref<515x67xi32> = uninitialized
   func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
     %c0_i32 = arith.constant 0 : i32
     %0 = memref.get_global @im : memref<515x67xi32>
     %2 = memref.get_global @out : memref<512x64xi32>
     affine.for %arg0 = 0 to 512 {
       affine.for %arg1 = 0 to 64 {
         affine.for %arg2 = 0 to 4 {
           affine.for %arg3 = 0 to 4 {
             %3 = affine.load %0[%arg0 + %arg2, %arg1 + %arg3] : memref<515x67xi32>
             %6 = affine.load %2[%arg0, %arg1] : memref<512x64xi32>
             %7 = arith.addi %6, %3 : i32
             affine.store %7, %2[%arg0, %arg1] : memref<512x64xi32>
           }
         }
       }
     }
     return %c0_i32 : i32
   }
 }

  module @conv_loop1_test {
    memref.global @out : memref<512x64xi32> = uninitialized
    memref.global @filter : memref<4x4xi32> = uninitialized
    memref.global @im : memref<515x67xi32> = uninitialized
    func.func @main(%arg0 : index, %arg1 : index, %arg2 : index) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @im : memref<515x67xi32>
      %1 = memref.get_global @filter : memref<4x4xi32>
      %2 = memref.get_global @out : memref<512x64xi32>
            affine.for %arg3 = 0 to 4 {
              %3 = affine.load %0[%arg0 + %arg2, %arg1 + %arg3] : memref<515x67xi32>
              %4 = affine.load %1[%arg2, %arg3] : memref<4x4xi32>
              %5 = arith.muli %3, %4 : i32
              %6 = affine.load %2[%arg0, %arg1] : memref<512x64xi32>
              %7 = arith.addi %6, %5 : i32
              affine.store %7, %2[%arg0, %arg1] : memref<512x64xi32>
            }
      return %c0_i32 : i32
    }
  }  

 module @submap_test {
   memref.global @out : memref<511x64xi32> = uninitialized
   memref.global @filter : memref<5x4xi32> = uninitialized
   memref.global @im : memref<515x67xi32> = uninitialized
   func.func @main(%arg0 : index, %arg1 : index) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
     %c0_i32 = arith.constant 0 : i32
     %0 = memref.get_global @im : memref<515x67xi32>
     %1 = memref.get_global @filter : memref<5x4xi32>
     %2 = memref.get_global @out : memref<511x64xi32>
         affine.for %arg2 = 0 to 5 {
           affine.for %arg3 = 0 to 4 {
             %3 = affine.load %0[%arg0 + %arg2, %arg1 + %arg3] : memref<515x67xi32>
             %4 = affine.load %1[%arg2, %arg3] : memref<5x4xi32>
             %5 = arith.muli %3, %4 : i32
             %6 = affine.load %2[%arg0, %arg1] : memref<511x64xi32>
             %7 = arith.addi %6, %5 : i32
             affine.store %7, %2[%arg0, %arg1] : memref<511x64xi32>
           }
         }
     return %c0_i32 : i32
   }
 } 


module @harris_score_1{
 memref.global @coeffs_y : memref<9xi32> = dense<[-3, -10, -3, 0, 0, 0, 3, 10, 3]>
  memref.global @coeffs_x : memref<9xi32> = dense<[-3, 0, 3, -10, 0, 10, -3, 0, 3]>
  memref.global @score : memref<512x512xi32> = uninitialized
  memref.global @img_ixy : memref<512x512xi32> = uninitialized
  memref.global @img_iyy : memref<512x512xi32> = uninitialized
  memref.global @img_ixx : memref<512x512xi32> = uninitialized
  memref.global @img_in : memref<518x518xi32> = uninitialized
  memref.global @img_gy : memref<516x516xi32> = uninitialized
  memref.global @img_gx : memref<516x516xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @img_gx : memref<516x516xi32>
    %1 = memref.get_global @img_gy : memref<516x516xi32>
    %2 = memref.get_global @img_in : memref<518x518xi32>
    %3 = memref.get_global @coeffs_x : memref<9xi32>
    %4 = memref.get_global @coeffs_y : memref<9xi32>
    affine.for %arg0 = 0 to 516 {
      affine.for %arg1 = 0 to 516 {
        affine.for %arg2 = 0 to 3 {
          affine.for %arg5 = 0 to 3 {
            %gx = affine.load %0[%arg0, %arg1] : memref<516x516xi32>
            %gy = affine.load %1[%arg0, %arg1] : memref<516x516xi32>
            %11 = affine.load %2[%arg0 + %arg2, %arg1 + %arg5] : memref<518x518xi32>
            %12 = affine.load %3[%arg5 + %arg2 * 3] : memref<9xi32>
            %13 = arith.muli %11, %12 : i32
            %14 = arith.addi %gx, %13 : i32
            %15 = affine.load %4[%arg5 + %arg2 * 3] : memref<9xi32>
            %16 = arith.muli %11, %15 : i32
            %17 = arith.addi %gy, %16 : i32
            affine.store %14, %0[%arg0, %arg1] : memref<516x516xi32>
            affine.store %17, %1[%arg0, %arg1] : memref<516x516xi32>
          }
        }
      }
    }
    %5 = memref.get_global @img_ixx : memref<512x512xi32>
    %6 = memref.get_global @img_iyy : memref<512x512xi32>
    %7 = memref.get_global @img_ixy : memref<512x512xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        affine.for %arg2 = 0 to 5 {
          affine.for %arg6 = 0 to 5 {
            %ixx = affine.load %5[%arg0, %arg1] : memref<512x512xi32>
            %iyy = affine.load %6[%arg0, %arg1] : memref<512x512xi32>
            %ixy = affine.load %7[%arg0, %arg1] : memref<512x512xi32>
            %11 = affine.load %0[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %12 = affine.load %1[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %13 = arith.muli %11, %11 : i32
            %14 = arith.addi %ixx, %13 : i32
            %15 = arith.muli %12, %12 : i32
            %16 = arith.addi %iyy, %15 : i32
            %17 = arith.muli %11, %12 : i32
            %18 = arith.addi %ixy, %17 : i32
            affine.store %14, %5[%arg0, %arg1] : memref<512x512xi32>
            affine.store %16, %6[%arg0, %arg1] : memref<512x512xi32>
            affine.store %18, %7[%arg0, %arg1] : memref<512x512xi32>
          }
        }
      }
    }
    %8 = memref.get_global @score : memref<512x512xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %9 = affine.load %5[%arg0, %arg1] : memref<512x512xi32>
        %10 = affine.load %6[%arg0, %arg1] : memref<512x512xi32>
        %11 = affine.load %7[%arg0, %arg1] : memref<512x512xi32>
        %12 = arith.muli %9, %10 : i32
        %13 = arith.muli %11, %11 : i32
        %14 = arith.subi %12, %13 : i32
        %15 = arith.addi %9, %10 : i32
        %16 = arith.muli %15, %c4_i32 : i32
        %17 = arith.muli %16, %15 : i32
        %18 = arith.subi %14, %17 : i32
        affine.store %18, %8[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    return %c0_i32 : i32
  }
}

module @harris_score_2 {
    memref.global @coeffs_y : memref<9xi32> = dense<[-3, -10, -3, 0, 0, 0, 3, 10, 3]>
  memref.global @coeffs_x : memref<9xi32> = dense<[-3, 0, 3, -10, 0, 10, -3, 0, 3]>
  memref.global @score : memref<512x512xi32> = uninitialized
  memref.global @img_ixy : memref<512x512xi32> = uninitialized
  memref.global @img_iyy : memref<512x512xi32> = uninitialized
  memref.global @img_ixx : memref<512x512xi32> = uninitialized
  memref.global @img_in : memref<518x518xi32> = uninitialized
  memref.global @img_gy : memref<516x516xi32> = uninitialized
  memref.global @img_gx : memref<516x516xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @img_gx : memref<516x516xi32>
    %1 = memref.get_global @img_gy : memref<516x516xi32>
    %2 = memref.get_global @img_in : memref<518x518xi32>
    %3 = memref.get_global @coeffs_x : memref<9xi32>
    %4 = memref.get_global @coeffs_y : memref<9xi32>
    affine.for %arg0 = 0 to 516 {
      affine.for %arg1 = 0 to 516 {
        %9:2 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
          %10:2 = affine.for %arg5 = 0 to 3 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (i32, i32) {
            %11 = affine.load %2[%arg0 + %arg2, %arg1 + %arg5] : memref<518x518xi32>
            %12 = affine.load %3[%arg5 + %arg2 * 3] : memref<9xi32>
            %13 = arith.muli %11, %12 : i32
            %14 = arith.addi %arg7, %13 : i32
            %15 = affine.load %4[%arg5 + %arg2 * 3] : memref<9xi32>
            %16 = arith.muli %11, %15 : i32
            %17 = arith.addi %arg6, %16 : i32
            affine.yield %17, %14 : i32, i32
          }
          affine.yield %10#0, %10#1 : i32, i32
        }
        affine.store %9#1, %0[%arg0, %arg1] : memref<516x516xi32>
        affine.store %9#0, %1[%arg0, %arg1] : memref<516x516xi32>
      }
    }
    %5 = memref.get_global @img_ixx : memref<512x512xi32>
    %6 = memref.get_global @img_iyy : memref<512x512xi32>
    %7 = memref.get_global @img_ixy : memref<512x512xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %9:3 = affine.for %arg2 = 0 to 5 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32) -> (i32, i32, i32) {
          %10:3 = affine.for %arg6 = 0 to 5 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (i32, i32, i32) {
            %11 = affine.load %0[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %12 = affine.load %1[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %13 = arith.muli %11, %11 : i32
            %14 = arith.addi %arg9, %13 : i32
            %15 = arith.muli %12, %12 : i32
            %16 = arith.addi %arg8, %15 : i32
            %17 = arith.muli %11, %12 : i32
            %18 = arith.addi %arg7, %17 : i32
            affine.yield %18, %16, %14 : i32, i32, i32
          }
          affine.yield %10#0, %10#1, %10#2 : i32, i32, i32
        }
        affine.store %9#2, %5[%arg0, %arg1] : memref<512x512xi32>
        affine.store %9#1, %6[%arg0, %arg1] : memref<512x512xi32>
        affine.store %9#0, %7[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    %8 = memref.get_global @score : memref<512x512xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %9 = affine.load %5[%arg0, %arg1] : memref<512x512xi32>
        %10 = affine.load %6[%arg0, %arg1] : memref<512x512xi32>
        %11 = affine.load %7[%arg0, %arg1] : memref<512x512xi32>
        %12 = arith.muli %9, %10 : i32
        %13 = arith.muli %11, %11 : i32
        %14 = arith.subi %12, %13 : i32
        %15 = arith.addi %9, %10 : i32
        %16 = arith.muli %15, %c4_i32 : i32
        %17 = arith.muli %16, %15 : i32
        %18 = arith.subi %14, %17 : i32
        affine.store %18, %8[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    return %c0_i32 : i32
  }
}

module @harris_score_local {
  memref.global @coeffs_y : memref<9xi32> = dense<[-3, -10, -3, 0, 0, 0, 3, 10, 3]>
  memref.global @coeffs_x : memref<9xi32> = dense<[-3, 0, 3, -10, 0, 10, -3, 0, 3]>
  memref.global @score : memref<512x512xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<512x512xi32>
    %alloca_0 = memref.alloca() : memref<512x512xi32>
    %alloca_1 = memref.alloca() : memref<512x512xi32>
    %alloca_2 = memref.alloca() : memref<516x516xi32>
    %alloca_3 = memref.alloca() : memref<516x516xi32>
    %alloca_4 = memref.alloca() : memref<518x518xi32>
    %0 = memref.get_global @coeffs_x : memref<9xi32>
    %1 = memref.get_global @coeffs_y : memref<9xi32>
    affine.for %arg0 = 0 to 516 {
      affine.for %arg1 = 0 to 516 {
        affine.for %arg2 = 0 to 3 {
          affine.for %arg5 = 0 to 3 {
            %gx = affine.load %alloca_3[%arg0, %arg1] : memref<516x516xi32>
            %gy = affine.load %alloca_2[%arg0, %arg1] : memref<516x516xi32>
            %5 = affine.load %alloca_4[%arg0 + %arg2, %arg1 + %arg5] : memref<518x518xi32>
            %6 = affine.load %0[%arg5 + %arg2 * 3] : memref<9xi32>
            %7 = arith.muli %5, %6 : i32
            %8 = arith.addi %gx, %7 : i32
            %9 = affine.load %1[%arg5 + %arg2 * 3] : memref<9xi32>
            %10 = arith.muli %5, %9 : i32
            %11 = arith.addi %gy, %10 : i32
            affine.store %8, %alloca_3[%arg0, %arg1] : memref<516x516xi32>
            affine.store %11, %alloca_2[%arg0, %arg1] : memref<516x516xi32>
          }
        }
      }
    }
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %3:3 = affine.for %arg2 = 0 to 5 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32) -> (i32, i32, i32) {
          %4:3 = affine.for %arg6 = 0 to 5 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (i32, i32, i32) {
            %ixx = affine.load %alloca_1[%arg0, %arg1] : memref<512x512xi32>
            %iyy = affine.load %alloca_0[%arg0, %arg1] : memref<512x512xi32>
            %ixy = affine.load %alloca[%arg0, %arg1] : memref<512x512xi32>
            %5 = affine.load %alloca_3[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %6 = affine.load %alloca_2[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %7 = arith.muli %5, %5 : i32
            %8 = arith.addi %arg9, %7 : i32
            %9 = arith.muli %6, %6 : i32
            %10 = arith.addi %arg8, %9 : i32
            %11 = arith.muli %5, %6 : i32
            %12 = arith.addi %arg7, %11 : i32
            affine.yield %12, %10, %8 : i32, i32, i32
          }
          affine.yield %4#0, %4#1, %4#2 : i32, i32, i32
        }
        affine.store %3#2, %alloca_1[%arg0, %arg1] : memref<512x512xi32>
        affine.store %3#1, %alloca_0[%arg0, %arg1] : memref<512x512xi32>
        affine.store %3#0, %alloca[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    %2 = memref.get_global @score : memref<512x512xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %3 = affine.load %alloca_1[%arg0, %arg1] : memref<512x512xi32>
        %4 = affine.load %alloca_0[%arg0, %arg1] : memref<512x512xi32>
        %5 = affine.load %alloca[%arg0, %arg1] : memref<512x512xi32>
        %6 = arith.muli %3, %4 : i32
        %7 = arith.muli %5, %5 : i32
        %8 = arith.subi %6, %7 : i32
        %9 = arith.addi %3, %4 : i32
        %10 = arith.muli %9, %c4_i32 : i32
        %11 = arith.muli %10, %9 : i32
        %12 = arith.subi %8, %11 : i32
        affine.store %12, %2[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    return %c0_i32 : i32
  }
}

module @harris_score_2d_kernel {
  memref.global "private" @_ZL8coeffs_y : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global "private" @_ZL8coeffs_x : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global @score : memref<512x512xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<512x512xi32>
    %alloca_0 = memref.alloca() : memref<512x512xi32>
    %alloca_1 = memref.alloca() : memref<512x512xi32>
    %alloca_2 = memref.alloca() : memref<516x516xi32>
    %alloca_3 = memref.alloca() : memref<516x516xi32>
    %alloca_4 = memref.alloca() : memref<518x518xi32>
    %0 = memref.get_global @_ZL8coeffs_x : memref<3x3xi32>
    %1 = memref.get_global @_ZL8coeffs_y : memref<3x3xi32>
    affine.for %arg0 = 0 to 516 {
      affine.for %arg1 = 0 to 516 {
        %3:2 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
          %4:2 = affine.for %arg5 = 0 to 3 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (i32, i32) {
            %5 = affine.load %alloca_4[%arg0 + %arg2, %arg1 + %arg5] : memref<518x518xi32>
            %6 = affine.load %0[%arg2, %arg5] : memref<3x3xi32>
            %7 = arith.muli %5, %6 : i32
            %8 = arith.addi %arg7, %7 : i32
            %9 = affine.load %1[%arg2, %arg5] : memref<3x3xi32>
            %10 = arith.muli %5, %9 : i32
            %11 = arith.addi %arg6, %10 : i32
            affine.yield %11, %8 : i32, i32
          }
          affine.yield %4#0, %4#1 : i32, i32
        }
        affine.store %3#1, %alloca_3[%arg0, %arg1] : memref<516x516xi32>
        affine.store %3#0, %alloca_2[%arg0, %arg1] : memref<516x516xi32>
      }
    }
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %3:3 = affine.for %arg2 = 0 to 5 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32) -> (i32, i32, i32) {
          %4:3 = affine.for %arg6 = 0 to 5 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (i32, i32, i32) {
            %5 = affine.load %alloca_3[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %6 = affine.load %alloca_2[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %7 = arith.muli %5, %5 : i32
            %8 = arith.addi %arg9, %7 : i32
            %9 = arith.muli %6, %6 : i32
            %10 = arith.addi %arg8, %9 : i32
            %11 = arith.muli %5, %6 : i32
            %12 = arith.addi %arg7, %11 : i32
            affine.yield %12, %10, %8 : i32, i32, i32
          }
          affine.yield %4#0, %4#1, %4#2 : i32, i32, i32
        }
        affine.store %3#2, %alloca_1[%arg0, %arg1] : memref<512x512xi32>
        affine.store %3#1, %alloca_0[%arg0, %arg1] : memref<512x512xi32>
        affine.store %3#0, %alloca[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    %2 = memref.get_global @score : memref<512x512xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %3 = affine.load %alloca_1[%arg0, %arg1] : memref<512x512xi32>
        %4 = affine.load %alloca_0[%arg0, %arg1] : memref<512x512xi32>
        %5 = affine.load %alloca[%arg0, %arg1] : memref<512x512xi32>
        %6 = arith.muli %3, %4 : i32
        %7 = arith.muli %5, %5 : i32
        %8 = arith.subi %6, %7 : i32
        %9 = arith.addi %3, %4 : i32
        %10 = arith.muli %9, %c4_i32 : i32
        %11 = arith.muli %10, %9 : i32
        %12 = arith.subi %8, %11 : i32
        affine.store %12, %2[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    return %c0_i32 : i32
  }
}

module @harris_score_with_gradient_extra_kernel {
  memref.global "private" @_ZL8coeffs_1 : memref<5x5xi32> = dense<1>
  memref.global "private" @_ZL8coeffs_y : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global "private" @_ZL8coeffs_x : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global @score : memref<512x512xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<512x512xi32>
    %alloca_0 = memref.alloca() : memref<512x512xi32>
    %alloca_1 = memref.alloca() : memref<512x512xi32>
    %alloca_2 = memref.alloca() : memref<516x516xi32>
    %alloca_3 = memref.alloca() : memref<516x516xi32>
    %alloca_4 = memref.alloca() : memref<518x518xi32>
    %0 = memref.get_global @_ZL8coeffs_x : memref<3x3xi32>
    %1 = memref.get_global @_ZL8coeffs_y : memref<3x3xi32>
    affine.for %arg0 = 0 to 516 {
      affine.for %arg1 = 0 to 516 {
        %4:2 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
          %5:2 = affine.for %arg5 = 0 to 3 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (i32, i32) {
            %6 = affine.load %alloca_4[%arg0 + %arg2, %arg1 + %arg5] : memref<518x518xi32>
            %7 = affine.load %0[%arg2, %arg5] : memref<3x3xi32>
            %8 = arith.muli %6, %7 : i32
            %9 = arith.addi %arg7, %8 : i32
            %10 = affine.load %1[%arg2, %arg5] : memref<3x3xi32>
            %11 = arith.muli %6, %10 : i32
            %12 = arith.addi %arg6, %11 : i32
            affine.yield %12, %9 : i32, i32
          }
          affine.yield %5#0, %5#1 : i32, i32
        }
        affine.store %4#1, %alloca_3[%arg0, %arg1] : memref<516x516xi32>
        affine.store %4#0, %alloca_2[%arg0, %arg1] : memref<516x516xi32>
      }
    }
    %2 = memref.get_global @_ZL8coeffs_1 : memref<5x5xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %4:3 = affine.for %arg2 = 0 to 5 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32) -> (i32, i32, i32) {
          %5:3 = affine.for %arg6 = 0 to 5 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (i32, i32, i32) {
            %6 = affine.load %alloca_3[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %7 = affine.load %alloca_2[%arg0 + %arg2, %arg1 + %arg6] : memref<516x516xi32>
            %8 = arith.muli %6, %6 : i32
            %9 = affine.load %2[%arg2, %arg6] : memref<5x5xi32>
            %10 = arith.muli %8, %9 : i32
            %11 = arith.addi %arg9, %10 : i32
            %12 = arith.muli %7, %7 : i32
            %13 = arith.muli %12, %9 : i32
            %14 = arith.addi %arg8, %13 : i32
            %15 = arith.muli %6, %7 : i32
            %16 = arith.muli %15, %9 : i32
            %17 = arith.addi %arg7, %16 : i32
            affine.yield %17, %14, %11 : i32, i32, i32
          }
          affine.yield %5#0, %5#1, %5#2 : i32, i32, i32
        }
        affine.store %4#2, %alloca_1[%arg0, %arg1] : memref<512x512xi32>
        affine.store %4#1, %alloca_0[%arg0, %arg1] : memref<512x512xi32>
        affine.store %4#0, %alloca[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    %3 = memref.get_global @score : memref<512x512xi32>
    affine.for %arg0 = 0 to 512 {
      affine.for %arg1 = 0 to 512 {
        %4 = affine.load %alloca_1[%arg0, %arg1] : memref<512x512xi32>
        %5 = affine.load %alloca_0[%arg0, %arg1] : memref<512x512xi32>
        %6 = affine.load %alloca[%arg0, %arg1] : memref<512x512xi32>
        %7 = arith.muli %4, %5 : i32
        %8 = arith.muli %6, %6 : i32
        %9 = arith.subi %7, %8 : i32
        %10 = arith.addi %4, %5 : i32
        %11 = arith.muli %10, %c4_i32 : i32
        %12 = arith.muli %11, %10 : i32
        %13 = arith.subi %9, %12 : i32
        affine.store %13, %3[%arg0, %arg1] : memref<512x512xi32>
      }
    }
    return %c0_i32 : i32
  }
}