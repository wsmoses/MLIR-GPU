// RUN: polygeist-opt --inline --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @eval_cost(%arg0: !llvm.ptr, %arg1: i1) -> memref<1x!llvm.struct<(f32, array<3 x f32>)>> {
    %1 = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<1x!llvm.struct<(f32, array<3 x f32>)>>
    return %1 : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
  }
  func.func @eval_res(%arg0: memref<1x!llvm.struct<(f32, array<3 x f32>)>>) -> !llvm.struct<(f32, array<3 x f32>)>
  {
    %c0_i1 = arith.constant 0 : i1
    %alloca = memref.alloca() : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
    %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(f32, array<3 x f32>)>>) -> !llvm.ptr
    %1 = func.call @eval_cost(%0, %c0_i1) : (!llvm.ptr, i1) ->  memref<1x!llvm.struct<(f32, array<3 x f32>)>>
    %2 = affine.load %1[0] : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
    return %2 : !llvm.struct<(f32, array<3 x f32>)>
  }

// CHECK-LABEL:   func.func @eval_cost(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                         %[[VAL_1:.*]]: i1) -> memref<1x!llvm.struct<(f32, array<3 x f32>)>> {
// CHECK:           %[[VAL_2:.*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<1x!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:           return %[[VAL_2]] : memref<1x!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:         }

// CHECK-LABEL:   func.func @eval_res(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<1x!llvm.struct<(f32, array<3 x f32>)>>) -> !llvm.struct<(f32, array<3 x f32>)> {
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_1]][] : memref<!llvm.struct<(f32, array<3 x f32>)>>
// CHECK:           return %[[VAL_2]] : !llvm.struct<(f32, array<3 x f32>)>
// CHECK:         }

  func.func private @use(%arg0: index, %arg1: index) -> index{
    %0 = arith.addi %arg0, %arg1 : index
    return %0 : index
  }
  
  func.func @f1(%gd : index, %bd : index) {
    %mc0 = arith.constant 0 : index
    %mc4 = arith.constant 4 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"() ({
      affine.parallel (%a1, %a2, %a3) = (0, 0, 0) to (%gd, %mc4, %bd) {
        "polygeist.noop"(%a3, %mc0, %mc0) {polygeist.noop_type="gpu_kernel.thread_only"} : (index, index, index) -> ()
        %a1r = func.call @use(%a1,%mc4) : (index, index) -> (index)
        %a2r = func.call @use(%a2,%a1r) : (index, index) -> (index)
        %a3r = func.call @use(%a3,%a2r) : (index, index) -> (index)
        "test.something"(%a3r) : (index) -> ()
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = "polygeist.gpu_wrapper"() ({
// CHECK:             affine.parallel (%[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) = (0, 0, 0) to (symbol(%[[VAL_0]]), 4, symbol(%[[VAL_1]])) {
// CHECK:               "polygeist.noop"(%[[VAL_7]], %[[VAL_2]], %[[VAL_2]]) {polygeist.noop_type = "gpu_kernel.thread_only"} : (index, index, index) -> ()
// CHECK:               %[[VAL_8:.*]] = arith.addi %[[VAL_5]], %[[VAL_3]] : index
// CHECK:               %[[VAL_9:.*]] = arith.addi %[[VAL_6]], %[[VAL_8]] : index
// CHECK:               %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:               "test.something"(%[[VAL_10]]) : (index) -> ()
// CHECK:             }
// CHECK:             "polygeist.polygeist_yield"() : () -> ()
// CHECK:           }) : () -> index
// CHECK:           return
// CHECK:         }

}