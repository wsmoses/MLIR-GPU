// RUN: mlir-opt %s -test-transform-dialect-interpreter --one-shot-bufferize="bufferize-function-boundaries" --func-bufferize --tensor-bufferize  --finalizing-bufferize --convert-linalg-to-affine-loops --raise-scf-to-affine -split-input-file -verify-diagnostics | FileCheck %s
// To test bufferization : pva-opt %s -test-transform-dialect-interpreter --one-shot-bufferize="bufferize-function-boundaries test-analysis-only print-conflicts"
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//#trait_conv = {
//  indexing_maps = [
//    affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>,
//    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
//    affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//  ],
//  iterator_types = ["parallel", "parallel", "reduction", "reduction"]
//}
//
////Remember to tile basd on output
//func.func @conv(%A : tensor<130x130xf32>, %B : tensor<3x3xf32>,
//                       %C : tensor<128x128xf32>) -> tensor<128x128xf32> {
//    %1 = linalg.generic #trait_conv
//      ins(%A, %B : tensor<130x130xf32>,
//                   tensor<3x3xf32>)
//      outs(%C : tensor<128x128xf32>) {
//      ^bb0(%a: f32, %b: f32, %c: f32) :
//        %d = arith.mulf %a, %b: f32
//        %e = arith.addf %c, %d: f32
//        linalg.yield %e : f32
//    } -> tensor<128x128xf32>
//    return %1 : tensor<128x128xf32>
//}
memref.global @out : memref<512x64xi32> = uninitialized
memref.global @rhs : memref<64x64xi32> = uninitialized
memref.global @filter : memref<4x4xi32> = uninitialized
memref.global @im : memref<515x67xi32> = uninitialized
// func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
//   %c512 = arith.constant 512 : index
//   %c64 = arith.constant 64 : index
//   %c4 = arith.constant 4 : index
//   %c0_i32 = arith.constant 0 : i32
//   %0 = memref.get_global @im : memref<515x67xi32>
//   %1 = memref.get_global @filter : memref<4x4xi32>
//   %2 = memref.get_global @out : memref<512x64xi32>
//   %rhs_memref = memref.get_global @rhs : memref<64x64xi32>
//   %4 = bufferization.to_tensor %0 : memref<515x67xi32>
//   %5 = bufferization.to_tensor %1 : memref<4x4xi32>
//   %x = tensor.empty() : tensor<512x64xi32> 
//   %out = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%4, %5 : tensor<515x67xi32>, tensor<4x4xi32>) outs(%x : tensor<512x64xi32>) {
//   ^bb0(%in: i32, %in_0: i32, %out: i32):
//     %6 = arith.muli %in, %in_0 : i32
//     %7 = arith.addi %out, %6 : i32
//     linalg.yield %7 : i32
//   } -> tensor<512x64xi32>
  
//   %materialize = bufferization.to_memref %out : memref<512x64xi32>
//   memref.copy %materialize, %2 : memref<512x64xi32> to memref<512x64xi32>
  
//   %conv_out = bufferization.to_tensor %2 : memref<512x64xi32>
//   %rhs = bufferization.to_tensor %rhs_memref : memref<64x64xi32>
//   %y = tensor.empty() : tensor<512x64xi32> 
//   %matmul = linalg.matmul ins(%conv_out, %rhs: tensor<512x64xi32>, tensor<64x64xi32>)
//                           outs(%y: tensor<512x64xi32>) -> tensor<512x64xi32>
//   %materialize2 = bufferization.to_memref %matmul : memref<512x64xi32>
//   memref.copy %materialize2, %2 : memref<512x64xi32> to memref<512x64xi32>
//   return %c0_i32 : i32
// }

func.func @main_opt() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.get_global @im : memref<515x67xi32>
  %1 = memref.get_global @filter : memref<4x4xi32>
  %2 = memref.get_global @out : memref<512x64xi32>
  %rhs_memref = memref.get_global @rhs : memref<64x64xi32>
  %4 = bufferization.to_tensor %0 : memref<515x67xi32>
  %5 = bufferization.to_tensor %1 : memref<4x4xi32>
  %x = tensor.empty() : tensor<512x64xi32> 
  %conv_out = bufferization.to_tensor %2 : memref<512x64xi32>
  %rhs = bufferization.to_tensor %rhs_memref : memref<64x64xi32>
  %y = tensor.empty() : tensor<512x64xi32> 
  %out = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%4, %5 : tensor<515x67xi32>, tensor<4x4xi32>) outs(%x : tensor<512x64xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %6 = arith.muli %in, %in_0 : i32
    %7 = arith.addi %out, %6 : i32
    linalg.yield %7 : i32
  } -> tensor<512x64xi32>
  %matmul = linalg.matmul ins(%out, %rhs: tensor<512x64xi32>, tensor<64x64xi32>)
                          outs(%y: tensor<512x64xi32>) -> tensor<512x64xi32>
  
  %materialize2 = bufferization.to_memref %matmul : memref<512x64xi32>
  memref.copy %materialize2, %2 : memref<512x64xi32> to memref<512x64xi32>
  return %c0_i32 : i32
}

// transform.sequence failures(propagate) {
//   ^bb0(%arg0 : !transform.any_op):
//     %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
//     //Note that these represent the outer dimension first for tiling
//     %1,%2,%3 = transform.structured.tile_using_for %0 [32,32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
//     transform.yield
// }

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op) :
        // Since the %arg2 handle is associated with both elementwise operations,
        // we need to split it into two handles so we can target only the second
        // elementwise operation.
        %generic = transform.structured.match ops{["linalg.matmul","linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        %conv, %mul = transform.split_handle %generic
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op)

        // The actual tiling transformation takes tile sizes as attributes. It
        // produces a handle to the loop generated during tiling.
        %tiled_mul, %loop =
            transform.structured.tile_using_forall %mul tile_sizes [8, 32]
              : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // We can now fuse the other operations into the loop. Here, we fuse
        // operations one by one. This requires the operation that is being fused to
        // define the value used within the loop, so the order of such fusions is
        // important. We could also use "transform.merge_handles" to obtain a single
        // handle to all operations and give it to `fuse_into_containing_op` that
        // would take care of the ordering in this case.
         %conv_fused, %loop_0 =
             transform.structured.fuse_into_containing_op %conv into %loop
               : (!transform.any_op, !transform.any_op)
                 -> (!transform.any_op, !transform.any_op)
  

        transform.yield
}

// -----