#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 3)>
#map2 = affine_map<(d0)[s0] -> (s0)>
#map3 = affine_map<(d0) -> (0)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map8 = affine_map<(d0, d1) -> (d0 + d1 * 2)>
#map9 = affine_map<(d0, d1, d2) -> (d2)>
#map10 = affine_map<(d0, d1, d2) -> (d1)>
#map11 = affine_map<(d0, d1, d2) -> (d0)>
#map12 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map13 = affine_map<(d0, d1, d2) -> (d1 * 4 + d2 + 3)>
#map14 = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + 2)>
#map15 = affine_map<(d0, d1, d2) -> (d0 + d2 * 2)>
#map16 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map17 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map18 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d1 + d3, d0 + d2)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map23 = affine_map<(d0, d1)[s0, s1] -> (d1 + s0, d0 + s1)>
#map24 = affine_map<(d0, d1) -> (d1, d0)>
#map25 = affine_map<(d0, d1)[s0, s1] -> (s0, s1)>
#map26 = affine_map<(d0)[s0, s1, s2] -> (s0 + s1, d0 + s2)>
#map27 = affine_map<(d0)[s0] -> (s0, d0)>
#map28 = affine_map<(d0)[s0, s1] -> (s0, s1)>
#map29 = affine_map<(d0, d1, d2, d3) -> (d0 + d1 * 3)>
module {
  module @constant_access {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 4.000000e+00 : f32
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.mulf %in, %cst : f32
        linalg.yield %5 : f32
      }
      return
    }
  }
//  module @constant_mem_access {
//    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>) {
//      %c13 = arith.constant 13 : index
//      %c4 = arith.constant 4 : index
//      %0 = arith.index_cast %arg1 : i32 to index
//      %1 = arith.muli %0, %c4 : index
//      %2 = arith.divui %1, %c4 : index
//      %alloca = memref.alloca(%2) : memref<?xf32>
//      %3 = "polygeist.submap"(%arg2, %c13) <{map = #map1}> : (memref<?xf32>, index) -> memref<?xf32>
//      %4 = "polygeist.submap"(%arg2, %c4, %c13) <{map = #map2}> : (memref<?xf32>, index, index) -> memref<?xf32>
//      %5 = "polygeist.submap"(%alloca, %c13) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
//      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4 : memref<?xf32>, memref<?xf32>) outs(%5 : memref<?xf32>) {
//      ^bb0(%in: f32, %in_0: f32, %out: f32):
//        %6 = arith.mulf %in, %in_0 : f32
//        linalg.yield %6 : f32
//      }
//      return
//    }
//  }
  module @no_if {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      return
    }
  }
  module @arith_mul {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.mulf %in, %in : f32
        linalg.yield %5 : f32
      }
      return
    }
  }
  module @arith_add {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%arg3, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %5 = "polygeist.submap"(%alloca, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4 : memref<?xf32>, memref<?xf32>) outs(%5 : memref<?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.addf %in, %in_0 : f32
        %7 = arith.mulf %6, %6 : f32
        linalg.yield %7 : f32
      }
      return
    }
  }
  module @cond_arith {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = scf.if %arg0 -> (f32) {
          %6 = arith.mulf %in, %in : f32
          scf.yield %6 : f32
        } else {
          scf.yield %in : f32
        }
        linalg.yield %5 : f32
      }
      return
    }
  }
  module @reduction {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca, %c17) <{map = #map3}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["reduction"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.addf %out, %in : f32
        linalg.yield %5 : f32
      }
      return
    }
  }
  module @reduction_transformed {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %alloca_0 = memref.alloca() : memref<1xf32>
      affine.store %cst, %alloca_0[0] : memref<1xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca_0, %c17) <{map = #map3}> : (memref<1xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["reduction"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %out, %in : f32
        linalg.yield %6 : f32
      }
      %5 = affine.load %alloca_0[0] : memref<1xf32>
      affine.store %5, %alloca[0] : memref<?xf32>
      return
    }
  }
  module @reduction_transformed_simplified {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      affine.store %cst, %alloca[0] : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17) <{map = #map}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca, %c17) <{map = #map3}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["reduction"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.addf %out, %in : f32
        linalg.yield %5 : f32
      }
      return
    }
  }
  module @cond_store_1 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>) {
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      affine.for %arg3 = 0 to 17 {
        %3 = affine.load %arg2[%arg3] : memref<?xf32>
        %4 = arith.mulf %3, %3 : f32
        scf.if %arg0 {
          affine.store %4, %alloca[%arg3] : memref<?xf32>
        }
      }
      return
    }
  }
  module @cond_store_2 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>) {
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      affine.for %arg3 = 0 to 17 {
        %3 = affine.load %arg2[%arg3] : memref<?xf32>
        scf.if %arg0 {
          %4 = arith.mulf %3, %3 : f32
          affine.store %4, %alloca[%arg3] : memref<?xf32>
        } else {
          affine.store %3, %alloca[%arg3] : memref<?xf32>
        }
      }
      return
    }
  }
  module @for_within_for {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17, %c21) <{map = #map4}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %5 = "polygeist.submap"(%alloca, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["reduction", "parallel"]} ins(%3, %4 : memref<?x?xf32>, memref<?x?xf32>) outs(%5 : memref<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      }
      return
    }
  }
  module @for_within_for_2 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17, %c21) <{map = #map7}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %5 = "polygeist.submap"(%alloca, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["reduction", "parallel"]} ins(%3, %4 : memref<?x?xf32>, memref<?x?xf32>) outs(%5 : memref<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      }
      return
    }
  }
  module @for_within_for_3 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17, %c21) <{map = #map7}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %4 = "polygeist.submap"(%arg2, %c17, %c21) <{map = #map4}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %5 = "polygeist.submap"(%arg3, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %6 = "polygeist.submap"(%alloca, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["reduction", "parallel"]} ins(%3, %4, %5 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) outs(%6 : memref<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
        %7 = arith.mulf %in, %in_0 : f32
        %8 = arith.mulf %7, %in_1 : f32
        linalg.yield %8 : f32
      }
      return
    }
  }
  module @for_within_for_4 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17, %c21) <{map = #map8}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %5 = "polygeist.submap"(%alloca, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["reduction", "parallel"]} ins(%3, %4 : memref<?x?xf32>, memref<?x?xf32>) outs(%5 : memref<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      }
      return
    }
  }
  module @for_no_loop_dependency {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
      %c15 = arith.constant 15 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c15) <{map = #map3}> : (memref<?xf32>, index) -> memref<?xf32>
      %4 = "polygeist.submap"(%alloca, %c15) <{map = #map3}> : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["reduction"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      return
    }
  }
  module @for_2_levels_no_loop_dependency {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
      %c17 = arith.constant 17 : index
      %c15 = arith.constant 15 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c15, %c17) <{map = #map4}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %4 = "polygeist.submap"(%alloca, %c15, %c17) <{map = #map4}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "reduction"]} ins(%3 : memref<?x?xf32>) outs(%4 : memref<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      return
    }
  }
  module @for_3_levels_0 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
      %c15 = arith.constant 15 : index
      %c17 = arith.constant 17 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c21, %c17, %c15) <{map = #map9}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c21, %c17, %c15) <{map = #map10}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %5 = "polygeist.submap"(%alloca, %c21, %c17, %c15) <{map = #map11}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      linalg.generic {indexing_maps = [#map12, #map12, #map12], iterator_types = ["reduction", "reduction", "parallel"]} ins(%3, %4 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%5 : memref<?x?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      }
      return
    }
  }
  module @for_3_levels_1 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17, %c21, %c21) <{map = #map10}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c17, %c21, %c21) <{map = #map11}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %5 = "polygeist.submap"(%alloca, %c17, %c21, %c21) <{map = #map11}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      linalg.generic {indexing_maps = [#map12, #map12, #map12], iterator_types = ["reduction", "reduction", "parallel"]} ins(%3, %4 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%5 : memref<?x?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      }
      return
    }
  }
  module @for_3_levels_2 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c21, %c17, %c21) <{map = #map9}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c21, %c17, %c21) <{map = #map10}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %5 = "polygeist.submap"(%arg4, %c21, %c17, %c21) <{map = #map11}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %6 = "polygeist.submap"(%alloca, %c21, %c17, %c21) <{map = #map10}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      linalg.generic {indexing_maps = [#map12, #map12, #map12, #map12], iterator_types = ["reduction", "parallel", "reduction"]} ins(%3, %4, %5 : memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%6 : memref<?x?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
        %7 = arith.mulf %in, %in_0 : f32
        %8 = arith.mulf %7, %in_1 : f32
        linalg.yield %8 : f32
      }
      return
    }
  }
  module @for_3_levels_3 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c21, %c17, %c21) <{map = #map9}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c21, %c17, %c21) <{map = #map10}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %5 = "polygeist.submap"(%arg3, %c21, %c17, %c21) <{map = #map11}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %6 = "polygeist.submap"(%alloca, %c21, %c17, %c21) <{map = #map10}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      linalg.generic {indexing_maps = [#map12, #map12, #map12, #map12], iterator_types = ["reduction", "parallel", "reduction"]} ins(%3, %4, %5 : memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%6 : memref<?x?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
        %7 = arith.mulf %in, %in_0 : f32
        %8 = arith.mulf %7, %in_1 : f32
        linalg.yield %8 : f32
      }
      return
    }
  }
  module @for_3_levels_4 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c21, %c17, %c21) <{map = #map13}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c21, %c17, %c21) <{map = #map14}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %5 = "polygeist.submap"(%arg3, %c21, %c17, %c21) <{map = #map15}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      %6 = "polygeist.submap"(%alloca, %c21, %c17, %c21) <{map = #map10}> : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
      linalg.generic {indexing_maps = [#map12, #map12, #map12, #map12], iterator_types = ["reduction", "parallel", "reduction"]} ins(%3, %4, %5 : memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%6 : memref<?x?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
        %7 = arith.mulf %in, %in_0 : f32
        %8 = arith.mulf %7, %in_1 : f32
        linalg.yield %8 : f32
      }
      return
    }
  }
  module @for_within_for2 {
    func.func @main(%arg0: i1, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
      %c21 = arith.constant 21 : index
      %c17 = arith.constant 17 : index
      %c4 = arith.constant 4 : index
      %0 = arith.index_cast %arg1 : i32 to index
      %1 = arith.muli %0, %c4 : index
      %2 = arith.divui %1, %c4 : index
      %alloca = memref.alloca(%2) : memref<?xf32>
      %3 = "polygeist.submap"(%arg2, %c17, %c21) <{map = #map4}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %4 = "polygeist.submap"(%arg3, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      %5 = "polygeist.submap"(%alloca, %c17, %c21) <{map = #map5}> : (memref<?xf32>, index, index) -> memref<?x?xf32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["reduction", "parallel"]} ins(%3, %4 : memref<?x?xf32>, memref<?x?xf32>) outs(%5 : memref<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        linalg.yield %6 : f32
      }
      return
    }
  }
  module @matmul_1 {
    memref.global @out : memref<32x8xi32> = uninitialized
    memref.global @im2 : memref<8x8xi32> = uninitialized
    memref.global @im1 : memref<32x8xi32> = uninitialized
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @im1 : memref<32x8xi32>
      %1 = memref.get_global @im2 : memref<8x8xi32>
      %2 = memref.get_global @out : memref<32x8xi32>
      %3 = "polygeist.submap"(%0, %c8, %c8, %c32) <{map = #map16}> : (memref<32x8xi32>, index, index, index) -> memref<?x?x?xi32>
      %4 = "polygeist.submap"(%1, %c8, %c8, %c32) <{map = #map17}> : (memref<8x8xi32>, index, index, index) -> memref<?x?x?xi32>
      %5 = "polygeist.submap"(%2, %c8, %c8, %c32) <{map = #map18}> : (memref<32x8xi32>, index, index, index) -> memref<?x?x?xi32>
      linalg.generic {indexing_maps = [#map12, #map12, #map12], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : memref<?x?x?xi32>, memref<?x?x?xi32>) outs(%5 : memref<?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %6 = arith.muli %in, %in_0 : i32
        %7 = arith.addi %out, %6 : i32
        linalg.yield %7 : i32
      }
      return %c0_i32 : i32
    }
  }
  module @matmul_2 {
    memref.global @out : memref<128x32xi32> = uninitialized
    memref.global @im2 : memref<64x32xi32> = uninitialized
    memref.global @im1 : memref<128x64xi32> = uninitialized
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @im1 : memref<128x64xi32>
      %1 = memref.get_global @im2 : memref<64x32xi32>
      %2 = memref.get_global @out : memref<128x32xi32>
      %3 = "polygeist.submap"(%0, %c64, %c32, %c128) <{map = #map16}> : (memref<128x64xi32>, index, index, index) -> memref<?x?x?xi32>
      %4 = "polygeist.submap"(%1, %c64, %c32, %c128) <{map = #map17}> : (memref<64x32xi32>, index, index, index) -> memref<?x?x?xi32>
      %5 = "polygeist.submap"(%2, %c64, %c32, %c128) <{map = #map18}> : (memref<128x32xi32>, index, index, index) -> memref<?x?x?xi32>
      linalg.generic {indexing_maps = [#map12, #map12, #map12], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : memref<?x?x?xi32>, memref<?x?x?xi32>) outs(%5 : memref<?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %6 = arith.muli %in, %in_0 : i32
        %7 = arith.addi %out, %6 : i32
        linalg.yield %7 : i32
      }
      return %c0_i32 : i32
    }
  }
  module @conv_1 {
    memref.global @out : memref<512x64xi32> = uninitialized
    memref.global @filter : memref<4x4xi32> = uninitialized
    memref.global @im : memref<515x67xi32> = uninitialized
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @im : memref<515x67xi32>
      %1 = memref.get_global @filter : memref<4x4xi32>
      %2 = memref.get_global @out : memref<512x64xi32>
      %3 = "polygeist.submap"(%0, %c4, %c4, %c64, %c512) <{map = #map19}> : (memref<515x67xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %4 = "polygeist.submap"(%1, %c4, %c4, %c64, %c512) <{map = #map20}> : (memref<4x4xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %5 = "polygeist.submap"(%2, %c4, %c4, %c64, %c512) <{map = #map21}> : (memref<512x64xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%3, %4 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%5 : memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %6 = arith.muli %in, %in_0 : i32
        %7 = arith.addi %out, %6 : i32
        linalg.yield %7 : i32
      }
      return %c0_i32 : i32
    }
  }
  // module @conv_1_reduction_test {
  //   memref.global @out : memref<512x64xi32> = uninitialized
  //   memref.global @filter : memref<4x4xi32> = uninitialized
  //   memref.global @im : memref<515x67xi32> = uninitialized
  //   func.func @main(%arg0: index, %arg1: index) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
  //     %c4 = arith.constant 4 : index
  //     %c0_i32 = arith.constant 0 : i32
  //     %0 = memref.get_global @im : memref<515x67xi32>
  //     %1 = memref.get_global @filter : memref<4x4xi32>
  //     %2 = memref.get_global @out : memref<512x64xi32>
  //     %3 = "polygeist.submap"(%0, %arg0, %arg1, %c4, %c4) <{map = #map23}> : (memref<515x67xi32>, index, index, index, index) -> memref<?x?xi32>
  //     %4 = "polygeist.submap"(%1, %c4, %c4) <{map = #map24}> : (memref<4x4xi32>, index, index) -> memref<?x?xi32>
  //     %5 = "polygeist.submap"(%2, %arg0, %arg1, %c4, %c4) <{map = #map25}> : (memref<512x64xi32>, index, index, index, index) -> memref<?x?xi32>
  //     linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["reduction", "reduction"]} ins(%3, %4 : memref<?x?xi32>, memref<?x?xi32>) outs(%5 : memref<?x?xi32>) {
  //     ^bb0(%in: i32, %in_0: i32, %out: i32):
  //       %6 = arith.muli %in, %in_0 : i32
  //       %7 = arith.addi %out, %6 : i32
  //       linalg.yield %7 : i32
  //     }
  //     return %c0_i32 : i32
  //   }
  // }
  module @conv_2 {
    memref.global @out : memref<512x64xi32> = uninitialized
    memref.global @filter : memref<4x4xi32> = uninitialized
    memref.global @im : memref<515x67xi32> = uninitialized
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @im : memref<515x67xi32>
      %1 = memref.get_global @filter : memref<4x4xi32>
      %2 = memref.get_global @out : memref<512x64xi32>
      %3 = "polygeist.submap"(%0, %c4, %c4, %c64, %c512) <{map = #map19}> : (memref<515x67xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %4 = "polygeist.submap"(%1, %c4, %c4, %c64, %c512) <{map = #map20}> : (memref<4x4xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %5 = "polygeist.submap"(%2, %c4, %c4, %c64, %c512) <{map = #map21}> : (memref<512x64xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%3, %4 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%5 : memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32):
        %6 = arith.muli %in, %in_0 : i32
        %7 = arith.addi %out, %6 : i32
        linalg.yield %7 : i32
      }
      return %c0_i32 : i32
    }
  }
  module @box_filter {
    memref.global @out : memref<512x64xi32> = uninitialized
    memref.global @filter : memref<4x4xi32> = uninitialized
    memref.global @im : memref<515x67xi32> = uninitialized
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @im : memref<515x67xi32>
      %1 = memref.get_global @out : memref<512x64xi32>
      %2 = "polygeist.submap"(%0, %c4, %c4, %c64, %c512) <{map = #map19}> : (memref<515x67xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %3 = "polygeist.submap"(%1, %c4, %c4, %c64, %c512) <{map = #map21}> : (memref<512x64xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%2 : memref<?x?x?x?xi32>) outs(%3 : memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %out: i32):
        %4 = arith.addi %out, %in : i32
        linalg.yield %4 : i32
      }
      return %c0_i32 : i32
    }
  }
//  module @conv_loop1_test {
//    memref.global @out : memref<512x64xi32> = uninitialized
//    memref.global @filter : memref<4x4xi32> = uninitialized
//    memref.global @im : memref<515x67xi32> = uninitialized
//    func.func @main(%arg0: index, %arg1: index, %arg2: index) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
//      %c4 = arith.constant 4 : index
//      %c0_i32 = arith.constant 0 : i32
//      %0 = memref.get_global @im : memref<515x67xi32>
//      %1 = memref.get_global @filter : memref<4x4xi32>
//      %2 = memref.get_global @out : memref<512x64xi32>
//      %3 = "polygeist.submap"(%0, %arg0, %arg2, %arg1, %c4) <{map = #map26}> : (memref<515x67xi32>, index, index, index, index) -> memref<?xi32>
//      %4 = "polygeist.submap"(%1, %arg2, %c4) <{map = #map27}> : (memref<4x4xi32>, index, index) -> memref<?xi32>
//      %5 = "polygeist.submap"(%2, %arg0, %arg1, %c4) <{map = #map28}> : (memref<512x64xi32>, index, index, index) -> memref<?xi32>
//      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["reduction"]} ins(%3, %4 : memref<?xi32>, memref<?xi32>) outs(%5 : memref<?xi32>) {
//      ^bb0(%in: i32, %in_0: i32, %out: i32):
//        %6 = arith.muli %in, %in_0 : i32
//        %7 = arith.addi %out, %6 : i32
//        linalg.yield %7 : i32
//      }
//      return %c0_i32 : i32
//    }
//  }
//  module @submap_test {
//    memref.global @out : memref<511x64xi32> = uninitialized
//    memref.global @filter : memref<5x4xi32> = uninitialized
//    memref.global @im : memref<515x67xi32> = uninitialized
//    func.func @main(%arg0: index, %arg1: index) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
//      %c5 = arith.constant 5 : index
//      %c4 = arith.constant 4 : index
//      %c0_i32 = arith.constant 0 : i32
//      %0 = memref.get_global @im : memref<515x67xi32>
//      %1 = memref.get_global @filter : memref<5x4xi32>
//      %2 = memref.get_global @out : memref<511x64xi32>
//      %3 = "polygeist.submap"(%0, %arg0, %arg1, %c4, %c5) <{map = #map23}> : (memref<515x67xi32>, index, index, index, index) -> memref<?x?xi32>
//      %4 = "polygeist.submap"(%1, %c4, %c5) <{map = #map24}> : (memref<5x4xi32>, index, index) -> memref<?x?xi32>
//      %5 = "polygeist.submap"(%2, %arg0, %arg1, %c4, %c5) <{map = #map25}> : (memref<511x64xi32>, index, index, index, index) -> memref<?x?xi32>
//      linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["reduction", "reduction"]} ins(%3, %4 : memref<?x?xi32>, memref<?x?xi32>) outs(%5 : memref<?x?xi32>) {
//      ^bb0(%in: i32, %in_0: i32, %out: i32):
//        %6 = arith.muli %in, %in_0 : i32
//        %7 = arith.addi %out, %6 : i32
//        linalg.yield %7 : i32
//      }
//      return %c0_i32 : i32
//    }
//  }
  module @harris_score_1 {
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
      %c516 = arith.constant 516 : index
      %c3 = arith.constant 3 : index
      %c512 = arith.constant 512 : index
      %c5 = arith.constant 5 : index
      %c4_i32 = arith.constant 4 : i32
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @img_gx : memref<516x516xi32>
      %1 = memref.get_global @img_gy : memref<516x516xi32>
      %2 = memref.get_global @img_in : memref<518x518xi32>
      %3 = memref.get_global @coeffs_x : memref<9xi32>
      %4 = memref.get_global @coeffs_y : memref<9xi32>
      %5 = "polygeist.submap"(%2, %c3, %c3, %c516, %c516) <{map = #map19}> : (memref<518x518xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %6 = "polygeist.submap"(%3, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %7 = "polygeist.submap"(%4, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %8 = "polygeist.submap"(%0, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %9 = "polygeist.submap"(%1, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5, %6, %7 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%8, %9 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i32, %out_2: i32):
        %23 = arith.muli %in, %in_0 : i32
        %24 = arith.addi %out, %23 : i32
        %25 = arith.muli %in, %in_1 : i32
        %26 = arith.addi %out_2, %25 : i32
        linalg.yield %24, %26 : i32, i32
      }
      %10 = memref.get_global @img_ixx : memref<512x512xi32>
      %11 = memref.get_global @img_iyy : memref<512x512xi32>
      %12 = memref.get_global @img_ixy : memref<512x512xi32>
      %13 = "polygeist.submap"(%0, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %14 = "polygeist.submap"(%1, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %15 = "polygeist.submap"(%10, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %16 = "polygeist.submap"(%11, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %17 = "polygeist.submap"(%12, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%13, %14 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%15, %16, %17 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32, %out_1: i32, %out_2: i32):
        %23 = arith.muli %in, %in : i32
        %24 = arith.addi %out, %23 : i32
        %25 = arith.muli %in_0, %in_0 : i32
        %26 = arith.addi %out_1, %25 : i32
        %27 = arith.muli %in, %in_0 : i32
        %28 = arith.addi %out_2, %27 : i32
        linalg.yield %24, %26, %28 : i32, i32, i32
      }
      %18 = memref.get_global @score : memref<512x512xi32>
      %19 = "polygeist.submap"(%10, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %20 = "polygeist.submap"(%11, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %21 = "polygeist.submap"(%12, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %22 = "polygeist.submap"(%18, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%19, %20, %21 : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) outs(%22 : memref<?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i32):
        %23 = arith.muli %in, %in_0 : i32
        %24 = arith.muli %in_1, %in_1 : i32
        %25 = arith.subi %23, %24 : i32
        %26 = arith.addi %in, %in_0 : i32
        %27 = arith.muli %26, %c4_i32 : i32
        %28 = arith.muli %27, %26 : i32
        %29 = arith.subi %25, %28 : i32
        linalg.yield %29 : i32
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
      %c516 = arith.constant 516 : index
      %c3 = arith.constant 3 : index
      %c512 = arith.constant 512 : index
      %c5 = arith.constant 5 : index
      %c4_i32 = arith.constant 4 : i32
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.get_global @img_gx : memref<516x516xi32>
      %1 = memref.get_global @img_gy : memref<516x516xi32>
      %2 = memref.get_global @img_in : memref<518x518xi32>
      %3 = memref.get_global @coeffs_x : memref<9xi32>
      %4 = memref.get_global @coeffs_y : memref<9xi32>
      %5 = "polygeist.submap"(%2, %c3, %c3, %c516, %c516) <{map = #map19}> : (memref<518x518xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %6 = "polygeist.submap"(%3, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %7 = "polygeist.submap"(%4, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %8 = "polygeist.submap"(%1, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %9 = "polygeist.submap"(%0, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5, %6, %7 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%8, %9 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i32, %out_2: i32):
        %23 = arith.muli %in, %in_0 : i32
        %24 = arith.addi %out_2, %23 : i32
        %25 = arith.muli %in, %in_1 : i32
        %26 = arith.addi %out, %25 : i32
        linalg.yield %26, %24 : i32, i32
      }
      %10 = memref.get_global @img_ixx : memref<512x512xi32>
      %11 = memref.get_global @img_iyy : memref<512x512xi32>
      %12 = memref.get_global @img_ixy : memref<512x512xi32>
      %13 = "polygeist.submap"(%0, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %14 = "polygeist.submap"(%1, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %15 = "polygeist.submap"(%12, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %16 = "polygeist.submap"(%11, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %17 = "polygeist.submap"(%10, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%13, %14 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%15, %16, %17 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %out: i32, %out_1: i32, %out_2: i32):
        %23 = arith.muli %in, %in : i32
        %24 = arith.addi %out_2, %23 : i32
        %25 = arith.muli %in_0, %in_0 : i32
        %26 = arith.addi %out_1, %25 : i32
        %27 = arith.muli %in, %in_0 : i32
        %28 = arith.addi %out, %27 : i32
        linalg.yield %28, %26, %24 : i32, i32, i32
      }
      %18 = memref.get_global @score : memref<512x512xi32>
      %19 = "polygeist.submap"(%10, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %20 = "polygeist.submap"(%11, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %21 = "polygeist.submap"(%12, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %22 = "polygeist.submap"(%18, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%19, %20, %21 : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) outs(%22 : memref<?x?xi32>) {
      ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i32):
        %23 = arith.muli %in, %in_0 : i32
        %24 = arith.muli %in_1, %in_1 : i32
        %25 = arith.subi %23, %24 : i32
        %26 = arith.addi %in, %in_0 : i32
        %27 = arith.muli %26, %c4_i32 : i32
        %28 = arith.muli %27, %26 : i32
        %29 = arith.subi %25, %28 : i32
        linalg.yield %29 : i32
      }
      return %c0_i32 : i32
    }
  }
  module @harris_score_local {
    memref.global @coeffs_y : memref<9xi32> = dense<[-3, -10, -3, 0, 0, 0, 3, 10, 3]>
    memref.global @coeffs_x : memref<9xi32> = dense<[-3, 0, 3, -10, 0, 10, -3, 0, 3]>
    memref.global @score : memref<512x512xi32> = uninitialized
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      %c516 = arith.constant 516 : index
      %c3 = arith.constant 3 : index
      %c512 = arith.constant 512 : index
      %c5 = arith.constant 5 : index
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
      %2 = "polygeist.submap"(%alloca_4, %c3, %c3, %c516, %c516) <{map = #map19}> : (memref<518x518xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %3 = "polygeist.submap"(%0, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %4 = "polygeist.submap"(%1, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %5 = "polygeist.submap"(%alloca_3, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %6 = "polygeist.submap"(%alloca_2, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%2, %3, %4 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%5, %6 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32):
        %17 = arith.muli %in, %in_5 : i32
        %18 = arith.addi %out, %17 : i32
        %19 = arith.muli %in, %in_6 : i32
        %20 = arith.addi %out_7, %19 : i32
        linalg.yield %18, %20 : i32, i32
      }
      %7 = "polygeist.submap"(%alloca_3, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %8 = "polygeist.submap"(%alloca_2, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %9 = "polygeist.submap"(%alloca, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %10 = "polygeist.submap"(%alloca_0, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      %11 = "polygeist.submap"(%alloca_1, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%7, %8 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%9, %10, %11 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
      ^bb0(%in: i32, %in_5: i32, %out: i32, %out_6: i32, %out_7: i32):
        %17 = arith.muli %in, %in : i32
        %18 = arith.addi %out_7, %17 : i32
        %19 = arith.muli %in_5, %in_5 : i32
        %20 = arith.addi %out_6, %19 : i32
        %21 = arith.muli %in, %in_5 : i32
        %22 = arith.addi %out, %21 : i32
        linalg.yield %22, %20, %18 : i32, i32, i32
      }
      %12 = memref.get_global @score : memref<512x512xi32>
      %13 = "polygeist.submap"(%alloca_1, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %14 = "polygeist.submap"(%alloca_0, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %15 = "polygeist.submap"(%alloca, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      %16 = "polygeist.submap"(%12, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%13, %14, %15 : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) outs(%16 : memref<?x?xi32>) {
      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32):
        %17 = arith.muli %in, %in_5 : i32
        %18 = arith.muli %in_6, %in_6 : i32
        %19 = arith.subi %17, %18 : i32
        %20 = arith.addi %in, %in_5 : i32
        %21 = arith.muli %20, %c4_i32 : i32
        %22 = arith.muli %21, %20 : i32
        %23 = arith.subi %19, %22 : i32
        linalg.yield %23 : i32
      }
      return %c0_i32 : i32
    }
  }
}

module @harris_score_2d_kernel {
  memref.global "private" @_ZL8coeffs_y : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global "private" @_ZL8coeffs_x : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global @score : memref<512x512xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c516 = arith.constant 516 : index
    %c3 = arith.constant 3 : index
    %c512 = arith.constant 512 : index
    %c5 = arith.constant 5 : index
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
    %2 = "polygeist.submap"(%alloca_4, %c3, %c3, %c516, %c516) <{map = #map19}> : (memref<518x518xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %3 = "polygeist.submap"(%0, %c3, %c3, %c516, %c516) <{map = #map20}> : (memref<3x3xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %4 = "polygeist.submap"(%1, %c3, %c3, %c516, %c516) <{map = #map20}> : (memref<3x3xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %5 = "polygeist.submap"(%alloca_2, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %6 = "polygeist.submap"(%alloca_3, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%2, %3, %4 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%5, %6 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
    ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32):
      %17 = arith.muli %in, %in_5 : i32
      %18 = arith.addi %out_7, %17 : i32
      %19 = arith.muli %in, %in_6 : i32
      %20 = arith.addi %out, %19 : i32
      linalg.yield %20, %18 : i32, i32
    }
    %7 = "polygeist.submap"(%alloca_3, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %8 = "polygeist.submap"(%alloca_2, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %9 = "polygeist.submap"(%alloca, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %10 = "polygeist.submap"(%alloca_0, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %11 = "polygeist.submap"(%alloca_1, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%7, %8 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%9, %10, %11 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
    ^bb0(%in: i32, %in_5: i32, %out: i32, %out_6: i32, %out_7: i32):
      %17 = arith.muli %in, %in : i32
      %18 = arith.addi %out_7, %17 : i32
      %19 = arith.muli %in_5, %in_5 : i32
      %20 = arith.addi %out_6, %19 : i32
      %21 = arith.muli %in, %in_5 : i32
      %22 = arith.addi %out, %21 : i32
      linalg.yield %22, %20, %18 : i32, i32, i32
    }
    %12 = memref.get_global @score : memref<512x512xi32>
    %13 = "polygeist.submap"(%alloca_1, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    %14 = "polygeist.submap"(%alloca_0, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    %15 = "polygeist.submap"(%alloca, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    %16 = "polygeist.submap"(%12, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%13, %14, %15 : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) outs(%16 : memref<?x?xi32>) {
    ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32):
      %17 = arith.muli %in, %in_5 : i32
      %18 = arith.muli %in_6, %in_6 : i32
      %19 = arith.subi %17, %18 : i32
      %20 = arith.addi %in, %in_5 : i32
      %21 = arith.muli %20, %c4_i32 : i32
      %22 = arith.muli %21, %20 : i32
      %23 = arith.subi %19, %22 : i32
      linalg.yield %23 : i32
    }
    return %c0_i32 : i32
  }
}

module @harris_score_gradient_1d_kernel {
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
    %c516 = arith.constant 516 : index
    %c3 = arith.constant 3 : index
    %c512 = arith.constant 512 : index
    %c5 = arith.constant 5 : index
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @img_gx : memref<516x516xi32>
    %1 = memref.get_global @img_gy : memref<516x516xi32>
    %2 = memref.get_global @img_in : memref<518x518xi32>
    %3 = memref.get_global @coeffs_x : memref<9xi32>
    %4 = memref.get_global @coeffs_y : memref<9xi32>
    %5 = "polygeist.submap"(%2, %c3, %c3, %c516, %c516) <{map = #map19}> : (memref<518x518xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %6 = "polygeist.submap"(%3, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %7 = "polygeist.submap"(%4, %c3, %c3, %c516, %c516) <{map = #map29}> : (memref<9xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %8 = "polygeist.submap"(%0, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %9 = "polygeist.submap"(%1, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5, %6, %7 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%8, %9 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i32, %out_2: i32):
      %23 = arith.muli %in, %in_0 : i32
      %24 = arith.addi %out, %23 : i32
      %25 = arith.muli %in, %in_1 : i32
      %26 = arith.addi %out_2, %25 : i32
      linalg.yield %24, %26 : i32, i32
    }
    return %c0_i32 : i32
  }
}

module @harris_score_gradient_2d_kernel {
  memref.global "private" @_ZL8coeffs_y : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global "private" @_ZL8coeffs_x : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
  memref.global @score : memref<512x512xi32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c516 = arith.constant 516 : index
    %c3 = arith.constant 3 : index
    %c512 = arith.constant 512 : index
    %c5 = arith.constant 5 : index
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
    %2 = "polygeist.submap"(%alloca_4, %c3, %c3, %c516, %c516) <{map = #map19}> : (memref<518x518xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %3 = "polygeist.submap"(%0, %c3, %c3, %c516, %c516) <{map = #map20}> : (memref<3x3xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %4 = "polygeist.submap"(%1, %c3, %c3, %c516, %c516) <{map = #map20}> : (memref<3x3xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %5 = "polygeist.submap"(%alloca_2, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %6 = "polygeist.submap"(%alloca_3, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%2, %3, %4 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%5, %6 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
    ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32):
      %17 = arith.muli %in, %in_5 : i32
      %18 = arith.addi %out_7, %17 : i32
      %19 = arith.muli %in, %in_6 : i32
      %20 = arith.addi %out, %19 : i32
      linalg.yield %20, %18 : i32, i32
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
    %c516 = arith.constant 516 : index
    %c3 = arith.constant 3 : index
    %c512 = arith.constant 512 : index
    %c5 = arith.constant 5 : index
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
    %2 = "polygeist.submap"(%alloca_4, %c3, %c3, %c516, %c516) <{map = #map19}> : (memref<518x518xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %3 = "polygeist.submap"(%0, %c3, %c3, %c516, %c516) <{map = #map20}> : (memref<3x3xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %4 = "polygeist.submap"(%1, %c3, %c3, %c516, %c516) <{map = #map20}> : (memref<3x3xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %5 = "polygeist.submap"(%alloca_2, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %6 = "polygeist.submap"(%alloca_3, %c3, %c3, %c516, %c516) <{map = #map21}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%2, %3, %4 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%5, %6 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
    ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32):
      %19 = arith.muli %in, %in_5 : i32
      %20 = arith.addi %out_7, %19 : i32
      %21 = arith.muli %in, %in_6 : i32
      %22 = arith.addi %out, %21 : i32
      linalg.yield %22, %20 : i32, i32
    }
    %7 = memref.get_global @_ZL8coeffs_1 : memref<5x5xi32>
    %8 = "polygeist.submap"(%alloca_3, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %9 = "polygeist.submap"(%alloca_2, %c5, %c5, %c512, %c512) <{map = #map19}> : (memref<516x516xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %10 = "polygeist.submap"(%7, %c5, %c5, %c512, %c512) <{map = #map20}> : (memref<5x5xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %11 = "polygeist.submap"(%alloca, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %12 = "polygeist.submap"(%alloca_0, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    %13 = "polygeist.submap"(%alloca_1, %c5, %c5, %c512, %c512) <{map = #map21}> : (memref<512x512xi32>, index, index, index, index) -> memref<?x?x?x?xi32>
    linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%8, %9, %10 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) outs(%11, %12, %13 : memref<?x?x?x?xi32>, memref<?x?x?x?xi32>, memref<?x?x?x?xi32>) {
    ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32, %out_8: i32):
      %19 = arith.muli %in, %in : i32
      %20 = arith.muli %19, %in_6 : i32
      %21 = arith.addi %out_8, %20 : i32
      %22 = arith.muli %in_5, %in_5 : i32
      %23 = arith.muli %22, %in_6 : i32
      %24 = arith.addi %out_7, %23 : i32
      %25 = arith.muli %in, %in_5 : i32
      %26 = arith.muli %25, %in_6 : i32
      %27 = arith.addi %out, %26 : i32
      linalg.yield %27, %24, %21 : i32, i32, i32
    }
    %14 = memref.get_global @score : memref<512x512xi32>
    %15 = "polygeist.submap"(%alloca_1, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    %16 = "polygeist.submap"(%alloca_0, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    %17 = "polygeist.submap"(%alloca, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    %18 = "polygeist.submap"(%14, %c512, %c512) <{map = #map24}> : (memref<512x512xi32>, index, index) -> memref<?x?xi32>
    linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%15, %16, %17 : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>) outs(%18 : memref<?x?xi32>) {
    ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32):
      %19 = arith.muli %in, %in_5 : i32
      %20 = arith.muli %in_6, %in_6 : i32
      %21 = arith.subi %19, %20 : i32
      %22 = arith.addi %in, %in_5 : i32
      %23 = arith.muli %22, %c4_i32 : i32
      %24 = arith.muli %23, %22 : i32
      %25 = arith.subi %21, %24 : i32
      linalg.yield %25 : i32
    }
    return %c0_i32 : i32
  }
}
