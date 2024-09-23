// RUN: cgeist %s --function=* -S | FileCheck %s

struct MOperandInfo {
  char device;
  char dtype;
};

template <typename T>
class MSmallVector {
 public:
  struct MOperandInfo *BeginX;

  const struct MOperandInfo& operator[](int idx) const {
    return BeginX[idx];
  }
};


struct MTensorIterator {
  char input_dtype() const { return operands_[0].dtype; }
  char device() const { return operands_[0].device; }
  MSmallVector<MOperandInfo> operands_;
};

template <typename func_t>
void igpu_kernel(MTensorIterator& iter, const func_t& f) {
  iter.device();
}

extern "C" {
void lt_kernel_cuda(MTensorIterator& iter) {
  if (iter.input_dtype()) {
    ([&]() { igpu_kernel(iter, []() -> bool { return false; }); })();
  }
}
}

// CHECK-LABEL:   func.func @lt_kernel_cuda(
// CHECK-SAME:                              %[[VAL_0:.*]]: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZNK15MTensorIterator11input_dtypeEv(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_1:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> memref<?x2xi8>
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_2]][0, 1] : memref<?x2xi8>
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         }

// CHECK-LABEL:   func.func @_ZNK12MSmallVectorI12MOperandInfoEixEi(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: memref<?x1xmemref<?x2xi8>>,
// CHECK-SAME:                                                      %[[VAL_1:.*]]: i32) -> memref<?x2xi8> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0, 0] : memref<?x1xmemref<?x2xi8>>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = "polygeist.subindex"(%[[VAL_2]], %[[VAL_3]]) : (memref<?x2xi8>, index) -> memref<?x2xi8>
// CHECK:           return %[[VAL_4]] : memref<?x2xi8>
// CHECK:         }

// CHECK-LABEL:   func.func @_ZNK15MTensorIterator6deviceEv(
// CHECK-SAME:                                              %[[VAL_0:.*]]: memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[VAL_1:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(!llvm.struct<(memref<?x2xi8>)>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> memref<?x2xi8>
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_2]][0, 0] : memref<?x2xi8>
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         }