// RUN: cgeist %s --no-inline --function=* -S | FileCheck %s

struct AOperandInfo {
  void* data;

  bool is_output;

  bool is_read_write;
};


/// This is all the non-templated stuff common to all SmallVectors.

/// This is the part of SmallVectorTemplateBase which does not depend on whether
/// the type T is a POD. The extra dummy template argument is used by ArrayRef
/// to avoid unnecessarily requiring T to be complete.
template <typename T>
class ASmallVectorTemplateCommon {
 public:
  void *BeginX, *EndX;

  // forward iterator creation methods.
  const T* begin() const {
    return (const T*)this->BeginX;
  }
};

unsigned long long int div_kernel_cuda(ASmallVectorTemplateCommon<AOperandInfo> &operands) {
  return (const AOperandInfo*)operands.EndX - operands.begin();
}
// CHECK-LABEL:   func.func @_Z15div_kernel_cudaR26ASmallVectorTemplateCommonI12AOperandInfoE(
// CHECK-SAME:                                                                                %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x2xmemref<?xi8>>) -> i64
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = affine.load %[[VAL_0]][0, 1] : memref<?x2xmemref<?xi8>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?xi8>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = call @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(%[[VAL_0]]) : (memref<?x2xmemref<?xi8>>) -> memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_3]]) : (memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_5:[A-Za-z0-9_]*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr to i64
// CHECK-DAG:       %[[VAL_6:[A-Za-z0-9_]*]] = llvm.ptrtoint %[[VAL_4]] : !llvm.ptr to i64
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = "polygeist.typeSize"() <{source = !llvm.struct<(memref<?xi8>, i8, i8)>}> : () -> index
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_8]] : index to i64
// CHECK:           %[[VAL_10:[A-Za-z0-9_]*]] = arith.divsi %[[VAL_7]], %[[VAL_9]] : i64
// CHECK:           return %[[VAL_10]] : i64
// CHECK:         }

// CHECK-LABEL:   func.func @_ZNK26ASmallVectorTemplateCommonI12AOperandInfoE5beginEv(
// CHECK-SAME:                                                                        %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x2xmemref<?xi8>>) -> memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = affine.load %[[VAL_0]][0, 0] : memref<?x2xmemref<?xi8>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?xi8>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK:           return %[[VAL_3]] : memref<?x!llvm.struct<(memref<?xi8>, i8, i8)>>
// CHECK:         }

