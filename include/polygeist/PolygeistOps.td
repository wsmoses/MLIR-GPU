//===- Polygeist.td - Polygeist dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_OPS
#define POLYGEIST_OPS

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"

def UndefOp
    : Polygeist_Op<"undef", [Pure]> {
  let summary = "More flexible undef op";
  let skipDefaultBuilders = 1;
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Type":$type), [{
      $_state.types.push_back(type);
    }]>];
  let hasCanonicalizer = true;
}

def NoopOp
    : Polygeist_Op<"noop",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Noop for preventing folding or transformations";
  let arguments = (ins Variadic<Index>:$blockDims);
  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins "ValueRange":$indices)>];
  let description = [{}];
}

def GetDeviceGlobalOp
    : Polygeist_Op<"get_device_global",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "";
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyStaticShapeMemRef:$result);
  let description = [{}];
}

def CacheLoad
    : Polygeist_Op<"cacheload"> {

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = memref.getType().cast<MemRefType>();
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];
  let summary = "load from cross barier cache";
  let description = [{}];
}

// HasParent<"ParallelOp">,
def BarrierOp
    : Polygeist_Op<"barrier",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let arguments = (ins Variadic<Index>:$indices);
  let summary = "barrier for parallel loops";
  let description = [{}];
  let hasCanonicalizer = true;
}

//===----------------------------------------------------------------------===//
// SubIndexOp
//===----------------------------------------------------------------------===//

def SubIndexOp : Polygeist_Op<"subindex", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "memref subview operation";

  let arguments = (ins AnyMemRef : $source, Index : $index);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def GPUBlockOp : Polygeist_Op<"gpu_block", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"polygeist::PolygeistYieldOp">]>,
  Arguments<(ins Index:$blockIndexX, Index:$blockIndexY, Index:$blockIndexZ)> {
  let summary = "Wraps a GPU kernel block to prevent restructuring";
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
      "Value":$blockIndexX, "Value":$blockIndexY, "Value":$blockIndexZ)>];
}

def GPUThreadOp : Polygeist_Op<"gpu_thread", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"polygeist::PolygeistYieldOp">]>,
  Arguments<(ins Index:$threadIndexX, Index:$threadIndexY, Index:$threadIndexZ)> {
  let summary = "Wraps a GPU kernel thread to prevent restructuring";
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
      "Value":$threadIndexX, "Value":$threadIndexY, "Value":$threadIndexZ)>];
}

def AlternativesOp : Polygeist_Op<"alternatives", [
  RecursiveMemoryEffects]> {
  let summary = "Provides several alternatives kernels for gpu code";
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "int":$regionNum)>];
  let hasCanonicalizer = 1;
}

def GPUWrapperOp : Polygeist_Op<"gpu_wrapper", [
  RecursiveMemoryEffects,
  AutomaticAllocationScope,
  SingleBlockImplicitTerminator<"polygeist::PolygeistYieldOp">]> {
  let arguments = (ins Variadic<Index>:$blockDims);
  let summary = "Indicates the region contained must be executed on the GPU";
  let description = [{
    The optional arguments to this operation are suggestions about what block
    dimensions this gpu kernel should have - usually taken from kernel launch
    params
  }];
  let results = (outs Index : $result);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins "ValueRange":$blockSizes)>,
      OpBuilder<(ins)>];
}

def GPUErrorOp : Polygeist_Op<"gpu_error", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"polygeist::PolygeistYieldOp">]>,
  Arguments<(ins)> {
  let summary = "Gets the error returned by the gpu operation inside";
  // TODO should be i32, not index
  let results = (outs Index : $result);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];

}

def PolygeistYieldOp : Polygeist_Op<"polygeist_yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["AlternativesOp", "GPUWrapperOp", "GPUErrorOp", "GPUBlockOp", "GPUThreadOp"]>]> {
  let summary = "Polygeist ops terminator";
}

def StreamToTokenOp : Polygeist_Op<"stream2token", [
  Pure
]> {
  let summary = "Extract an async stream from a cuda stream";

  let arguments = (ins AnyType : $source);
  let results = (outs AnyType : $result);
}

//===----------------------------------------------------------------------===//
// Memref2PointerOp
//===----------------------------------------------------------------------===//

def Memref2PointerOp : Polygeist_Op<"memref2pointer", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "Extract and LLVM pointer from a MemRef";

  let arguments = (ins AnyMemRef : $source);
  let results = (outs LLVM_AnyPointer : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def MemrefCastOp : Polygeist_Op<"memref_cast", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "Cast memrefs like c/c++ pointers";

  let arguments = (ins AnyMemRef : $source);
  let results = (outs AnyMemRef : $result);

  //let hasFolder = 1;
  //let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def Pointer2MemrefOp : Polygeist_Op<"pointer2memref", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "Upgrade a pointer to a memref";

  let arguments = (ins LLVM_AnyPointer : $source);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def GetFuncOp : Polygeist_Op<"get_func",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "get the pointer pointing to a function";
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs LLVM_AnyPointer : $result);
  // let assemblyFormat = "$name `:` type($result) attr-dict";
  let hasCanonicalizer = 1;
}

def TrivialUseOp : Polygeist_Op<"trivialuse"> {
  let summary = "memref subview operation";

  let arguments = (ins AnyMemRef : $source);

}

def TypeSizeOp : Polygeist_Op<"typeSize", [Pure]> {
  let summary = "Get size of type";
  let arguments = (ins TypeAttr : $source);
  let results = (outs AnyType : $result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def TypeAlignOp : Polygeist_Op<"typeAlign", [Pure]> {
  let summary = "Get alignment of type";
  let arguments = (ins TypeAttr : $source);
  let results = (outs AnyType : $result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SubmapOp : Polygeist_Op<"submap", [Pure, ViewLikeOpInterface]> {
  let arguments = (ins Arg<AnyMemRef, "the reference to load from">:$memref,
      Variadic<Index>:$indices_and_sizes,
      AffineMapAttr:$map
      );
  let results = (outs AnyMemRef : $result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::ValueRange getSymbols() { return getOperands().slice(1, getMap().getNumSymbols()); }
    ::mlir::ValueRange getSizes() { return getOperands().slice(getMap().getNumSymbols()+1, getType().getShape().size()); }
    ::mlir::Value getViewSource() { return getMemref(); }
  }];
}

#endif // POLYGEIST_OPS
