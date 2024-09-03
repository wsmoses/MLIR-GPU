#ifndef POLYGEIST_PASSES
#define POLYGEIST_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def PolyhedralOpt : Pass<"polyhedral-opt"> {
  let summary = "Optimize affine regions with pluto";
  let constructor = "mlir::polygeist::createPolyhedralOptPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "affine::AffineDialect",
    // This vector dialect is needed because lower-affine needs it. We launch
    // a pass manager to use lower-affine from this pass itself, and that
    // causes the multi-threaded mlir context to try to load the vector dialect
    // which is unsupported. Instead we preload it here.
    "vector::VectorDialect",
  ];
}

def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::replaceAffineCFGPass()";
}

def PolygeistMem2Reg : Pass<"polygeist-mem2reg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::createPolygeistMem2RegPass()";
}

def SCFParallelLoopUnroll : Pass<"scf-parallel-loop-unroll"> {
  let summary = "Unroll and interleave scf parallel loops";
  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect",
  ];
  let constructor = "mlir::polygeist::createSCFParallelLoopUnrollPass()";
  let options = [
  Option<"unrollFactor", "unrollFactor", "int", /*default=*/"2", "Unroll factor">
  ];
}

def CollectKernelStatistics : Pass<"collect-kernel-statistics", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to cpu versions";
  let dependentDialects = [];
  let constructor = "mlir::polygeist::createCollectKernelStatisticsPass()";
}

def LowerAlternatives : Pass<"lower-alternatives", "mlir::ModuleOp"> {
  let summary = "Lower alternatives if in opt mode";
  let dependentDialects = [];
  let constructor = "mlir::polygeist::createLowerAlternativesPass()";
}

def ConvertCudaRTtoCPU : Pass<"convert-cudart-to-cpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to cpu versions";
  let dependentDialects = [
    "memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect",
    "cf::ControlFlowDialect",
  ];
  let constructor = "mlir::polygeist::createConvertCudaRTtoCPUPass()";
}

def FixGPUFunc : Pass<"fix-gpu-func", "mlir::gpu::GPUModuleOp"> {
  let summary = "Fix nested calls to gpu functions we generate in the frontend";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createFixGPUFuncPass()";
}

def ConvertCudaRTtoGPU : Pass<"convert-cudart-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to generic gpu versions";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createConvertCudaRTtoGPUPass()";
}

def ConvertCudaRTtoHipRT : Pass<"convert-cudart-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to generic gpu versions";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createConvertCudaRTtoGPUPass()";
}

def ParallelLower : Pass<"parallel-lower", "mlir::ModuleOp"> {
  let summary = "Lower gpu launch op to parallel ops";
  let dependentDialects = [
    "scf::SCFDialect",
    "polygeist::PolygeistDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
  ];
  let constructor = "mlir::polygeist::createParallelLowerPass()";
}

def AffineReduction : Pass<"detect-reduction"> {
  let summary = "Detect reductions in affine.for";
  let constructor = "mlir::polygeist::detectReductionPass()";
}

def SCFCPUify : Pass<"cpuify"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createCPUifyPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let options = [
  Option<"method", "method", "std::string", /*default=*/"\"distribute\"", "Method of doing distribution">
  ];
}

def ConvertParallelToGPU1 : Pass<"convert-parallel-to-gpu1"> {
  let summary = "Convert parallel loops to gpu";
  let constructor = "mlir::polygeist::createConvertParallelToGPUPass1()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
  let options = [
  Option<"arch", "arch", "std::string", /*default=*/"\"sm_60\"", "Target GPU architecture">
  ];
}

def ConvertParallelToGPU2 : Pass<"convert-parallel-to-gpu2"> {
  let summary = "Convert parallel loops to gpu";
  let constructor = "mlir::polygeist::createConvertParallelToGPUPass2()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
}

def ConvertToOpaquePtrPass : Pass<"convert-to-opaque-ptr"> {
  let summary = "Convert typed llvm pointers to opaque";
  let constructor = "mlir::polygeist::createConvertToOpaquePtrPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def MergeGPUModulesPass : Pass<"merge-gpu-modules", "mlir::ModuleOp"> {
  let summary = "Merge all gpu modules into one";
  let constructor = "mlir::polygeist::createMergeGPUModulesPass()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
}

def InnerSerialization : Pass<"inner-serialize"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createInnerSerializationPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
}

def Serialization : Pass<"serialize"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createSerializationPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
}

def SCFBarrierRemovalContinuation : InterfacePass<"barrier-removal-continuation", "FunctionOpInterface"> {
  let summary = "Remove scf.barrier using continuations";
  let constructor = "mlir::polygeist::createBarrierRemovalContinuation()";
  let dependentDialects = ["memref::MemRefDialect", "func::FuncDialect"];
}

def SCFRaiseToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::polygeist::createRaiseSCFToAffinePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "scf::SCFDialect",
  ];
}

def AffineRaiseToLinalg : Pass<"raise-affine-to-linalg"> {
  let summary = "Raise affine to linalg";
  let constructor = "mlir::polygeist::createRaiseAffineToLinalgPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
  ];
}

def SCFCanonicalizeFor : Pass<"canonicalize-scf-for"> {
  let summary = "Run some additional canonicalization for scf::for";
  let constructor = "mlir::polygeist::createCanonicalizeForPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "math::MathDialect",
  ];
}

def ForBreakToWhile : Pass<"for-break-to-while"> {
  let summary = "Rewrite scf.for(scf.if) to scf.while";
  let constructor = "mlir::polygeist::createForBreakToWhilePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
  ];
}

def ParallelLICM : Pass<"parallel-licm"> {
  let summary = "Perform LICM on known parallel (and serial) loops";
  let constructor = "mlir::polygeist::createParallelLICMPass()";
}

def OpenMPOptPass : Pass<"openmp-opt"> {
  let summary = "Optimize OpenMP";
  let constructor = "mlir::polygeist::createOpenMPOptPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "omp::OpenMPDialect",
    "LLVM::LLVMDialect",
  ];
}

def PolygeistCanonicalize : Pass<"canonicalize-polygeist"> {
  let constructor = "mlir::polygeist::createPolygeistCanonicalizePass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "scf::SCFDialect",
    "polygeist::PolygeistDialect",
  ];
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Perform control flow optimizations to the region tree">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Max. iterations between applying patterns / simplifying regions">,
    Option<"maxNumRewrites", "max-num-rewrites", "int64_t", /*default=*/"-1",
           "Max. number of pattern rewrites within an iteration">,
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic pattern">
  ] # RewritePassUtils.options;
}

def LoopRestructure : Pass<"loop-restructure"> {
  let constructor = "mlir::polygeist::createLoopRestructurePass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
    "polygeist::PolygeistDialect",
  ];
}

def RemoveTrivialUse : Pass<"trivialuse"> {
  let constructor = "mlir::polygeist::createRemoveTrivialUsePass()";
}

def ConvertPolygeistToLLVM : Pass<"convert-polygeist-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let description = [{
    Convert standard operations into the LLVM IR dialect operations.

    #### Input invariant

    -   operations including: arithmetic on integers and floats, constants,
        direct calls, returns and branches;
    -   no `tensor` types;
    -   all `vector` are one-dimensional;
    -   all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.

    #### Output IR

    Functions converted to LLVM IR. Function arguments types are converted
    one-to-one. Function results are converted one-to-one and, in case more than
    1 value is returned, packed into an LLVM IR struct type. Function calls and
    returns are updated accordingly. Block argument types are updated to use
    LLVM IR types.
  }];
  let constructor = "mlir::polygeist::createConvertPolygeistToLLVMPass()";
  let dependentDialects = [
    "polygeist::PolygeistDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "scf::SCFDialect",
  ];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">,
    Option<"useCStyleMemRef", "use-c-style-memref", "bool",
           /*default=*/"true",
           "Use C-style nested-array lowering of memref instead of "
           "the default MLIR descriptor structure">
  ];
}

#endif // POLYGEIST_PASSES
