//===- cgeist.cpp - cgeist Driver ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for cgeist when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "clang/../../lib/Driver/ToolChains/Cuda.h"
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include <fstream>

#ifdef POLYGEIST_ENABLE_POLYMER
#include "polymer/Transforms/ExtractScopStmt.h"
#include "polymer/Transforms/PlutoTransform.h"
#include "polymer/Transforms/Reg2Mem.h"
#endif

#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"

#include <fstream>

#include "ArgumentList.h"

using namespace llvm;

#define POLYGEIST_ENABLE_GPU (POLYGEIST_ENABLE_CUDA || POLYGEIST_ENABLE_ROCM)

static cl::OptionCategory toolOptions("clang to mlir - tool options");

static cl::opt<bool>
    ClPolyhedralOpt("polyhedral-opt", cl::init(false),
                    cl::desc("Use polymer to optimize affine regions"));

static cl::opt<bool> CudaLower("cuda-lower", cl::init(false),
                               cl::desc("Add parallel loops around cuda"));

static cl::opt<bool> EmitCUDA("emit-cuda", cl::init(false),
                              cl::desc("Emit CUDA code"));

static cl::opt<bool> EmitROCM("emit-rocm", cl::init(false),
                              cl::desc("Emit ROCM code"));

static cl::opt<bool>
    OutputIntermediateGPU("output-intermediate-gpu", cl::init(false),
                          cl::desc("Output intermediate gpu code"));

static cl::opt<bool>
    UseOriginalGPUBlockSize("use-original-gpu-block-size", cl::init(false),
                            cl::desc("Try not to alter the GPU kernel block "
                                     "sizes originally used in the code"));

static cl::opt<PolygeistGPUStructureMode> GPUKernelStructureMode(
    "gpu-kernel-structure-mode", cl::init(PGSM_Discard),
    cl::desc("How to hangle the original gpu kernel parallel structure"),
    llvm::cl::values(
        clEnumValN(PGSM_Discard, "discard", "Discard the structure"),
        clEnumValN(PGSM_BlockThreadWrappers, "block_thread_wrappers",
                   "Wrap blocks and thread in operations"),
        clEnumValN(PGSM_ThreadNoop, "thread_noop", "Put noop in the thread"),
        clEnumValN(PGSM_BlockThreadNoops, "block_thread_noops",
                   "Put noops in the block and thread")));

static cl::opt<bool> EmitGPUKernelLaunchBounds(
    "emit-gpu-kernel-launch-bounds", cl::init(true),
    cl::desc("Emit GPU kernel launch bounds where possible"));

static cl::opt<int> DeviceOptLevel("device-opt-level", cl::init(4),
                                   cl::desc("Optimization level for ptxas"));

static cl::opt<bool> EmitLLVM("emit-llvm", cl::init(false),
                              cl::desc("Emit llvm"));

static cl::opt<bool> EmitOpenMPIR("emit-openmpir", cl::init(false),
                                  cl::desc("Emit OpenMP IR"));

static cl::opt<bool> EmitLLVMDialect("emit-llvm-dialect", cl::init(false),
                                     cl::desc("Emit LLVM Dialect"));

static cl::opt<bool> PrintDebugInfo("print-debug-info", cl::init(false),
                                    cl::desc("Print debug info from MLIR"));

static cl::opt<bool> EmitAssembly("S", cl::init(false),
                                  cl::desc("Emit Assembly"));

static cl::opt<bool> NoInline("no-inline", cl::init(false),
                              cl::desc("Prevent inlining"));

static cl::opt<bool> Opt0("O0", cl::init(false), cl::desc("Opt level 0"));
static cl::opt<bool> Opt1("O1", cl::init(false), cl::desc("Opt level 1"));
static cl::opt<bool> Opt2("O2", cl::init(false), cl::desc("Opt level 2"));
static cl::opt<bool> Opt3("O3", cl::init(false), cl::desc("Opt level 3"));

static cl::opt<bool> SCFOpenMP("scf-openmp", cl::init(true),
                               cl::desc("Emit llvm"));

static cl::opt<bool> OpenMPOpt("openmp-opt", cl::init(true),
                               cl::desc("Turn on openmp opt"));

static cl::opt<bool> ParallelLICM("parallel-licm", cl::init(true),
                                  cl::desc("Turn on parallel licm"));

static cl::opt<bool> InnerSerialize("inner-serialize", cl::init(false),
                                    cl::desc("Turn on parallel licm"));

static cl::opt<bool>
    EarlyInnerSerialize("early-inner-serialize", cl::init(false),
                        cl::desc("Perform early inner serialization"));

static cl::opt<bool> ShowAST("show-ast", cl::init(false), cl::desc("Show AST"));

static cl::opt<bool> ImmediateMLIR("immediate", cl::init(false),
                                   cl::desc("Emit immediate mlir"));

static cl::opt<bool> RaiseToAffine("raise-scf-to-affine", cl::init(false),
                                   cl::desc("Raise SCF to Affine"));

static cl::opt<bool> ScalarReplacement("scal-rep", cl::init(true),
                                       cl::desc("Raise SCF to Affine"));

static cl::opt<bool> LoopUnroll("unroll-loops", cl::init(false),
                                cl::desc("Unroll Affine Loops"));

static cl::opt<bool>
    DetectReduction("detect-reduction", cl::init(false),
                    cl::desc("Detect reduction in inner most loop"));

static cl::opt<std::string> Standard("std", cl::init(""),
                                     cl::desc("C/C++ std"));

static cl::opt<std::string> AMDGPUArch("amd-gpu-arch", cl::init(""),
                                       cl::desc("AMD GPU arch"));

static cl::opt<std::string> CUDAGPUArch("cuda-gpu-arch", cl::init(""),
                                        cl::desc("CUDA GPU arch"));

static cl::opt<std::string> CUDAPath("cuda-path", cl::init(""),
                                     cl::desc("CUDA Path"));

static cl::opt<std::string> ROCMPath("rocm-path", cl::init(""),
                                     cl::desc("ROCM Path"));

static cl::opt<bool> NoCUDAInc("nocudainc", cl::init(false),
                               cl::desc("Do not include CUDA headers"));

static cl::opt<bool> NoCUDALib("nocudalib", cl::init(false),
                               cl::desc("Do not link CUDA libdevice"));

static cl::opt<std::string> Output("o", cl::init("-"), cl::desc("Output file"));

static cl::opt<std::string> cfunction("function",
                                      cl::desc("<Specify function>"),
                                      cl::init("*"), cl::cat(toolOptions));

static cl::opt<bool> FOpenMP("fopenmp", cl::init(false),
                             cl::desc("Enable OpenMP"));

static cl::opt<std::string> ToCPU("cpuify", cl::init(""),
                                  cl::desc("Convert to cpu"));

static cl::opt<std::string> MArch("march", cl::init(""),
                                  cl::desc("Architecture"));

static cl::opt<std::string> ResourceDir("resource-dir", cl::init(""),
                                        cl::desc("Resource-dir"));

static cl::opt<std::string> SysRoot("sysroot", cl::init(""),
                                    cl::desc("sysroot"));

static cl::opt<bool> EarlyVerifier("early-verifier", cl::init(false),
                                   cl::desc("Enable verifier ASAP"));

static cl::opt<bool> Verbose("v", cl::init(false), cl::desc("Verbose"));

static cl::opt<std::string> Lang("x", cl::init(""),
                                 cl::desc("Treat input as language"));

static cl::list<std::string> includeDirs("I", cl::desc("include search path"),
                                         cl::cat(toolOptions));

static cl::list<std::string> defines("D", cl::desc("defines"),
                                     cl::cat(toolOptions));

static cl::list<std::string> Includes("include", cl::desc("includes"),
                                      cl::cat(toolOptions));

static cl::opt<std::string> TargetTripleOpt("target", cl::init(""),
                                            cl::desc("Target triple"),
                                            cl::cat(toolOptions));

static cl::opt<bool> InBoundsGEP("inbounds-gep", cl::init(false),
                                 cl::desc("Use inbounds GEP operations"),
                                 cl::cat(toolOptions));

static cl::opt<int>
    CanonicalizeIterations("canonicalizeiters", cl::init(400),
                           cl::desc("Number of canonicalization iterations"));

static cl::opt<std::string>
    McpuOpt("mcpu", cl::init(""), cl::desc("Target CPU"), cl::cat(toolOptions));

static cl::opt<bool> PMEnablePrinting(
    "pm-enable-printing", cl::init(false),
    cl::desc("Enable printing of IR before and after all passes"));

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

class PolygeistCudaDetectorArgList : public llvm::opt::ArgList {
public:
  virtual ~PolygeistCudaDetectorArgList() {}
  template <typename... OptSpecifiers> bool hasArg(OptSpecifiers... Ids) const {
    std::vector _Ids({Ids...});
    for (auto &Id : _Ids) {
      if (Id == clang::driver::options::OPT_nogpulib) {
        continue;
      } else if (Id == clang::driver::options::OPT_cuda_path_EQ) {
        if (CUDAPath == "")
          continue;
        else
          return true;
      } else if (Id == clang::driver::options::OPT_cuda_path_ignore_env) {
        continue;
      } else {
        continue;
      }
    }
    return false;
  }
  StringRef getLastArgValue(llvm::opt::OptSpecifier Id,
                            StringRef Default = "") const {
    if (Id == clang::driver::options::OPT_cuda_path_EQ) {
      return CUDAPath;
    }
    return Default;
  }
  const char *getArgString(unsigned Index) const override { return ""; }
  unsigned getNumInputArgStrings() const override { return 0; }
  const char *MakeArgStringRef(StringRef Str) const override { return ""; }
};

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

extern int cc1_main(ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr);
extern int cc1as_main(ArrayRef<const char *> Argv, const char *Argv0,
                      void *MainAddr);
extern int cc1gen_reproducer_main(ArrayRef<const char *> Argv,
                                  const char *Argv0, void *MainAddr,
                                  const llvm::ToolContext &ToolContext);
std::string GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes) {
    SmallString<128> ExecutablePath(Argv0);
    // Do a PATH lookup if Argv0 isn't a valid path.
    if (!llvm::sys::fs::exists(ExecutablePath))
      if (llvm::ErrorOr<std::string> P =
              llvm::sys::findProgramByName(ExecutablePath))
        ExecutablePath = *P;
    return std::string(ExecutablePath.str());
  }

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void *)(intptr_t)GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}

static int ExecuteCC1Tool(SmallVectorImpl<const char *> &ArgV,
                          const llvm::ToolContext &ToolContext) {
  // If we call the cc1 tool from the clangDriver library (through
  // Driver::CC1Main), we need to clean up the options usage count. The options
  // are currently global, and they might have been used previously by the
  // driver.
  llvm::cl::ResetAllOptionOccurrences();

  llvm::BumpPtrAllocator A;
  llvm::cl::ExpansionContext ECtx(A, llvm::cl::TokenizeGNUCommandLine);
  if (llvm::Error Err = ECtx.expandResponseFiles(ArgV)) {
    llvm::errs() << toString(std::move(Err)) << '\n';
    return 1;
  }
  StringRef Tool = ArgV[1];
  void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
  if (Tool == "-cc1")
    return cc1_main(makeArrayRef(ArgV).slice(1), ArgV[0], GetExecutablePathVP);
  if (Tool == "-cc1as")
    return cc1as_main(makeArrayRef(ArgV).slice(2), ArgV[0],
                      GetExecutablePathVP);
  if (Tool == "-cc1gen-reproducer")
    return cc1gen_reproducer_main(makeArrayRef(ArgV).slice(2), ArgV[0],
                                  GetExecutablePathVP, ToolContext);
  // Reject unknown tools.
  llvm::errs() << "error: unknown integrated tool '" << Tool << "'. "
               << "Valid tools include '-cc1' and '-cc1as'.\n";
  return 1;
}

int emitBinary(char *Argv0, const char *filename,
               SmallVectorImpl<const char *> &LinkArgs, bool LinkOMP) {

  using namespace clang;
  using namespace clang::driver;
  using namespace std;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagBuffer =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagBuffer);

  string TargetTriple;
  if (TargetTripleOpt == "")
    TargetTriple = llvm::sys::getDefaultTargetTriple();
  else
    TargetTriple = TargetTripleOpt;

  const char *binary = Argv0;
  const unique_ptr<Driver> driver(new Driver(binary, TargetTriple, Diags));
  mlirclang::ArgumentList Argv;
  Argv.push_back(Argv0);
  // Argv.push_back("-x");
  // Argv.push_back("ir");
  Argv.push_back(filename);
  if (LinkOMP)
    Argv.push_back("-fopenmp");
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    Argv.emplace_back(ResourceDir);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (CUDAGPUArch != "") {
    Argv.emplace_back("--cuda-gpu-arch=", CUDAGPUArch);
  }
  if (CUDAPath != "") {
    Argv.emplace_back("--cuda-path=", CUDAPath);
  }
  if (Opt0) {
    Argv.push_back("-O0");
  }
  if (Opt1) {
    Argv.push_back("-O1");
  }
  if (Opt2) {
    Argv.push_back("-O2");
  }
  if (Opt3) {
    Argv.push_back("-O3");
  }
  if (Output != "") {
    Argv.push_back("-o");
    Argv.emplace_back(Output);
  }
  for (const auto *arg : LinkArgs)
    Argv.push_back(arg);

  const unique_ptr<Compilation> compilation(
      driver->BuildCompilation(Argv.getArguments()));

  if (ResourceDir != "")
    driver->ResourceDir = ResourceDir;
  if (SysRoot != "")
    driver->SysRoot = SysRoot;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  int Res = 0;

  driver->ExecuteCompilation(*compilation, FailingCommands);
  for (const auto &P : FailingCommands) {
    int CommandRes = P.first;
    const Command *FailingCommand = P.second;
    if (!Res)
      Res = CommandRes;

    // If result status is < 0, then the driver command signalled an error.
    // If result status is 70, then the driver command reported a fatal error.
    // On Windows, abort will return an exit code of 3.  In these cases,
    // generate additional diagnostic information if possible.
    bool IsCrash = CommandRes < 0 || CommandRes == 70;
#ifdef _WIN32
    IsCrash |= CommandRes == 3;
#endif
    if (IsCrash) {
      driver->generateCompilationDiagnostics(*compilation, *FailingCommand);
      break;
    }
  }
  Diags.getClient()->finish();

  return Res;
}

#include "Lib/clang-mlir.cc"
int main(int argc, char **argv) {

  if (argc >= 1) {
    if (std::string(argv[1]) == "-cc1") {
      SmallVector<const char *> Argv;
      for (int i = 0; i < argc; i++)
        Argv.push_back(argv[i]);
      return ExecuteCC1Tool(Argv, {});
    }
  }
  SmallVector<const char *> LinkageArgs;
  SmallVector<const char *> MLIRArgs;
  {
    bool linkOnly = false;
    for (int i = 0; i < argc; i++) {
      StringRef ref(argv[i]);
      if (ref == "-Wl,--start-group")
        linkOnly = true;
      if (!linkOnly) {
        if (ref == "-fPIC" || ref == "-c" || ref.startswith("-fsanitize")) {
          LinkageArgs.push_back(argv[i]);
        } else if (ref == "-L" || ref == "-l") {
          LinkageArgs.push_back(argv[i]);
          i++;
          LinkageArgs.push_back(argv[i]);
        } else if (ref.startswith("-L") || ref.startswith("-l") ||
                   ref.startswith("-Wl")) {
          LinkageArgs.push_back(argv[i]);
        } else if (ref == "-D" || ref == "-I") {
          MLIRArgs.push_back(argv[i]);
          i++;
          MLIRArgs.push_back(argv[i]);
        } else if (ref.startswith("-D")) {
          MLIRArgs.push_back("-D");
          MLIRArgs.push_back(&argv[i][2]);
        } else if (ref.startswith("-I")) {
          MLIRArgs.push_back("-I");
          MLIRArgs.push_back(&argv[i][2]);
        } else if (ref == "-g") {
          LinkageArgs.push_back(argv[i]);
        } else {
          MLIRArgs.push_back(argv[i]);
        }
      } else {
        LinkageArgs.push_back(argv[i]);
      }
      if (ref == "-Wl,--end-group")
        linkOnly = false;
    }
  }
  using namespace mlir;

  int size = MLIRArgs.size();
  const char **data = MLIRArgs.data();
  InitLLVM y(size, data);
  std::vector<std::string> files;
  {
    cl::list<std::string> inputFileName(cl::Positional, cl::OneOrMore,
                                        cl::desc("<Specify input file>"),
                                        cl::cat(toolOptions));
    cl::ParseCommandLineOptions(size, data);
    assert(inputFileName.size());
    for (auto inp : inputFileName) {
      std::ifstream inputFile(inp);
      if (!inputFile.good()) {
        outs() << "Not able to open file: " << inp << "\n";
        return -1;
      }
      files.push_back(inp);
    }
  }

  mlir::registerAllPasses();
  mlir::registerAllTranslations();
  mlir::DialectRegistry registry;
  mlir::registerOpenMPDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::func::registerInlinerExtension(registry);
  polygeist::registerGpuSerializeToCubinPass();
  polygeist::registerGpuSerializeToHsacoPass();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllFromLLVMIRTranslations(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  MLIRContext context(registry);

  context.disableMultithreading();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<DLTIDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::async::AsyncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::polygeist::PolygeistDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMArrayType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);
  IndexType::attachInterface<PtrElementModel<IndexType>>(context);
  LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(
      context);
  LLVM::LLVMPointerType::attachInterface<
      PtrElementModel<LLVM::LLVMPointerType>>(context);
  LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
      context);

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::OpBuilder(&context).getUnknownLoc()));

  llvm::Triple triple;
  llvm::DataLayout DL("");
  llvm::Triple gpuTriple;
  llvm::DataLayout gpuDL("");
  if (!parseMLIR(argv[0], files, cfunction, includeDirs, defines, module,
                 triple, DL, gpuTriple, gpuDL)) {
    return 1;
  }

  auto convertGepInBounds = [](llvm::Module &llvmModule) {
    for (auto &F : llvmModule) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto g = dyn_cast<GetElementPtrInst>(&I))
            g->setIsInBounds(true);
        }
      }
    }
  };
  auto addLICM = [](auto &pm) {
    if (ParallelLICM)
      pm.addPass(polygeist::createParallelLICMPass());
    else
      pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  };
  auto enablePrinting = [](auto &pm) {
    if (PMEnablePrinting)
      pm.enableIRPrinting();
  };

  mlir::PassManager pm(&context);
  enablePrinting(pm);

  OpPrintingFlags flags;
  if (PrintDebugInfo)
    flags.enableDebugInfo(/*pretty*/ false);

  if (ImmediateMLIR) {
    module->print(llvm::outs(), flags);
    return 0;
  }

  int optLevel = 0;
  if (Opt0)
    optLevel = 0;
  if (Opt1)
    optLevel = 1;
  if (Opt2)
    optLevel = 2;
  if (Opt3)
    optLevel = 3;

#if !POLYGEIST_ENABLE_CUDA
  if (EmitCUDA) {
    llvm::errs() << "error: no CUDA support, aborting\n";
    return 1;
  }
#endif
#if !POLYGEIST_ENABLE_ROCM
  if (EmitROCM) {
    llvm::errs() << "error: no ROCM support, aborting\n";
    return 1;
  }
#endif
  if (EmitCUDA && EmitROCM) {
    llvm::errs() << "Cannot emit both CUDA and ROCM\n";
    return 1;
  }
  bool EmitGPU = EmitROCM || EmitCUDA;

  int unrollSize = 32;
  bool LinkOMP = FOpenMP;
  pm.enableVerifier(EarlyVerifier);

  pm.addPass(polygeist::createConvertToOpaquePtrPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;
  if (true) {
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::createPolygeistMem2RegPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::createPolygeistMem2RegPass());
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::createRemoveTrivialUsePass());
    optPM.addPass(polygeist::createPolygeistMem2RegPass());
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::createLoopRestructurePass());
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    if (ScalarReplacement)
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    addLICM(optPM);
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::createCanonicalizeForPass());
    optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
        canonicalizerConfig, {}, {}));
    if (RaiseToAffine) {
      optPM.addPass(polygeist::createCanonicalizeForPass());
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      addLICM(optPM);
      optPM.addPass(polygeist::createRaiseSCFToAffinePass());
      optPM.addPass(polygeist::replaceAffineCFGPass());
      if (ScalarReplacement)
        optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }

    {
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();

      if (DetectReduction)
        optPM.addPass(polygeist::detectReductionPass());

      // Disable inlining for -O0
      if (!Opt0) {
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM.addPass(mlir::createCSEPass());
        // Affine must be lowered to enable inlining
        if (RaiseToAffine)
          optPM.addPass(mlir::createLowerAffinePass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        if (!NoInline)
          pm.addPass(mlir::createInlinerPass());
        mlir::OpPassManager &optPM2 = pm.nest<mlir::func::FuncOp>();
        optPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM2.addPass(mlir::createCSEPass());
        optPM2.addPass(polygeist::createPolygeistMem2RegPass());
        optPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM2.addPass(mlir::createCSEPass());
        optPM2.addPass(polygeist::createCanonicalizeForPass());
        if (RaiseToAffine) {
          optPM2.addPass(polygeist::createRaiseSCFToAffinePass());
        }
        optPM2.addPass(polygeist::replaceAffineCFGPass());
        optPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM2.addPass(mlir::createCSEPass());
        addLICM(optPM2);
        optPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
      }
    }

    if (CudaLower || EmitROCM) {
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::createLowerAffinePass());
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      if (CudaLower) {
        pm.addPass(polygeist::createParallelLowerPass(
            /* wrapParallelOps */ EmitGPU, GPUKernelStructureMode));
      }
      pm.addPass(polygeist::createConvertCudaRTtoGPUPass());
      if (ToCPU.size() > 0) {
        pm.addPass(polygeist::createConvertCudaRTtoCPUPass());
      } else if (EmitROCM) {
        pm.addPass(polygeist::createConvertCudaRTtoHipRTPass());
      }
      pm.addPass(mlir::createSymbolDCEPass());
      mlir::OpPassManager &noptPM = pm.nest<mlir::func::FuncOp>();
      noptPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      noptPM.addPass(polygeist::createPolygeistMem2RegPass());
      noptPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      if (!NoInline)
        pm.addPass(mlir::createInlinerPass());
      mlir::OpPassManager &noptPM2 = pm.nest<mlir::func::FuncOp>();
      noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      noptPM2.addPass(polygeist::createPolygeistMem2RegPass());
      noptPM2.addPass(polygeist::createCanonicalizeForPass());
      noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      noptPM2.addPass(mlir::createCSEPass());
      addLICM(noptPM2);
      noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      if (RaiseToAffine) {
        noptPM2.addPass(polygeist::createCanonicalizeForPass());
        noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        addLICM(noptPM2);
        noptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
        noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        noptPM2.addPass(polygeist::replaceAffineCFGPass());
        noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        if (LoopUnroll)
          noptPM2.addPass(
              mlir::affine::createLoopUnrollPass(unrollSize, false, true));
        noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        noptPM2.addPass(mlir::createCSEPass());
        noptPM2.addPass(polygeist::createPolygeistMem2RegPass());
        noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        addLICM(noptPM2);
        noptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
        noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        noptPM2.addPass(polygeist::replaceAffineCFGPass());
        noptPM2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        if (ScalarReplacement)
          noptPM2.addPass(mlir::affine::createAffineScalarReplacementPass());
      }
    }

    if (CudaLower) {
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      optPM.addPass(mlir::createCSEPass());
      optPM.addPass(polygeist::createPolygeistMem2RegPass());
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      optPM.addPass(mlir::createCSEPass());
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      optPM.addPass(polygeist::createCanonicalizeForPass());
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));

      if (RaiseToAffine) {
        optPM.addPass(polygeist::createCanonicalizeForPass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        addLICM(optPM);
        optPM.addPass(polygeist::createRaiseSCFToAffinePass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM.addPass(polygeist::replaceAffineCFGPass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        if (ScalarReplacement)
          optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
      }
      if (ToCPU == "continuation") {
        optPM.addPass(polygeist::createBarrierRemovalContinuation());
        // pm.nest<mlir::FuncOp>().addPass(mlir::polygeist::createPolygeistCanonicalizePass());
      } else if (ToCPU.size() != 0) {
        optPM.addPass(polygeist::createCPUifyPass(ToCPU));
      }
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      optPM.addPass(mlir::createCSEPass());
      optPM.addPass(polygeist::createPolygeistMem2RegPass());
      optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      optPM.addPass(mlir::createCSEPass());
      if (RaiseToAffine) {
        optPM.addPass(polygeist::createCanonicalizeForPass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        addLICM(optPM);
        if (EarlyInnerSerialize) {
          optPM.addPass(mlir::createLowerAffinePass());
          optPM.addPass(polygeist::createInnerSerializationPass());
          optPM.addPass(polygeist::createCanonicalizeForPass());
        }
        optPM.addPass(polygeist::createRaiseSCFToAffinePass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM.addPass(polygeist::replaceAffineCFGPass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        if (LoopUnroll)
          optPM.addPass(
              mlir::affine::createLoopUnrollPass(unrollSize, false, true));
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM.addPass(mlir::createCSEPass());
        optPM.addPass(polygeist::createPolygeistMem2RegPass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        addLICM(optPM);
        optPM.addPass(polygeist::createRaiseSCFToAffinePass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        optPM.addPass(polygeist::replaceAffineCFGPass());
        optPM.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
        if (ScalarReplacement)
          optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
      }
    }
    pm.addPass(mlir::createSymbolDCEPass());

#ifdef POLYGEIST_ENABLE_POLYMER
    if (ClPolyhedralOpt) {
      pm.addPass(polygeist::createPolyhedralOptPass());
      pm.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
    }
#endif

    if (EmitGPU || EmitLLVM || !EmitAssembly || EmitOpenMPIR ||
        EmitLLVMDialect) {
      pm.addPass(mlir::createLowerAffinePass());
      if (InnerSerialize)
        pm.addPass(polygeist::createInnerSerializationPass());
      addLICM(pm);
    }

#if POLYGEIST_ENABLE_GPU
    if (EmitGPU) {
      pm.addPass(mlir::createCSEPass());
      if (CudaLower)
        pm.addPass(polygeist::createConvertParallelToGPUPass1(
            EmitCUDA ? CUDAGPUArch : AMDGPUArch));
      // We cannot canonicalize here because we have sunk some operations in the
      // kernel which the canonicalizer would hoist

      // TODO pass in gpuDL, the format is weird
      pm.addPass(mlir::createGpuKernelOutliningPass());
      pm.addPass(polygeist::createMergeGPUModulesPass());
      pm.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      // TODO maybe preserve info about which original kernel corresponds to
      // which outlined kernel, might be useful for calls to
      // cudaFuncSetCacheConfig e.g.
      pm.addPass(polygeist::createConvertParallelToGPUPass2(
          EmitGPUKernelLaunchBounds));
      pm.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));

      addLICM(pm);

      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));

      mlir::OpPassManager &gpuPM = pm.nest<gpu::GPUModuleOp>();
      gpuPM.addPass(polygeist::createFixGPUFuncPass());
      pm.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      pm.addPass(polygeist::createLowerAlternativesPass());
      pm.addPass(polygeist::createCollectKernelStatisticsPass());
    }
#endif

    if (mlir::failed(pm.run(module.get()))) {
      module->dump();
      return 12;
    }

    // Prune unused gpu module funcs
    module.get()->walk([&](gpu::GPUModuleOp gpum) {
      bool changed;
      do {
        changed = false;
        std::vector<Operation *> unused;
        gpum->walk([&](Operation *op) {
          if (isa<gpu::GPUFuncOp>(op) || isa<func::FuncOp>(op) ||
              isa<LLVM::LLVMFuncOp>(op)) {
            auto symbolUses = SymbolTable::getSymbolUses(op, module.get());
            if (symbolUses && symbolUses->empty()) {
              unused.push_back(op);
            }
          }
        });
        for (auto op : unused) {
          changed = true;
          op->erase();
        }
      } while (changed);
    });

    if (EmitLLVM || !EmitAssembly || EmitOpenMPIR || EmitLLVMDialect) {
      mlir::PassManager pm2(&context);
      enablePrinting(pm2);
      if (SCFOpenMP) {
        pm2.addPass(createConvertSCFToOpenMPPass());
      } else
        pm2.addPass(polygeist::createSerializationPass());
      pm2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      if (OpenMPOpt) {
        pm2.addPass(polygeist::createOpenMPOptPass());
        pm2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));
      }
      pm2.nest<mlir::func::FuncOp>().addPass(
          polygeist::createPolygeistMem2RegPass());
      pm2.addPass(mlir::createCSEPass());
      pm2.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
          canonicalizerConfig, {}, {}));
      if (mlir::failed(pm2.run(module.get()))) {
        module->dump();
        return 9;
      }
      if (!EmitOpenMPIR) {
        module->walk([&](mlir::omp::ParallelOp) { LinkOMP = true; });
        mlir::PassManager pm3(&context);
        enablePrinting(pm3);
        LowerToLLVMOptions options(&context);
        options.dataLayout = DL;
        // invalid for gemm.c init array
        // options.useBarePtrCallConv = true;

#if POLYGEIST_ENABLE_GPU
        if (EmitGPU) {
          // Set the max block size to 1024 by default for ROCM (otherwise it
          // will be 256)
          OpBuilder builder(module.get()->getContext());
          module.get().walk([&](gpu::GPUFuncOp gpuFuncOp) {
            StringRef attrName = "rocdl.max_flat_work_group_size";
            if (!gpuFuncOp->hasAttr(attrName)) {
              gpuFuncOp->setAttr(attrName, builder.getIntegerAttr(
                                               builder.getIndexType(), 1024));
            }
          });

          pm3.addPass(polygeist::createConvertPolygeistToLLVMPass(
              options, CStyleMemRef, /* onlyGpuModules */ true,
              EmitCUDA ? "cuda" : "rocm"));

          using namespace clang;
          using namespace clang::driver;
          using namespace std;
          IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
          IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
              new DiagnosticOptions();
          TextDiagnosticPrinter *DiagBuffer =
              new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
          DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagBuffer);
          const unique_ptr<Driver> driver(
              new Driver("clang", triple.str(), Diags));
          PolygeistCudaDetectorArgList argList;
          CudaInstallationDetector detector(*driver, triple, argList);

          if (EmitCUDA) {
#if POLYGEIST_ENABLE_CUDA
            std::string arch = CUDAGPUArch;
            if (arch == "")
              arch = "sm_60";
            std::string libDevicePath = detector.getLibDeviceFile(arch);
            std::string ptxasPath =
                std::string(detector.getBinPath()) + "/ptxas";

            // TODO what should the ptx version be?
            mlir::OpPassManager &gpuPM = pm3.nest<gpu::GPUModuleOp>();
            gpuPM.addPass(polygeist::createGpuSerializeToCubinPass(
                arch, "+ptx74", optLevel, DeviceOptLevel, ptxasPath,
                libDevicePath, OutputIntermediateGPU));
#endif
          } else if (EmitROCM) {
#if POLYGEIST_ENABLE_ROCM
            std::string arch = AMDGPUArch;
            if (arch == "")
              arch = "gfx1030";

            {
              // AMDGPU triple is fixed for our purposes
              auto triple = "amdgcn-amd-amdhsa";
              // TODO this should probably depend on the gpu arch
              auto DL = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:"
                        "32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:"
                        "128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-"
                        "n32:64-S32-A5-G1-ni:7";

              module.get()->setAttr(
                  StringRef("polygeist.gpu_module." +
                            LLVM::LLVMDialect::getTargetTripleAttrName().str()),
                  StringAttr::get(module->getContext(), triple));
              module.get()->setAttr(
                  StringRef("polygeist.gpu_module." +
                            LLVM::LLVMDialect::getDataLayoutAttrName().str()),
                  StringAttr::get(module.get()->getContext(), DL));
            }

            mlir::OpPassManager &gpuPM = pm3.nest<gpu::GPUModuleOp>();
            int HsaOptLevel = DeviceOptLevel;
            gpuPM.addPass(polygeist::createGpuSerializeToHsacoPass(
                arch, "", optLevel,
                /* TODO do we need this param? */ HsaOptLevel, ROCMPath,
                OutputIntermediateGPU));
#endif
          } else {
            assert(0);
            llvm_unreachable("?");
          }
        }
#endif

        pm3.addPass(polygeist::createConvertPolygeistToLLVMPass(
            options, CStyleMemRef, /* onlyGpuModules */ false,
            EmitCUDA ? "cuda" : "rocm"));
        pm3.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
            canonicalizerConfig, {}, {}));

        if (mlir::failed(pm3.run(module.get()))) {
          module->dump();
          return 10;
        }
      }
    }
    if (mlir::failed(mlir::verify(module.get()))) {
      module->dump();
      return 5;
    }
  }

  if (EmitLLVM || !EmitAssembly) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule) {
      module->dump();
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
    }
#if POLYGEIST_ENABLE_CUDA
    if (EmitCUDA) {
// This header defines:
// unsigned char CudaRuntimeWrappers_cpp_bc[]
// unsigned int CudaRuntimeWrappers_cpp_bc_len
#include "../lib/polygeist/ExecutionEngine/CudaRuntimeWrappers.cpp.bin.h"
      StringRef blobStrRef((const char *)CudaRuntimeWrappers_cpp_bc,
                           CudaRuntimeWrappers_cpp_bc_len);
      MemoryBufferRef blobMemoryBufferRef(blobStrRef, "Binary include");
      llvm::SMDiagnostic err;
      std::unique_ptr<llvm::Module> cudaWrapper =
          llvm::parseIR(blobMemoryBufferRef, err, llvmContext);
      if (!cudaWrapper || llvm::verifyModule(*cudaWrapper, &llvm::errs())) {
        llvm::errs() << "Failed to load CUDA wrapper bitcode module\n";
        return -1;
      }
      // Link in required wrapper functions
      //
      // TODO currently the wrapper symbols have weak linkage which does not
      // allow them to be inlined - we should either internalize them after
      // linking or make them linkeonce_odr (preferred) in so that llvm can
      // inline them
      llvm::Linker::linkModules(*llvmModule, std::move(cudaWrapper),
                                llvm::Linker::Flags::LinkOnlyNeeded);
    }
#endif
#if POLYGEIST_ENABLE_ROCM
    if (EmitROCM) {
// This header defines:
// unsigned char RocmRuntimeWrappers_cpp_bc[]
// unsigned int RocmRuntimeWrappers_cpp_bc_len
#include "../lib/polygeist/ExecutionEngine/RocmRuntimeWrappers.cpp.bin.h"
      StringRef blobStrRef((const char *)RocmRuntimeWrappers_cpp_bc,
                           RocmRuntimeWrappers_cpp_bc_len);
      MemoryBufferRef blobMemoryBufferRef(blobStrRef, "Binary include");
      llvm::SMDiagnostic err;
      std::unique_ptr<llvm::Module> rocmWrapper =
          llvm::parseIR(blobMemoryBufferRef, err, llvmContext);
      if (!rocmWrapper || llvm::verifyModule(*rocmWrapper, &llvm::errs())) {
        llvm::errs() << "Failed to load ROCM wrapper bitcode module\n";
        return -1;
      }
      // Link in required wrapper functions
      //
      // TODO currently the wrapper symbols have weak linkage which does not
      // allow them to be inlined - we should either internalize them after
      // linking or make them linkeonce_odr (preferred) in so that llvm can
      // inline them
      llvm::Linker::linkModules(*llvmModule, std::move(rocmWrapper),
                                llvm::Linker::Flags::LinkOnlyNeeded);
    }
#endif
    if (InBoundsGEP) {
      convertGepInBounds(*llvmModule);
    }
    for (auto &F : *llvmModule) {
      for (auto AttrName : {"target-cpu", "tune-cpu", "target-features"})
        if (auto V = module.get()->getAttrOfType<mlir::StringAttr>(
                (StringRef("polygeist.") + AttrName).str())) {
          F.addFnAttr(AttrName, V.getValue());
        }
    }
    if (auto F = llvmModule->getFunction("malloc")) {
      // allocsize
      for (auto Attr : {llvm::Attribute::MustProgress, llvm::Attribute::NoFree,
                        llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn})
        F->addFnAttr(Attr);
      F->setOnlyAccessesInaccessibleMemory();
      F->addRetAttr(llvm::Attribute::NoAlias);
      F->addRetAttr(llvm::Attribute::NoUndef);
      SmallVector<llvm::Value *> todo = {F};
      while (todo.size()) {
        auto cur = todo.back();
        todo.pop_back();
        if (isa<llvm::Function>(cur)) {
          for (auto u : cur->users())
            todo.push_back(u);
          continue;
        }
        if (auto CE = dyn_cast<llvm::ConstantExpr>(cur))
          if (CE->isCast()) {
            for (auto u : cur->users())
              todo.push_back(u);
            continue;
          }
        if (auto CI = dyn_cast<llvm::CallInst>(cur)) {
          CI->addRetAttr(llvm::Attribute::NoAlias);
          CI->addRetAttr(llvm::Attribute::NoUndef);
        }
      }
    }
    llvmModule->setDataLayout(DL);
    llvmModule->setTargetTriple(triple.getTriple());
    if (!EmitAssembly) {
      auto tmpFile =
          llvm::sys::fs::TempFile::create("/tmp/intermediate%%%%%%%.ll");
      if (!tmpFile) {
        llvm::errs() << "Failed to create temp file\n";
        return -1;
      }
      std::error_code EC;
      {
        llvm::raw_fd_ostream out(tmpFile->FD, /*shouldClose*/ false);
        out << *llvmModule << "\n";
        out.flush();
      }
      int res =
          emitBinary(argv[0], tmpFile->TmpName.c_str(), LinkageArgs, LinkOMP);
      if (tmpFile->discard()) {
        llvm::errs() << "Failed to erase temp file\n";
        return -1;
      }
      return res;
    } else if (Output == "-") {
      llvm::outs() << *llvmModule << "\n";
    } else {
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      out << *llvmModule << "\n";
    }

  } else {
    if (Output == "-") {
      module->print(outs(), flags);
    } else {
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      module->print(out, flags);
    }
  }
  return 0;
}
