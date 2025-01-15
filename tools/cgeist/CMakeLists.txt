set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  IPO
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
)

add_clang_executable(cgeist
  driver.cc
  Lib/CGStmt.cc
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1as_main.cpp"
  "${LLVM_SOURCE_DIR}/../clang/tools/driver/cc1gen_reproducer_main.cpp"
  Lib/pragmaHandler.cc
  Lib/AffineUtils.cc
  Lib/ValueCategory.cc
  Lib/utils.cc
  Lib/IfScope.cc
  Lib/TypeUtils.cc
  Lib/CGCall.cc 
)
if(POLYGEIST_ENABLE_CUDA)
  target_compile_definitions(cgeist
    PRIVATE
    POLYGEIST_ENABLE_CUDA=1
  )
  add_dependencies(cgeist execution_engine_cuda_wrapper_binary_include)
endif()
if(POLYGEIST_ENABLE_ROCM)
  target_compile_definitions(cgeist
    PRIVATE
    POLYGEIST_ENABLE_ROCM=1
  )
  add_dependencies(cgeist execution_engine_rocm_wrapper_binary_include)
endif()
install(TARGETS cgeist
EXPORT PolygeistTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT cgeist)

target_include_directories(cgeist PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

if(POLYGEIST_ENABLE_POLYMER)
  target_include_directories(cgeist PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../polymer/include"
  )
  target_link_libraries(cgeist PRIVATE
    PolymerTransforms
  )
  target_compile_definitions(cgeist
    PRIVATE
    POLYGEIST_ENABLE_POLYMER=1
  )
endif()


target_compile_definitions(cgeist PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")
set(CGEIST_LIBS
  MLIRPolygeist
  MLIRPolygeistTransforms
)
if(POLYGEIST_ENABLE_POLYMER)
        list(APPEND CGEIST_LIBS
                PolymerTransforms)
endif()
list(APPEND CGEIST_LIBS
  MLIRPolygeist

  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRGPUTransforms
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRFunctionInterfaces
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRSCFTransforms
  MLIRPolygeistTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIRGPUToNVVMTransforms
  MLIRGPUToGPURuntimeTransforms
  MLIRFuncAllExtensions
  MLIRLLVMIRToLLVMTranslation
  MLIRFromLLVMIRTranslationRegistration
  MLIRAffineAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransformUtils
  MLIRCAPIRegisterEverything
  MLIRBuiltinToLLVMIRTranslation

  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangSerialization
)

target_link_libraries(cgeist PRIVATE ${CGEIST_LIBS})
add_dependencies(cgeist MLIRPolygeistOpsIncGen MLIRPolygeistPassIncGen)
add_subdirectory(Test)
