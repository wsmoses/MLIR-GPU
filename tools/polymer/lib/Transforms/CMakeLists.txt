add_mlir_conversion_library(PolymerTransforms
  BullsEyeAnalysis.cc
  PlutoTransform.cc
  Reg2Mem.cc
  ExtractScopStmt.cc
  ScopStmtOpt.cc
  LoopAnnotate.cc
  LoopExtract.cc
  FoldSCFIf.cc
  AnnotateScop.cc

  ADDITIONAL_HEADER_DIRS
  "${POLYMER_MAIN_INCLUDE_DIR}/polymer/Transforms"

  DEPENDS
  PolymerTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  MLIRAffineToStandard
  MLIRAffineTransforms

  PolymerSupport
  PolymerTargetOpenScop
)
