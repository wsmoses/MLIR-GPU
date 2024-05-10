add_mlir_library(PolymerSupport
  OslScop.cc
  OslScopStmtOpSet.cc
  OslSymbolTable.cc
  ScopStmt.cc
  ScatteringUtils.cc
  Utils.cc

  DEPENDS
  mlir-headers

  ADDITIONAL_HEADERS
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Support

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRAffineAnalysis

  # libosl
   libcloog
  # libisl
  libplutoisl
  libplutoosl
  libplutopip
  libplutopolylib
  libplutocloog
  libplutopet
  libplutocandl
  libpluto
)

target_link_libraries(PolymerSupport PUBLIC ${GMP_LIBRARY})
