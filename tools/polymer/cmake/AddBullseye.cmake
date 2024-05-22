
# execute_process(
#   COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../build_pluto.sh" "${POLYGEIST_PLUTO_DIR}"
# )

set(BULLSEYE_INCLUDE_DIR "${POLYGEIST_PLUTO_DIR}/bullseye/install/include")
set(BULLSEYE_LIB_DIR "${POLYGEIST_PLUTO_DIR}/bullseye/install/lib")
set(BULLSEYE_LLVM_LIB_DIR "${POLYGEIST_PLUTO_DIR}/llvm/install/lib")

add_library(libbullseyellvmclang SHARED IMPORTED)
set_target_properties(libbullseyellvmclang PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LLVM_LIB_DIR}/libclang-cpp.so.10")
add_library(libbullseyellvm SHARED IMPORTED)
set_target_properties(libbullseyellvm PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LLVM_LIB_DIR}/libLLVM-10.so")

add_dependencies(libbullseyellvmclang bullseye)
add_dependencies(libbullseyellvm bullseye)
add_library(libbullseyebarvinok STATIC IMPORTED)
set_target_properties(libbullseyebarvinok PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libbarvinok.a")
add_library(libbullseyeisl STATIC IMPORTED)
set_target_properties(libbullseyeisl PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libisl.a")
add_library(libbullseyepet SHARED IMPORTED)
set_target_properties(libbullseyepet PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libpet.so")
# add_library(libplutopet SHARED IMPORTED)
# set_target_properties(libplutopet PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpet.so")
add_library(libbullseyepolylibgmp STATIC IMPORTED)
set_target_properties(libbullseyepolylibgmp PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libpolylibgmp.a")

add_library(libbullseye STATIC IMPORTED)
set_target_properties(libbullseye PROPERTIES
    IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libbullseye.a"
    # INTERFACE_LINK_LIBRARIES "libbullseyebarvinok;libbullseyeisl;libbullseyepet;libbullseyepolylibgmp;libbullseyellvmclang;libbullseyellvm"
    # INTERFACE_INCLUDE_DIRECTORIES "${BULLSEYE_INCLUDE_DIR};${BULLSEYE_LLVM_INCLUDE_DIR}"
)

add_dependencies(libbullseye bullseye)
add_dependencies(libbullseyebarvinok bullseye)
add_dependencies(libbullseyeisl bullseye)
add_dependencies(libbullseyepet bullseye)
add_dependencies(libbullseyepolylibgmp bullseye)