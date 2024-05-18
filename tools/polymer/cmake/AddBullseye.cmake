
# execute_process(
#   COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../build_pluto.sh" "${POLYGEIST_PLUTO_DIR}"
# )

set(BULLSEYE_INCLUDE_DIR "${POLYGEIST_PLUTO_DIR}/bullseye/install/include")
set(BULLSEYE_LIB_DIR "${POLYGEIST_PLUTO_DIR}/bullseye/install/lib")

add_library(libbullseye STATIC IMPORTED)
set_target_properties(libbullseye PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libbullseye.a")
add_library(libbullseyebarvinok STATIC IMPORTED)
set_target_properties(libbullseyebarvinok PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libbarvinok.a")
# message(STATUS "BULLSEYE_LIB_DIR: _> ${BULLSEYE_LIB_DIR}")
add_library(libbullseyeisl STATIC IMPORTED)
set_target_properties(libbullseyeisl PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libisl.a")
add_library(libbullseyepet STATIC IMPORTED)
set_target_properties(libbullseyepet PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libpet.a")
add_library(libbullseyepolylibgmp STATIC IMPORTED)
set_target_properties(libbullseyepolylibgmp PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libpolylibgmp.a")

add_dependencies(libbullseye bullseye)
add_dependencies(libbullseyebarvinok bullseye)
add_dependencies(libbullseyeisl bullseye)
add_dependencies(libbullseyepet bullseye)
add_dependencies(libbullseyepolylibgmp bullseye)

