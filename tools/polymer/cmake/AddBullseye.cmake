
# execute_process(
#   COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../build_pluto.sh" "${POLYGEIST_PLUTO_DIR}"
# )

set(BULLSEYE_INCLUDE_DIR "${POLYGEIST_PLUTO_DIR}/bullseye/install/include")
set(BULLSEYE_LIB_DIR "${POLYGEIST_PLUTO_DIR}/bullseye/install/lib")

add_library(libbullseye STATIC IMPORTED)
set_target_properties(libbullseye PROPERTIES IMPORTED_LOCATION "${BULLSEYE_LIB_DIR}/libbullseye.a")

add_dependencies(libbullseye bullseye)

