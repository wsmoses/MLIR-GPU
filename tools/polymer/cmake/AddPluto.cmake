
# execute_process(
#   COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../build_pluto.sh" "${POLYGEIST_PLUTO_DIR}"
# )

set(PLUTO_INCLUDE_DIR "${POLYGEIST_PLUTO_DIR}/pluto/install/include")

# add_library(libpluto SHARED IMPORTED)
# set_target_properties(libpluto PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpluto.so")
# add_library(libplutoosl SHARED IMPORTED)
# set_target_properties(libplutoosl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libosl.so")
# add_library(libplutoisl SHARED IMPORTED)
# set_target_properties(libplutoisl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libisl.so")
# add_library(libplutopip SHARED IMPORTED)
# set_target_properties(libplutopip PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpiplib_dp.so")
# add_library(libplutopolylib SHARED IMPORTED)
# set_target_properties(libplutopolylib PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpolylib64.so")
# add_library(libplutocloog SHARED IMPORTED)
# set_target_properties(libplutocloog PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libcloog-isl.so")
# add_library(libplutocandl STATIC IMPORTED)
# set_target_properties(libplutocandl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libcandl.so")
#
# add_dependencies(libpluto pluto)
# add_dependencies(libplutoisl pluto)
# add_dependencies(libplutoosl pluto)
# add_dependencies(libplutopip pluto)
# add_dependencies(libplutopolylib pluto)
# add_dependencies(libplutocloog pluto)
# add_dependencies(libplutocandl pluto)
