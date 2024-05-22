set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set(POLYMER_TRANSFORM_DEPENDS
  MLIRAffineOpsIncGen
  MLIRAffinePassIncGen
  MLIRLoopLikeInterfaceIncGen
)

set(POLYMER_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set(POLYMER_MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )

set(POLYMER_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")
set(POLYMER_BINARY_DIR  "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(POLYMER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" ) 
set(POLYMER_TOOLS_DIR   ${CMAKE_BINARY_DIR}/bin)

# ----------------------------- Customize CMake

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ----------------------------- Dependencies
# Find whether the GMP package exists.
find_package(GMP REQUIRED)
# find_package(CPLEX REQUIRED)
find_package(NTL REQUIRED)
find_package(GLPK REQUIRED)

# build pluto, openscop, cloog, bullseye
execute_process(
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build_polymer_deps_with_bullseye.sh" "${POLYGEIST_PLUTO_DIR}" "${CPLEX_HOME_DIR}" "${BULLSEYE_LLVM_INSTALL_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Polygeist Pluto directory: ${POLYGEIST_PLUTO_DIR}")

# configure the cmake variables for pluto
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddPluto.cmake")
include_directories("${PLUTO_INCLUDE_DIR}")
message(STATUS "Will build and use Polymer-bundled Pluto")


# configure the cmake variables for bullseye
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddCPLEX.cmake")
include_directories("${CPLEX_INCLUDE_DIR}")
message(STATUS "Will build and use CPLEX")

# configure the cmake variables for bullseye
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddBullseye.cmake")
include_directories("${BULLSEYE_INCLUDE_DIR}")
message(STATUS "Will build and use Polymer-bundled Bullseye")

set(Boost_MIN_VERSION "1.60.0")

# Find the Boost library, specifically looking for the program_options component
find_package(Boost ${Boost_MIN_VERSION} REQUIRED COMPONENTS program_options)

# Include Boost headers
include_directories(${Boost_INCLUDE_DIRS})

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenScop.cmake")
include_directories("${OSL_INCLUDE_DIR}")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CLooG.cmake")
include_directories("${CLOOG_INCLUDE_DIR}")


# ------------------------------------------------- This project
include_directories("${POLYMER_MAIN_INCLUDE_DIR}")
include_directories("${POLYMER_INCLUDE_DIR}")

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

add_subdirectory(include/polymer)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/polymer
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    )
endif()
