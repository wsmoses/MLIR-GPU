set(GLPK_PREFIX "" CACHE PATH "path ")

find_path(GLPK_INCLUDE_DIR glpk.h 
    PATHS ${GLPK_PREFIX}/include /usr/include /usr/local/include )

find_library(GLPK_LIBRARY NAMES libglpk glpk
    PATHS ${GLPK_PREFIX}/lib /usr/lib/ /usr/local/lib/ )

if(GLPK_INCLUDE_DIR AND GLPK_LIBRARY)
    get_filename_component(GLPK_LIBRARY_DIR ${GLPK_LIBRARY} PATH)
    set(GLPK_FOUND TRUE)
endif()

if(GLPK_FOUND)
   if(NOT GLPK_FIND_QUIETLY)
      MESSAGE(STATUS "Found GLPK: ${GLPK_LIBRARY}")
   endif()
elseif(GLPK_FOUND)
   if(GLPK_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find GLPK")
   endif()
endif()
