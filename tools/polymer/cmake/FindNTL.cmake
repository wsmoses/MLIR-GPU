set(NTL_PREFIX "" CACHE PATH "path ")

find_path(NTL_INCLUDE_DIR 
    PATHS ${NTL_PREFIX}/include/NTL /usr/include/NTL /usr/local/include/NTL )

find_library(NTL_LIBRARY NAMES libntl ntl
    PATHS ${NTL_PREFIX}/lib /usr/lib/x86_64-linux-gnu/ /usr/local/lib)

if(NTL_INCLUDE_DIR AND NTL_LIBRARY)
    get_filename_component(NTL_LIBRARY_DIR ${NTL_LIBRARY} PATH)
    set(NTL_FOUND TRUE)
endif()

if(NTL_FOUND)
   if(NOT NTL_FIND_QUIETLY)
      MESSAGE(STATUS "Found NTL: ${NTL_LIBRARY}")
   endif()
elseif(NTL_FOUND)
   if(NTL_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find NTL")
   endif()
endif()
