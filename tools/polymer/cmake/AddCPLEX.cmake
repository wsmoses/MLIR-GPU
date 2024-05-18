set(CPLEX_INCLUDE_DIR 
   "${CPLEX_HOME_DIR}/cplex/include/ilcplex/"
   "${CPLEX_HOME_DIR}/concert/include/ilcplex/"
)

set(CPLEX_LIBRARY_DIR
   "${CPLEX_HOME_DIR}/cplex/lib/x86-64_linux/static_pic/"
   "${CPLEX_HOME_DIR}/concert/lib/x86-64_linux/static_pic/"
)

find_library(CPLEX_LIBRARY_TEST NAMES cplex concert ilocplex libcplex libconcert libilocplex
    PATHS ${CPLEX_HOME_DIR}/cplex/lib/x86-64_linux/static_pic/ ${CPLEX_HOME_DIR}/concert/lib/x86-64_linux/static_pic/)

add_library(libcplex STATIC IMPORTED)
set_target_properties(libcplex PROPERTIES IMPORTED_LOCATION "${CPLEX_HOME_DIR}/cplex/lib/x86-64_linux/static_pic/libcplex.a")
add_library(libconcert STATIC IMPORTED)
set_target_properties(libconcert PROPERTIES IMPORTED_LOCATION "${CPLEX_HOME_DIR}/concert/lib/x86-64_linux/static_pic/libconcert.a")
add_library(libilocplex STATIC IMPORTED)
set_target_properties(libilocplex PROPERTIES IMPORTED_LOCATION "${CPLEX_HOME_DIR}/cplex/lib/x86-64_linux/static_pic/libilocplex.a")


add_dependencies(libcplex bullseye)
add_dependencies(libconcert bullseye)
add_dependencies(libilocplex bullseye)
