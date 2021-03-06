
include_directories(.)


set(QUDA_TEST_COMMON gtest-all.cc test_util.cpp misc.cpp)

cuda_add_library(quda_test STATIC ${QUDA_TEST_COMMON})

set(TEST_LIBS quda quda_test ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS})

if(QUDA_QIO)
  LIST(APPEND TEST_LIBS ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QMP)
  LIST(APPEND TEST_LIBS ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  LIST(APPEND TEST_LIBS ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  find_library(MAGMA ${MAGMA_LIBRARIES} PATH ${MAGMA_LIBRARY_DIRS})
  LIST(APPEND TEST_LIBS ${MAGMA})
endif()


if(${QUDA_DIRAC_WILSON} OR ${QUDA_DIRAC_DOMAIN_WALL})
  cuda_add_executable(dslash_test dslash_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp ${QIO_UTIL})
  target_link_libraries(dslash_test ${TEST_LIBS} )

  cuda_add_executable(invert_test invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp ${QIO_UTIL})
  target_link_libraries(invert_test ${TEST_LIBS})
endif()

cuda_add_executable(deflation_test deflation_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp blas_reference.cpp ${QIO_UTIL})
target_link_libraries(deflation_test ${TEST_LIBS})

if(${QUDA_DIRAC_STAGGERED})
  cuda_add_executable(staggered_dslash_test staggered_dslash_test.cpp   staggered_dslash_reference.cpp)
  target_link_libraries(staggered_dslash_test ${TEST_LIBS})

  cuda_add_executable(staggered_invert_test staggered_invert_test.cpp  staggered_dslash_reference.cpp  blas_reference.cpp)
  target_link_libraries(staggered_invert_test ${TEST_LIBS})
endif()

cuda_add_executable(su3_test su3_test.cpp ${QIO_UTIL})
target_link_libraries(su3_test ${TEST_LIBS})

cuda_add_executable(pack_test pack_test.cpp)
target_link_libraries(pack_test ${TEST_LIBS})

cuda_add_executable(blas_test blas_test.cu)
target_link_libraries(blas_test ${TEST_LIBS})

if(${QUDA_LINK_ASQTAD} OR ${QUDA_LINK_HISQ})
  cuda_add_executable(llfat_test llfat_test.cpp llfat_reference.cpp)
  target_link_libraries(llfat_test ${TEST_LIBS})
endif()

if(${QUDA_LINK_HISQ})
  cuda_add_executable(unitarize_link_test unitarize_link_test.cpp)
  target_link_libraries(unitarize_link_test ${TEST_LIBS})
endif()

if(${QUDA_FORCE_GAUGE})
  cuda_add_executable(gauge_force_test gauge_force_test.cpp gauge_force_reference.cpp)
  target_link_libraries(gauge_force_test ${TEST_LIBS})
endif()

if(${QUDA_FORCE_ASQTAD})
  cuda_add_executable(fermion_force_test fermion_force_test.cpp fermion_force_reference.cpp)
  target_link_libraries(fermion_force_test ${TEST_LIBS})
endif()

if(${QUDA_GAUGE_ALG})
  cuda_add_executable(gauge_alg_test gauge_alg_test.cpp)
  target_link_libraries(gauge_alg_test ${TEST_LIBS})
endif()

if(${QUDA_FORCE_HISQ})
  cuda_add_executable(hisq_paths_force_test hisq_paths_force_test.cpp hisq_force_reference.cpp hisq_force_reference2.cpp fermion_force_reference.cpp   )
  target_link_libraries(hisq_paths_force_test ${TEST_LIBS})

  cuda_add_executable(hisq_unitarize_force_test hisq_unitarize_force_test.cpp hisq_force_reference.cpp )
  target_link_libraries(hisq_unitarize_force_test ${TEST_LIBS})
endif()
