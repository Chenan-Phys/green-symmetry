#
# Library for self-consistency procedure
#

project(symmetry-lib)

add_library(symmetry
        except.cpp
        bose_kmesh.cpp
        fermi_kmesh.cpp
        symmetry.cpp
        thc_host_fft.cpp
)

find_package(Eigen3 REQUIRED)
find_package(BLAS)
if (BLAS_FOUND)
    if (BLAS_LIBRARIES MATCHES ".*Accelerate")
        message(INFO " Using Apple BLAS")
    endif ()
    if (BLAS_LIBRARIES MATCHES ".*openblas")
        message(INFO " Using OpenBLAS")
        add_compile_definitions(USE_GEMM3M)
    endif ()
    add_compile_definitions(EIGEN_USE_BLAS)
endif ()



target_link_libraries(symmetry PUBLIC GREEN::H5PP GREEN::NDARRAY GREEN::PARAMS Eigen3::Eigen)
if(BLAS_FOUND)
    target_link_libraries(symmetry PUBLIC BLAS::BLAS)
endif(BLAS_FOUND)
option(GREEN_THC_USE_FFTW "Use batched FFTW for native CPU THC momentum transforms when available" ON)
if(GREEN_THC_USE_FFTW)
    find_path(GREEN_THC_FFTW_INCLUDE_DIR fftw3.h)
    find_library(GREEN_THC_FFTW_LIBRARY NAMES fftw3)
    if(GREEN_THC_FFTW_INCLUDE_DIR AND GREEN_THC_FFTW_LIBRARY)
        target_include_directories(symmetry PUBLIC ${GREEN_THC_FFTW_INCLUDE_DIR})
        target_link_libraries(symmetry PUBLIC ${GREEN_THC_FFTW_LIBRARY})
        target_compile_definitions(symmetry PRIVATE GREEN_THC_FFTW GREEN_THC_FFTW_HEADER="${GREEN_THC_FFTW_INCLUDE_DIR}/fftw3.h")
    endif()
endif()
target_include_directories(symmetry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
