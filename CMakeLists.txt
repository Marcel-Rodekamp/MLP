# C++ modules for MLP
cmake_minimum_required(VERSION 3.9.6)

project(MLP_CVN CXX)

find_package(Torch REQUIRED)

set(LIBNAME "complexifyTorch_cpp")

set(SOURCES
    src/ComplexifyTorch/cmatmul.cpp
    src/ComplexifyTorch/cadd.cpp
)

set(HEADERS
    src/ComplexifyTorch/linalg.hpp
)

add_library(${LIBNAME} SHARED ${HEADERS} ${SOURCES} )

target_compile_features(${LIBNAME} PRIVATE
    cxx_std_14
)

# Link against LibTorch
target_link_libraries(${LIBNAME} "${TORCH_LIBRARIES}")
