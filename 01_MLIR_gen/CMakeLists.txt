cmake_minimum_required(VERSION 3.20)

project(py39compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ statndard to conform to")
set(CMAKE_C_STANDARD 17 CACHE STRING "C statndard to conform to")

if(MSVC)
    set(LLVM_DIR "C:\\llvm-project\\build_vs_2022\\lib\\cmake\\llvm")
    set(MLIR_DIR "C:\\llvm-project\\build_vs_2022\\lib\\cmake\\mlir")
else()
    set(LLVM_DIR /usr/lib/llvm-19/lib/cmake/llvm)
    set(MLIR_DIR /usr/lib/llvm-19/lib/cmake/mlir)
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Got MLIRConfig.cmake from: ${MLIR_DIR}")
message(STATUS "Got LLVMConfig.cmake from: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
    ${dialect_libs}
    )

add_llvm_executable(py39compiler main.cpp)

llvm_update_compile_flags(py39compiler)
target_link_libraries(py39compiler PRIVATE ${LIBS})

mlir_check_all_link_libraries(py39compiler)
