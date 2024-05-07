cmake_minimum_required(VERSION 3.20)

project(py39_to_MLIR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ statndard to conform to")

#add_subdirectory(include)

set(MLIR_DIR /usr/lib/llvm-18/lib/cmake/mlir/)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Got MLIRConfig.cmake from: ${MLIR_DIR}")
message(STATUS "Got LLVMConfig.cmake from: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(TableGen) # ???
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions) # LLVM command line options

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
    ${dialect_libs}
    #MLIRExecutionEngine
    )

add_llvm_executable(py39_to_MLIR main.cpp)

llvm_update_compile_flags(py39_to_MLIR)
target_link_libraries(py39_to_MLIR PRIVATE ${LIBS})

mlir_check_all_link_libraries(py39_to_MLIR)
