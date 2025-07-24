cmake_minimum_required(VERSION 3.15)

execute_process(
    COMMAND python -m pylmcf --include
    OUTPUT_VARIABLE PYLMCF_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


project(wnetalign)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(wnetalign_cpp
    NB_STATIC
    src/cpp/wnetalign.cpp)

target_include_directories(wnetalign_cpp PRIVATE ${PYLMCF_INCLUDE_DIRS})

install(TARGETS wnetalign_cpp LIBRARY DESTINATION wnetalign)
