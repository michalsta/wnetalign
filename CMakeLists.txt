cmake_minimum_required(VERSION 3.15)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS deployment version")
endif()

project(wnetalign LANGUAGES CXX)
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)

# --- nanobind build mode ------------------------------------------------------
# Split mode (nanobind >= 3) leaves no nanobind library code in the extension and
# resolves a shared backend module at import time. Every extension sharing that
# backend has full mutual visibility of the others' registered types -- which is
# what lets wnetalign nb::cast a class registered inside wnet_cpp.so without the
# two being pinned to a common nanobind ABI version.
#
# Split mode requires CPython, and on a free-threaded interpreter it requires the
# 'abi3t' stable ABI of Python 3.15+. Fall back to the classic linked build
# everywhere else, and whenever WNET_NB_LINKED is set: sanitizer builds need that,
# because the published backend is an uninstrumented libstdc++ binary and the
# frontend/backend pair must agree on compiler, C++ runtime and debug mode.
option(WNET_NB_LINKED "Force a linked (NB_STATIC) nanobind build instead of split mode" OFF)

if (WNET_NB_LINKED)
    set(NB_MODE NB_STATIC)
    set(NB_MODE_DESC "linked (WNET_NB_LINKED=ON)")
    # A linked nanobind cannot target the 3.10/3.11 stable ABI -- that is what
    # split mode is for -- and nanobind turns the combination into a hard
    # configure error. WNET_NB_LINKED exists for local sanitizer and debug
    # builds, not for producing distributable wheels, so drop the abi3 request
    # rather than making the caller pass a second incantation.
    if (DEFINED SKBUILD_SABI_VERSION AND NOT SKBUILD_SABI_VERSION STREQUAL ""
        AND SKBUILD_SABI_VERSION VERSION_LESS "3.12")
        message(WARNING
            "WNET_NB_LINKED=ON overrides wheel.py-api=${SKBUILD_SABI_VERSION}: "
            "a linked build is specific to Python ${Python_VERSION}. Any wheel "
            "produced from this build is NOT abi3 and must not be published.")
        set(SKBUILD_SABI_VERSION "")
    endif()
elseif (NOT Python_INTERPRETER_ID STREQUAL "Python")
    set(NB_MODE NB_STATIC)
    set(NB_MODE_DESC "linked (${Python_INTERPRETER_ID} is not CPython)")
elseif (NB_ABI MATCHES "[0-9]t" AND Python_VERSION VERSION_LESS 3.15)
    set(NB_MODE NB_STATIC)
    set(NB_MODE_DESC "linked (free-threaded Python ${Python_VERSION} predates abi3t)")
else()
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend)
    set(NB_MODE_DESC "split (backend module: nanobind_backend)")
endif()
message(STATUS "wnetalign: nanobind build mode: ${NB_MODE_DESC}")


execute_process(
   COMMAND ${Python_EXECUTABLE} -m pylmcf --include
   OUTPUT_VARIABLE PYLMCF_INCLUDE_DIRS
   OUTPUT_STRIP_TRAILING_WHITESPACE
   RESULT_VARIABLE PYLMCF_INCLUDE_RESULT
)
if(NOT PYLMCF_INCLUDE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get include path from pylmcf (is it installed for ${Python_EXECUTABLE}?)")
endif()

execute_process(
   COMMAND ${Python_EXECUTABLE} -m wnet --include
   OUTPUT_VARIABLE WNET_INCLUDE_DIRS
   OUTPUT_STRIP_TRAILING_WHITESPACE
   RESULT_VARIABLE WNET_INCLUDE_RESULT
)
if(NOT WNET_INCLUDE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get include path from wnet (is it installed for ${Python_EXECUTABLE}?)")
endif()


set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(wnetalign_cpp
    ${NB_MODE} NOMINSIZE
    src/wnetalign/cpp/wnetalign/wnetalign.cpp)

target_include_directories(wnetalign_cpp PRIVATE ${PYLMCF_INCLUDE_DIRS} ${WNET_INCLUDE_DIRS})
target_compile_definitions(wnetalign_cpp PRIVATE INCLUDE_NANOBIND_STUFF)

install(TARGETS wnetalign_cpp LIBRARY DESTINATION wnetalign)

# Pure C++ test executable. tests_cpp/ is deliberately excluded from the sdist
# (see sdist.exclude in pyproject.toml), so this target only exists in a git
# checkout -- guard it, or every sdist build dies on a missing source file.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests_cpp/test_lcms_align.cpp)
    add_executable(test_lcms_align tests_cpp/test_lcms_align.cpp)
    target_include_directories(test_lcms_align PRIVATE
        src/wnetalign/cpp
        ${PYLMCF_INCLUDE_DIRS}
        ${WNET_INCLUDE_DIRS}
    )
    target_compile_features(test_lcms_align PRIVATE cxx_std_20)
endif()

