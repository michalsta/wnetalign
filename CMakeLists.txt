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

# Which platforms have no backend to reach? Ask _wnetalign_metadata -- the very
# predicates the metadata provider uses to decide whether to declare
# nanobind-backend -- rather than reimplementing the tests here, so the build
# mode and the declared dependency can never disagree about the platform. A
# failure is fatal on purpose: guessing would mean silently building a mode that
# cannot be installed, and the module ships in the sdist right beside this file,
# so it being unimportable means the build is already broken.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sys;sys.path.insert(0,sys.argv[1]);import _wnetalign_metadata as m;print(int(m.is_musl()),int(m.is_win32()))"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE WNETALIGN_PLATFORM_RESULT
    OUTPUT_VARIABLE WNETALIGN_PLATFORM_PROBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE WNETALIGN_PLATFORM_ERROR)
if (NOT WNETALIGN_PLATFORM_RESULT EQUAL 0)
    message(FATAL_ERROR
        "wnetalign: could not determine the target platform via _wnetalign_metadata "
        "(is_musl / is_win32): ${WNETALIGN_PLATFORM_ERROR}")
endif()
separate_arguments(WNETALIGN_PLATFORM_PROBE)
list(GET WNETALIGN_PLATFORM_PROBE 0 WNETALIGN_IS_MUSL)
list(GET WNETALIGN_PLATFORM_PROBE 1 WNETALIGN_IS_WIN32)

if (WNET_NB_LINKED)
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (WNET_NB_LINKED=ON)")
elseif (NOT Python_INTERPRETER_ID STREQUAL "Python")
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (${Python_INTERPRETER_ID} is not CPython)")
elseif (WNETALIGN_IS_MUSL)
    # nanobind-backend ships manylinux, macOS and Windows wheels only -- no
    # musllinux wheel, and no sdist to build one from -- because auditwheel
    # bundles a private libstdc++ while split mode needs the C++ runtime shared
    # for exceptions to propagate. So split mode is unsatisfiable on musl by any
    # route, and this branch must be tested before the free-threaded ones: a
    # free-threaded musl interpreter has no backend to reach either.
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (musl libc: nanobind-backend publishes no musllinux wheel)")
elseif (WNETALIGN_IS_WIN32)
    # Same hole, different platform: nanobind-backend ships win_amd64 and
    # win_arm64 and nothing 32-bit, at any Python version. Split mode here
    # produces a wheel that builds cleanly and then cannot be installed on
    # the platform it was built for, because it depends on a distribution
    # that does not exist there -- pylmcf 1.2.0 shipped exactly that.
    # 32-bit Windows gets no wheel from build_wheels.yml (CIBW_SKIP), so
    # this branch is what makes the sdist install path work there.
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (32-bit Windows: nanobind-backend publishes no win32 wheel)")
elseif (NB_ABI MATCHES "[0-9]t" AND Python_VERSION VERSION_LESS 3.15)
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (free-threaded Python ${Python_VERSION} predates abi3t)")
elseif (NB_ABI MATCHES "[0-9]t")
    # Free-threaded 3.15+. Split mode here is abi3t-only -- the classic Stable
    # ABI does not exist for a free-threaded interpreter -- and nanobind makes
    # FREE_THREADED mandatory for it, refusing to configure otherwise. That
    # option declares Py_MOD_GIL_NOT_USED, so the module must not depend on the
    # GIL for its own internal state.
    # Verified for wnetalign_cpp: no file-scope mutable state at all beyond
    # nanobind's own module def. Per-aligner state only.
    #
    # This branch must also stay in step with pylmcf and the rest of the stack:
    # mode is what decides whether these extensions can see each other's
    # registered types, so one package falling back here while another goes
    # split would break type sharing on that interpreter. is_nanobind_split()
    # and the import-time check in __init__.py are the guard on that.
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend FREE_THREADED)
    set(NB_LINKED FALSE)
    set(NB_MODE_DESC "split, free-threaded abi3t (backend module: nanobind_backend)")
else()
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend)
    set(NB_LINKED FALSE)
    set(NB_MODE_DESC "split (backend module: nanobind_backend)")
endif()
message(STATUS "wnetalign: nanobind build mode: ${NB_MODE_DESC}")

# A linked nanobind cannot target the classic 3.10/3.11 stable ABI -- that is
# precisely what split mode is for -- and nanobind turns the combination into a
# hard configure error. Every linked path above can walk into it, not just
# WNET_NB_LINKED: musl is the one that bites in practice, because there
# wheel.py-api is honoured. (PyPy and free-threaded builds escape only because
# scikit-build-core discards the classic abi3 request for them anyway.) Drop the
# request rather than making the platform pass a second incantation.
if (NB_LINKED AND DEFINED SKBUILD_SABI_VERSION
    AND NOT SKBUILD_SABI_VERSION STREQUAL ""
    AND SKBUILD_SABI_VERSION VERSION_LESS "3.12")
    message(WARNING
        "wnetalign: dropping the wheel.py-api=${SKBUILD_SABI_VERSION} stable-ABI "
        "request -- this is a ${NB_MODE_DESC} build, which is specific to "
        "Python ${Python_VERSION}. scikit-build-core still tags the wheel abi3, "
        "so the wheel produced here is NOT portable across Python versions: "
        "install it and discard it, never publish or reuse it.")
    set(SKBUILD_SABI_VERSION "")
endif()


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

