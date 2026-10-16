#!/usr/bin/env bash
# Build and run the standalone C++ suites listed in tests_cpp/suites.txt with
# one toolchain/flag set:
#
#   CXX=clang++ CXXFLAGS="-O1 -g -fsanitize=address,undefined" tests_cpp/run_all.sh
#
# Environment: CXX, CXXFLAGS, LDFLAGS, JOBS, OUT, TIMEOUT, RUN_PREFIX and
# SUITE_LIGHT as documented in .github/scripts/run_cpp_suites.sh. The wnet and
# pylmcf headers come from WNET_INCLUDE / PYLMCF_INCLUDE, else from the
# installed packages.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
WNET_INCLUDE=${WNET_INCLUDE:-$(python3 -m wnet --include)}
PYLMCF_INCLUDE=${PYLMCF_INCLUDE:-$(python3 -m pylmcf --include)}
export CPP_INCLUDES="-I$root/src/wnetalign/cpp -I$WNET_INCLUDE -I$WNET_INCLUDE/wnet -I$PYLMCF_INCLUDE ${CPP_INCLUDES:-}"
exec "$root/.github/scripts/run_cpp_suites.sh" "$here/suites.txt"
