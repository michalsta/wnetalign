#!/bin/bash

pip uninstall -y wnetalign
VERBOSE=1 pip install -v --no-build-isolation -e .
