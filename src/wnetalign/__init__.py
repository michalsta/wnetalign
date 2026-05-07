#! /usr/bin/env python
# -*- coding: utf-8 -*-

from wnet import wnet_cpp  # noqa: F401 — must precede wnetalign_cpp so SolverMethod is registered
from . import wnetalign_cpp
from .aligner import WNetAligner
from .spectrum import Spectrum, Spectrum_1D

def py_hello():
    print("Hello, World from WNetAlign (Python)!")
