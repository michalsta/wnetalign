#! /usr/bin/env python
# -*- coding: utf-8 -*-

from wnet import wnet_cpp  # noqa: F401 — must precede wnetalign_cpp so solver config types are registered
from . import wnetalign_cpp
from .aligner import WNetAligner
from .spectrum import Spectrum, Spectrum_1D


def is_nanobind_split() -> bool:
    """True when wnetalign_cpp was built in nanobind split mode. See pylmcf.nanobind_mode."""
    from pylmcf.nanobind_mode import extension_is_split

    return extension_is_split(wnetalign_cpp)


def _check_nanobind_modes() -> None:
    # This one is load-bearing rather than precautionary: wnetalign_cpp casts a
    # class registered inside wnet_cpp (nb::cast<Spectrum<DIM>*>), so a mode
    # mismatch with wnet breaks the aligner outright.
    import pylmcf.pylmcf_cpp
    from pylmcf.nanobind_mode import check_consistent

    check_consistent(
        [
            ("pylmcf", pylmcf.pylmcf_cpp),
            ("wnet", wnet_cpp),
            ("wnetalign", wnetalign_cpp),
        ]
    )


_check_nanobind_modes()


def py_hello():
    print("Hello, World from WNetAlign (Python)!")
