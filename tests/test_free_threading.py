"""The free-threaded build must actually stay free-threaded.

On CPython 3.15t the extension is built in split mode with nanobind's
FREE_THREADED option, which declares ``Py_MOD_GIL_NOT_USED``. If that option is
ever dropped, or the build quietly falls back to a linked mode, importing the
extension re-enables the GIL and the wheel is free-threaded in name only --
nothing else in the suite would notice, because every test still passes.

The whole module is skipped unless the GIL is genuinely off *after* importing
the extension. That is the point: on 3.14t the linked fallback is expected to
turn the GIL back on, so this correctly stays quiet there rather than failing.

wnetalign is the module with the most to lose here: it nb::casts a class
registered inside wnet_cpp, so a solve here crosses two extension modules and
the shared nanobind backend on every call.
"""

import sys
import threading

import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnetalign import WNetAligner
from wnetalign.spectrum import Spectrum

pytestmark = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="needs a free-threaded interpreter with the GIL still disabled after import",
)

N_POINTS = 20
N_THREADS = 8
N_ROUNDS = 25


def _spectra():
    # Seeded, so every call below has the same right answer. Two 2-D spectra
    # close enough that most of the mass matches rather than going to trash.
    rng = np.random.default_rng(0)
    pos1 = rng.uniform(0.0, 10.0, (2, N_POINTS))
    pos2 = pos1 + rng.uniform(-0.3, 0.3, (2, N_POINTS))
    int1 = rng.integers(1, 6, N_POINTS).astype(float)
    int2 = rng.integers(1, 6, N_POINTS).astype(float)
    return (pos1, int1), (pos2, int2)


def _solve_once(spectra):
    # A fresh aligner per call: the claim under test is that the module holds no
    # *global* state, not that one aligner may be driven from two threads.
    (p1, i1), (p2, i2) = spectra
    aligner = WNetAligner(
        Spectrum(p1.copy(), i1.copy()),
        [Spectrum(p2.copy(), i2.copy())],
        DistanceMetric.L2,
        1000000,
        1000,
    )
    aligner.set_point([1.0])
    return aligner.total_cost()


def test_gil_stays_disabled_after_import():
    assert not sys._is_gil_enabled()


def test_concurrent_alignments_agree_with_serial():
    spectra = _spectra()
    expected = _solve_once(spectra)

    results = []
    errors = []

    def worker():
        try:
            for _ in range(N_ROUNDS):
                results.append(_solve_once(spectra))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert len(results) == N_THREADS * N_ROUNDS
    assert all(r == expected for r in results)
    # If the GIL had been re-enabled behind our back, the run above proves nothing.
    assert not sys._is_gil_enabled()
