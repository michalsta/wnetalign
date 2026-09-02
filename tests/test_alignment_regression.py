"""
Regression tests for alignment quantization and scaling.

These guard the properties that the aligner's *scaling* has to preserve, which
the rest of the suite cannot see: analytic costs on toy problems, ground-truth
recall on synthetic shifted spectra, survival of faint peaks through intensity
quantization, and agreement between the solver's own reported cost and an exact
double-precision recomputation of the plan it returned.

The committed baseline in ``baselines/alignment_baseline.json`` pins the real
LC-MS results.  Re-capture it with ``python tests/capture_baseline.py`` only
when a change to the alignment result is intended, and say so in the commit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import align_harness as H

REPO = Path(__file__).resolve().parent.parent
DATADIR = REPO / "tutorials" / "lcms" / "data"
BASELINE = Path(__file__).resolve().parent / "baselines" / "alignment_baseline.json"

#: Cases where every empirical unit finds a match, so the solver's reported
#: total cost must equal the exactly-recomputed transport cost of its own plan.
FULLY_MATCHED = [
    "dynrange_1e3",
    "dynrange_1e6",
    "dynrange_1e9",
    "dynrange_norm_1e6",
    "dynrange_norm_1e9",
    "fine_separation",
    "asymmetric_trash",
]


@pytest.fixture(scope="module")
def toys():
    return H.toy_cases()


@pytest.fixture(scope="module")
def synthetics():
    return {name: H.run_case(case) for name, case in H.synthetic_cases().items()}


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.is_file():
        pytest.skip(f"no committed baseline at {BASELINE}")
    return json.loads(BASELINE.read_text())


# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(H.toy_cases()))
def test_toy_exact_costs(toys, name):
    """Analytically known costs, computed by hand from the trash model."""
    case = toys[name]
    summary = H.run_case(case)
    assert summary["total_cost"] == pytest.approx(case["exact_total_cost"], rel=1e-6)


@pytest.mark.parametrize("name", list(H.synthetic_cases()))
def test_ground_truth_recall(synthetics, name):
    """Every synthetic pair is a known rigid shift, so the identity matching is
    the unique optimum and the consensus must recover all of it."""
    summary = synthetics[name]
    assert summary["truth_recall"] == 1.0, (
        f"{name}: recovered {summary['truth_correct']} of the known pairs "
        f"(recall {summary['truth_recall']:.4f}); peaks are being lost before "
        f"the solver sees them, or matched wrongly."
    )
    assert summary["truth_precision"] == 1.0


@pytest.mark.parametrize("name", list(H.synthetic_cases()))
def test_no_peak_is_quantized_away(synthetics, name):
    """A peak with positive intensity must never truncate to zero supply.

    This is the failure the tied scale factor used to produce, and it is
    invisible from the outside: the vanished peaks are the faint ones, so the
    total mass barely moves and the reported cost looks entirely reasonable.
    """
    summary = synthetics[name]
    assert summary["emp_dropped_peaks"] == 0, (
        f"{name}: {summary['emp_dropped_peaks']} peaks quantized to zero supply "
        f"at intensity_scale={summary['intensity_scale']:.4g} — they are gone "
        f"from the alignment, and only {summary['emp_dropped_mass_frac']:.2e} "
        f"of the total mass went with them, so no mass check would catch it."
    )


@pytest.mark.parametrize("name", FULLY_MATCHED)
def test_reported_cost_matches_recomputed_plan(synthetics, name):
    """The solver's scaled arithmetic must agree with an exact recomputation.

    ``transport_cost`` is ``sum(distance * flow)`` evaluated in double
    precision against the original positions.  When all mass is matched it is
    the whole objective, so a discrepancy means the cost grid is too coarse to
    represent the distances the solver is actually charging for.
    """
    summary = synthetics[name]
    assert summary["total_cost"] == pytest.approx(summary["transport_cost"], rel=1e-6)


def test_scales_are_chosen_independently():
    """The intensity and cost grids must be separate numbers.

    They used to be one tied factor.  Sharing a single value is precisely the
    defect this guards against, so assert they were sized on their own terms.
    """
    emp, theo, _ = H.make_shifted_pair(
        n=100, dyn_range=1e6, shift=1.0, seed=7, normalize=True
    )
    aligner = H.WNetAligner(
        emp, [theo], H.DistanceMetric.L1, 5.0, trash_cost=50.0
    )
    aligner.set_point([1.0])
    assert aligner.cost_scale > 1, (
        "cost_scale == 1 means p == 1 legacy truncation: every real distance "
        "is being rounded to a whole number."
    )
    assert aligner.intensity_scale > 0
    assert aligner.intensity_scale != float(aligner.cost_scale)
    # The back-compatible alias must track the factor flows are denominated in,
    # or flows() unscales with the wrong number.
    assert aligner.scale_factor == aligner.intensity_scale


def test_flows_are_returned_in_real_units():
    """flows() must divide by the intensity scale, not some other factor."""
    emp, theo, _ = H.make_shifted_pair(
        n=50, dyn_range=1e4, shift=0.5, seed=11, normalize=True
    )
    aligner = H.WNetAligner(emp, [theo], H.DistanceMetric.L1, 5.0, trash_cost=50.0)
    aligner.set_point([1.0])
    moved = float(np.asarray(aligner.flows()[0].flow, dtype=float).sum())
    total = float(np.asarray(emp.intensities, dtype=float).sum())
    # Every peak has a partner within the cap, so all of the mass should move.
    assert moved == pytest.approx(total, rel=1e-6)


# --------------------------------------------------------------------------
# Real published data
#
# Everything below is marked ``realdata``: it reads the committed LC-MS and NMR
# datasets and pins exact numbers.  CI runs these on one canonical job and
# deselects them elsewhere with ``-m "not realdata"`` — see conftest.py for why.
# The fixtures are module-scoped so each dataset is read and solved once rather
# than once per assertion.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lcms_results():
    if not DATADIR.is_dir():
        pytest.skip("LC-MS tutorial data not present")
    return {name: H.run_case(case) for name, case in H.lcms_cases(DATADIR).items()}


@pytest.fixture(scope="module")
def nmr_results():
    cases = H.nmr_cases(REPO)
    if not cases:
        pytest.skip("publication/nmr data or loader not present")
    return {name: H.run_case(case) for name, case in cases.items()}


def _assert_matches_baseline(name, got, expected):
    for key, rel in (
        ("total_cost", 1e-6),
        ("transport_cost", 1e-6),
        ("matched_mass", 1e-6),
    ):
        assert got[key] == pytest.approx(expected[key], rel=rel), (
            f"{name}.{key}: {got[key]!r} != baseline {expected[key]!r}"
        )
    assert got["emp_dropped_peaks"] == expected["emp_dropped_peaks"]
    # The consensus pairing is the scientific output; a change in which peaks
    # were matched is worth failing over even when the aggregate cost agrees.
    assert got["consensus_digest"] == expected["consensus_digest"], (
        f"{name}: the consensus matching changed "
        f"({got['n_consensus']} pairs vs baseline {expected['n_consensus']})"
    )


@pytest.mark.realdata
@pytest.mark.parametrize("name", ["lcms_replicates", "lcms_cross_batch", "lcms_full"])
def test_lcms_matches_baseline(baseline, lcms_results, name):
    """Pin the real LC-MS alignments against the committed baseline."""
    if name not in baseline:
        pytest.skip(f"{name} not in the committed baseline")
    _assert_matches_baseline(name, lcms_results[name], baseline[name])


@pytest.mark.realdata
def test_nmr_matches_baseline(baseline, nmr_results):
    """Pin the paper's own NMR alignments (GB1 2D series, 2LX7 4D)."""
    checked = 0
    for name, got in nmr_results.items():
        if name not in baseline:
            continue
        _assert_matches_baseline(name, got, baseline[name])
        checked += 1
    if checked == 0:
        pytest.skip("no NMR cases in the committed baseline")


@pytest.mark.realdata
def test_lcms_faint_peaks_are_resolved(lcms_results):
    """The full LC-MS files span five orders of magnitude in intensity after
    normalization; the faint end must still land on a usable integer supply."""
    got = lcms_results["lcms_full"]
    assert got["emp_dropped_peaks"] == 0
    assert got["emp_min_supply"] >= 100, (
        f"faintest peak quantized to {got['emp_min_supply']} units — a "
        f"relative error of {1.0 / max(got['emp_min_supply'], 1):.0%} on a real "
        f"peak from the published dataset."
    )


@pytest.mark.realdata
def test_nmr_faint_peaks_are_resolved(nmr_results):
    """Same contract on the paper's NMR data, which is normalized the same way."""
    for name, got in nmr_results.items():
        assert got["emp_dropped_peaks"] == 0, (
            f"{name}: {got['emp_dropped_peaks']} peaks quantized to zero supply"
        )
