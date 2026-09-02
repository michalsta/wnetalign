"""
Scaling regression harness for :class:`wnetalign.WNetAligner`.

Why this exists
---------------
The aligner quantizes two independent quantities onto integer grids before the
min-cost-flow solver sees them: **intensities** (peak masses become integer
supplies) and **distances** (real ground distances become integer edge costs).
Both quantizations are lossy, and both fail *silently*: a peak whose intensity
truncates to zero supply simply vanishes from the alignment, and distances that
collapse onto the same integer cost make the solver indifferent between matches
it should be able to rank.

Neither failure raises.  Neither changes the shape of the output.  A test that
only checks "did we get an answer" or "did we get enough rows" cannot see them.

So this harness judges an alignment by quantities that are computed in exact
double precision from the *original, unscaled* data, independent of whatever
scaling the aligner chose internally:

``transport_cost``
    ``sum(distance(emp_pos, theo_pos) * flow)`` recomputed from the returned
    flows against the original positions.  This is the true real-units cost of
    the transport plan the aligner actually produced — not the solver's own
    scaled arithmetic reporting on itself.

``matched_mass``
    Total flow the plan moved, in real intensity units.

Together these two make plans comparable across scaling changes: a plan that
moves at least as much mass for no more transport cost is not worse.  Comparing
``total_cost()`` alone cannot distinguish "cheaper because better" from
"cheaper because mass went missing".

``dropped_peaks`` / ``dropped_mass_frac``
    Peaks whose intensity truncates to a zero integer supply at the aligner's
    chosen intensity scale — i.e. peaks that were deleted by quantization
    before the solver ran.  This is the silent failure, measured directly.

``truth_recall``
    For synthetic cases built by shifting a spectrum by a known offset, the
    fraction of the known-correct pairs the consensus matching recovers.  This
    is ground truth, not self-consistency.

The harness deliberately drives only the public Python API, so it exercises the
wrapper's own unscaling (``flows()`` divides by the intensity scale) rather than
reaching past it into C++.  If the wrapper unscales with the wrong factor, the
recomputed transport cost diverges from ``total_cost()`` and the harness says so.
"""

from __future__ import annotations

import zlib

import numpy as np

from wnet.distances import DistanceMetric
from wnetalign import Spectrum, Spectrum_1D, WNetAligner


# --------------------------------------------------------------------------
# Exact (double-precision) reference computations on the ORIGINAL data
# --------------------------------------------------------------------------


def pairwise_distance(p: np.ndarray, q: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Row-wise distance between two (N, DIM) arrays of positions."""
    d = np.abs(p - q)
    if metric == DistanceMetric.L1:
        return d.sum(axis=1)
    if metric == DistanceMetric.L2:
        return np.sqrt((d**2).sum(axis=1))
    if metric == DistanceMetric.LINF:
        return d.max(axis=1)
    raise ValueError(f"unhandled metric {metric!r}")


def intensity_scale_of(aligner: WNetAligner) -> float:
    """The factor real intensities are multiplied by before truncation.

    Pre-migration the aligner had a single tied ``scale_factor`` serving both
    intensities and positions; post-migration the intensity scale is its own
    attribute.  Accept either so one baseline spans both.
    """
    return float(getattr(aligner, "intensity_scale", aligner.scale_factor))


def quantization_report(spectrum: Spectrum, sf_intensity: float) -> dict:
    """How much of a spectrum survives truncation onto the integer supply grid.

    Mirrors the network's own quantization: ``trunc(intensity * sf_intensity)``.
    A peak that lands on zero is gone — it cannot receive or send flow.
    """
    intens = np.asarray(spectrum.intensities, dtype=float)
    total = float(intens.sum())
    supplies = np.trunc(intens * sf_intensity)
    kept = float(supplies.sum()) / sf_intensity if sf_intensity > 0 else 0.0
    positive = intens > 0
    dropped = int(np.count_nonzero(positive & (supplies == 0)))
    # A peak that survives can still be badly resolved: at a supply of s the
    # relative quantization error is up to 1/s.  Below 100 units that is worse
    # than 1%, which for a faint peak is the difference between a match and a
    # coin flip.  Counting these catches degradation that `dropped` misses.
    coarse = int(np.count_nonzero(positive & (supplies > 0) & (supplies < 100)))
    return {
        "n_peaks": int(intens.size),
        "dropped_peaks": dropped,
        "dropped_peak_frac": dropped / max(1, int(np.count_nonzero(positive))),
        "dropped_mass_frac": (total - kept) / total if total > 0 else 0.0,
        "coarse_peaks": coarse,
        "min_supply": float(supplies[positive].min()) if positive.any() else 0.0,
    }


# --------------------------------------------------------------------------
# Running one alignment and summarizing it
# --------------------------------------------------------------------------


def summarize_alignment(
    aligner: WNetAligner,
    empirical: Spectrum,
    theoreticals: list[Spectrum],
    metric: DistanceMetric,
    truth: np.ndarray | None = None,
) -> dict:
    """Reduce a solved aligner to a compact, scale-independent summary.

    ``truth``, when given, is an array mapping empirical peak index -> the
    theoretical peak index it *should* match (-1 for peaks with no true
    partner), used to compute recall of the consensus matching.
    """
    emp_pos = np.asarray(empirical.positions, dtype=float).T  # (N, DIM)

    total_transport = 0.0
    total_mass = 0.0
    n_flow_edges = 0
    emp_touched: set[int] = set()

    flows = aligner.flows()
    for target_id, flow in enumerate(flows):
        emp_idx = np.asarray(flow.empirical_peak_idx, dtype=np.int64)
        theo_idx = np.asarray(flow.theoretical_peak_idx, dtype=np.int64)
        amounts = np.asarray(flow.flow, dtype=float)
        if emp_idx.size == 0:
            continue
        theo_pos = np.asarray(theoreticals[target_id].positions, dtype=float).T
        dists = pairwise_distance(emp_pos[emp_idx], theo_pos[theo_idx], metric)
        total_transport += float((dists * amounts).sum())
        total_mass += float(amounts.sum())
        n_flow_edges += int(emp_idx.size)
        emp_touched.update(emp_idx[amounts > 0].tolist())

    summary = {
        "total_cost": float(aligner.total_cost()),
        "transport_cost": total_transport,
        "matched_mass": total_mass,
        "n_flow_edges": n_flow_edges,
        "n_empirical_matched": len(emp_touched),
        "no_subgraphs": int(aligner.no_subgraphs()),
    }

    sf_int = intensity_scale_of(aligner)
    summary["intensity_scale"] = sf_int
    qr = quantization_report(empirical, sf_int)
    summary["emp_dropped_peaks"] = qr["dropped_peaks"]
    summary["emp_dropped_mass_frac"] = qr["dropped_mass_frac"]
    summary["emp_coarse_peaks"] = qr["coarse_peaks"]
    summary["emp_min_supply"] = qr["min_supply"]

    cons_emp, cons_theo = aligner.consensus(0)
    cons_emp = np.asarray(cons_emp, dtype=np.int64)
    cons_theo = np.asarray(cons_theo, dtype=np.int64)
    summary["n_consensus"] = int(cons_emp.size)
    # A stable fingerprint of the actual pairing, so a baseline diff reports
    # "the matching changed" even when the aggregate numbers happen to agree.
    order = np.argsort(cons_emp, kind="stable")
    pairs = np.stack([cons_emp[order], cons_theo[order]], axis=1)
    # zlib.crc32, not hash(): Python randomizes hash() of bytes per process, so
    # a baseline pinned on it would differ from one run to the next.
    summary["consensus_digest"] = zlib.crc32(np.ascontiguousarray(pairs).tobytes())

    if truth is not None:
        expected = truth[cons_emp]
        correct = int(np.count_nonzero(expected == cons_theo))
        n_true = int(np.count_nonzero(truth >= 0))
        summary["truth_correct"] = correct
        summary["truth_recall"] = correct / n_true if n_true else 0.0
        summary["truth_precision"] = correct / max(1, int(cons_emp.size))

    return summary


def run_case(case: dict) -> dict:
    """Build, solve and summarize one harness case."""
    empirical = case["empirical"]
    theoreticals = case["theoreticals"]
    kwargs = dict(case.get("kwargs", {}))
    aligner = WNetAligner(
        empirical,
        theoreticals,
        case["metric"],
        case["max_distance"],
        **kwargs,
    )
    aligner.set_point(case.get("point", [1.0] * len(theoreticals)))
    return summarize_alignment(
        aligner, empirical, theoreticals, case["metric"], case.get("truth")
    )


# --------------------------------------------------------------------------
# Synthetic case builders — ground truth by construction
# --------------------------------------------------------------------------


def make_shifted_pair(
    n: int,
    dyn_range: float,
    shift: float,
    seed: int,
    dim: int = 1,
    gap: float = 10.0,
    normalize: bool = False,
) -> tuple[Spectrum, Spectrum, np.ndarray]:
    """A spectrum and a copy of it rigidly shifted by ``shift``.

    Peaks are laid on a jittered grid with minimum spacing ``gap``, so with
    ``shift << gap`` the identity pairing is the unique optimal matching and
    the ground truth is known exactly.  ``dyn_range`` is the ratio between the
    largest and smallest intensity, log-uniformly spread — this is the axis
    that a too-coarse intensity scale destroys.
    """
    rng = np.random.default_rng(seed)
    base = np.arange(n, dtype=float) * gap
    jitter = rng.uniform(-gap * 0.2, gap * 0.2, size=n)
    coords = base + jitter
    if dim == 1:
        emp_pos = coords[np.newaxis, :]
        theo_pos = (coords + shift)[np.newaxis, :]
    else:
        extra = rng.uniform(0.0, gap * 0.1, size=(dim - 1, n))
        emp_pos = np.vstack([coords[np.newaxis, :], extra])
        theo_pos = np.vstack([(coords + shift)[np.newaxis, :], extra])
    exponents = rng.uniform(0.0, np.log10(dyn_range), size=n)
    intens = 10.0**exponents
    if normalize:
        # What the published pipelines actually do.  Normalizing to unit total
        # mass leaves the *ratios* untouched but drives the absolute intensity
        # of the faint tail down to ~1/sum, which is where a coarse integer
        # supply grid deletes it outright.
        intens = intens / intens.sum()
    truth = np.arange(n, dtype=np.int64)
    return (
        Spectrum(emp_pos, intens.copy()),
        Spectrum(theo_pos, intens.copy()),
        truth,
    )


def make_fine_separation_pair(
    n: int, separation: float, seed: int
) -> tuple[Spectrum, Spectrum, np.ndarray]:
    """Peaks separated by ``separation`` in position but huge in intensity.

    The distance differences that decide the matching are tiny relative to the
    intensity magnitudes, so this is the axis a too-coarse *cost* scale
    destroys — the mirror image of :func:`make_shifted_pair`.
    """
    rng = np.random.default_rng(seed)
    coords = np.arange(n, dtype=float) * separation * 10.0
    intens = rng.uniform(1e6, 1e7, size=n)
    emp_pos = coords[np.newaxis, :]
    theo_pos = (coords + separation)[np.newaxis, :]
    truth = np.arange(n, dtype=np.int64)
    return (
        Spectrum(emp_pos, intens.copy()),
        Spectrum(theo_pos, intens.copy()),
        truth,
    )


# --------------------------------------------------------------------------
# The case registry
# --------------------------------------------------------------------------


def toy_cases() -> dict[str, dict]:
    """Small hand-checkable alignments; costs are known analytically."""
    cases = {}

    cases["toy_split"] = {
        "empirical": Spectrum_1D([0.0], [10.0]),
        "theoreticals": [Spectrum_1D([1.0, 2.0], [5.0, 5.0])],
        "metric": DistanceMetric.L2,
        "max_distance": 100,
        "kwargs": {"trash_cost": 10, "scale_factor": 10000},
        "exact_total_cost": 15.0,
    }
    cases["toy_two_targets"] = {
        "empirical": Spectrum_1D([0.0], [10.0]),
        "theoreticals": [Spectrum_1D([1.0], [4.0]), Spectrum_1D([2.0], [6.0])],
        "metric": DistanceMetric.L2,
        "max_distance": 100,
        "kwargs": {"trash_cost": 10, "scale_factor": 10000},
        "exact_total_cost": 16.0,
    }
    cases["toy_out_of_range"] = {
        "empirical": Spectrum_1D([0.0], [10.0]),
        "theoreticals": [Spectrum_1D([1.0], [4.0]), Spectrum_1D([200.0], [6.0])],
        "metric": DistanceMetric.L2,
        "max_distance": 10,
        # s3 is isolated (beyond max_distance); annihilating trash is priced
        # network-wide, so the excess annihilates once: 4 + 10 * 6.
        "kwargs": {"trash_cost": 10, "scale_factor": 100},
        "exact_total_cost": 64.0,
    }
    cases["toy_2d"] = {
        "empirical": Spectrum(
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([1.0, 1.0, 1.0])
        ),
        "theoreticals": [
            Spectrum(
                np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]), np.array([1.0, 1.0, 1.0])
            )
        ],
        "metric": DistanceMetric.L2,
        "max_distance": 1000000,
        "kwargs": {"trash_cost": 1000},
        "exact_total_cost": np.sqrt(2.0),
    }
    return cases


def synthetic_cases() -> dict[str, dict]:
    """Ground-truth cases probing the intensity and distance resolution axes."""
    cases = {}

    # Intensity axis: increasing dynamic range at fixed geometry.  A tied
    # scale factor has to spend its budget on positions too, so the faint end
    # of these spectra is the first thing to disappear.
    for dyn in (1e3, 1e6, 1e9):
        emp, theo, truth = make_shifted_pair(
            n=200, dyn_range=dyn, shift=1.0, seed=20260901, gap=10.0
        )
        cases[f"dynrange_1e{int(np.log10(dyn))}"] = {
            "empirical": emp,
            "theoreticals": [theo],
            "metric": DistanceMetric.L1,
            "max_distance": 5.0,
            "kwargs": {"trash_cost": 50.0},
            "truth": truth,
        }

    # The same geometry with intensities normalized to unit total mass, as the
    # published LC-MS and NMR pipelines do.  Ratios are unchanged, so the
    # correct alignment is identical -- only the absolute magnitudes move, and
    # with them whether the faint tail survives quantization at all.
    for dyn in (1e6, 1e9):
        emp, theo, truth = make_shifted_pair(
            n=200, dyn_range=dyn, shift=1.0, seed=20260901, gap=10.0, normalize=True
        )
        cases[f"dynrange_norm_1e{int(np.log10(dyn))}"] = {
            "empirical": emp,
            "theoreticals": [theo],
            "metric": DistanceMetric.L1,
            "max_distance": 5.0,
            "kwargs": {"trash_cost": 50.0},
            "truth": truth,
        }

    # Same, in 3-D, where the position budget is spread over more coordinates.
    emp, theo, truth = make_shifted_pair(
        n=150, dyn_range=1e6, shift=0.5, seed=20260902, dim=3, gap=10.0
    )
    cases["dynrange_3d"] = {
        "empirical": emp,
        "theoreticals": [theo],
        "metric": DistanceMetric.L2,
        "max_distance": 5.0,
        "kwargs": {"trash_cost": 50.0},
        "truth": truth,
    }

    # Distance axis: the decisive distances are ~1e-4 while intensities are
    # ~1e7, so the cost grid is what has to be fine here.
    emp, theo, truth = make_fine_separation_pair(n=120, separation=1e-4, seed=20260903)
    cases["fine_separation"] = {
        "empirical": emp,
        "theoreticals": [theo],
        "metric": DistanceMetric.L1,
        "max_distance": 5e-4,
        "kwargs": {"trash_cost": 1e-3},
        "truth": truth,
    }

    # Asymmetric trash, exercised on real-ish dynamic range.
    emp, theo, truth = make_shifted_pair(
        n=150, dyn_range=1e5, shift=1.0, seed=20260904, gap=10.0
    )
    cases["asymmetric_trash"] = {
        "empirical": emp,
        "theoreticals": [theo],
        "metric": DistanceMetric.L1,
        "max_distance": 5.0,
        "kwargs": {"experimental_trash_cost": 30.0, "theoretical_trash_cost": 40.0},
        "truth": truth,
    }
    return cases


#: The published LC-MS recipe, copied from ``tests/test_lcms_align.py`` /
#: ``tutorials/lcms``: the m/z axis is stretched by ``max_rt_shift /
#: max_mz_shift`` so that a single isotropic LINF tolerance expresses both the
#: m/z and the retention-time window, and intensities are normalized to sum to
#: 1.  Both details matter here.  The stretch pushes m/z coordinates to ~1e8,
#: and normalization pushes intensities down to ~1e-8 — so this workload sits
#: at the far end of *both* scaling axes simultaneously, which is exactly the
#: squeeze a single tied factor cannot serve.
LCMS_MAX_MZ_SHIFT = 0.005
LCMS_MAX_RT_SHIFT = 800.0

LCMS_FILES = {
    "run_2010_08_15": "100825O2c1_MT-AU-0044-2010-08-15_038.csv",
    "run_2010_08_01": "100820O2c1_MT-AU-0044-2010-08-1_030.csv",
    "run_2012_10_04": "121004OTc2_TDM-AU-0324-EMC-1_011.csv",
}


def load_lcms(datadir, name: str, n_peaks: int | None = None) -> Spectrum:
    """Load one LC-MS run exactly as the published pipeline does."""
    import pandas as pd

    df = pd.read_csv(datadir / LCMS_FILES[name])
    if n_peaks is not None and len(df) > n_peaks:
        df = df.nlargest(n_peaks, "Intensity")
    scale_mz = LCMS_MAX_RT_SHIFT / LCMS_MAX_MZ_SHIFT
    positions = np.array(
        [df["m/z"].values * scale_mz, df["Retention time"].values], dtype=float
    )
    intens = np.array(df["Intensity"].values, dtype=float)
    return Spectrum(positions, intens / intens.sum())


#: The publication NMR settings, from ``publication/nmr/nmr/align.py``'s
#: ``align_pair`` defaults and the notebooks that call it.  ``scale_nucl``
#: divides a nucleus axis so that one isotropic ``max_distance`` means the same
#: thing on every dimension: 15N spans ~100-135 ppm against 1H's 6-11, and 13C
#: ~14-34 against 1H's -0.5-6.5, so both get divided by 10.  Getting this wrong
#: does not error, it just silently aligns on the wrong axis.
NMR_MAX_DISTANCE = 0.05
NMR_TRASH_COST = 0.09


def nmr_cases(repo_root) -> dict[str, dict]:
    """Real NMR spectra from ``publication/nmr`` — the paper's own data.

    Returns an empty dict when the publication tree or its loader is absent, so
    the harness still runs from a wheel install.
    """
    import sys

    pub = repo_root / "publication" / "nmr"
    if not pub.is_dir():
        return {}
    if str(pub) not in sys.path:
        sys.path.insert(0, str(pub))
    try:
        from nmr.load_spectra import load_spectrum
    except Exception:
        return {}

    def load(rel, dim, scale_nucl, max_peak_fraction):
        path = pub / rel
        if not path.is_file():
            return None
        spec = load_spectrum(
            str(path),
            dim=dim,
            scale_nucl=scale_nucl,
            max_peak_fraction=max_peak_fraction,
        )
        # align_pair normalizes by default; keep the published pipeline exactly.
        return Spectrum(
            np.asarray(spec.positions, dtype=float),
            np.asarray(spec.intensities, dtype=float)
            / float(np.sum(spec.intensities)),
        )

    cases: dict[str, dict] = {}

    # 2D: GB1 15N-HSQC temperature series, consecutive pairs as the notebook
    # aligns them, filtered to the top 10% of peaks.
    gb1 = "2D/15N_HSQC_GB1_reduced/gNhsqc_GB1_{}C.csv"
    for lo, hi in (("25", "30"), ("30", "35")):
        a = load(gb1.format(lo), 2, {"15N": 10}, 0.1)
        b = load(gb1.format(hi), 2, {"15N": 10}, 0.1)
        if a is None or b is None:
            continue
        cases[f"nmr_2d_gb1_{lo}_vs_{hi}"] = {
            "empirical": a,
            "theoreticals": [b],
            "metric": DistanceMetric.L2,
            "max_distance": NMR_MAX_DISTANCE,
            "kwargs": {"trash_cost": NMR_TRASH_COST},
        }

    # 4D: 2LX7 CCNOESY aliphatic against its own local maxima (the 127 cluster
    # centres the paper uses to define ground-truth classes).
    full = load("4D/2LX7/2LX7_0.1/2LX7_CCNOESY_@ALI.csv", 4, {"C": 10}, None)
    peaks = load(
        "4D/2LX7/2LX7_0.1_localmax/2LX7_CCNOESY_@ALI_localmax_5.csv", 4, {"C": 10}, None
    )
    if full is not None and peaks is not None:
        cases["nmr_4d_2lx7_vs_localmax"] = {
            "empirical": full,
            "theoreticals": [peaks],
            "metric": DistanceMetric.L2,
            "max_distance": 0.10,
            "kwargs": {"trash_cost": NMR_TRASH_COST},
        }
    return cases


def lcms_cases(datadir, n_peaks: int | None = 6000) -> dict[str, dict]:
    """Real LC-MS spectra from ``tutorials/lcms/data``.

    ``n_peaks`` keeps the most intense peaks so the case stays fast enough for
    routine runs; ``None`` uses the full ~40k-peak files.  Note that
    subsampling by intensity *narrows* the dynamic range, so the full-file
    variant is the honest quantization test — hence the ``lcms_full`` case.
    """
    mtd = round(LCMS_MAX_RT_SHIFT)

    def case(emp_name, theo_name, n):
        return {
            "empirical": load_lcms(datadir, emp_name, n),
            "theoreticals": [load_lcms(datadir, theo_name, n)],
            "metric": DistanceMetric.LINF,
            "max_distance": mtd,
            "kwargs": {"trash_cost": mtd},
        }

    return {
        # Two runs of the same sample five days apart: replicates, so most
        # peaks genuinely correspond.
        "lcms_replicates": case("run_2010_08_15", "run_2010_08_01", n_peaks),
        # A different sample two years later: a genuinely harder alignment.
        "lcms_cross_batch": case("run_2010_08_15", "run_2012_10_04", n_peaks),
        # Full files, no subsampling — the real published workload and the one
        # whose faint tail actually probes the intensity grid.
        "lcms_full": case("run_2010_08_15", "run_2010_08_01", None),
    }
