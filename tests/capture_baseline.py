#!/usr/bin/env python
"""Capture (or re-capture) the alignment regression baseline.

    python tests/capture_baseline.py [-o OUT] [--quick]

Writes a JSON summary of every harness case.  The committed copy at
``tests/baselines/alignment_baseline.json`` is what
``test_alignment_regression.py`` pins against; re-run this only when a change
to the alignment result is intended, and say so in the commit message.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import align_harness as H  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(__file__).resolve().parent / "baselines" / "alignment_baseline.json"
DATADIR = REPO / "tutorials" / "lcms" / "data"


def all_cases(quick: bool) -> dict:
    cases = {}
    cases.update(H.toy_cases())
    cases.update(H.synthetic_cases())
    cases.update(H.nmr_cases(REPO))
    if DATADIR.is_dir():
        lcms = H.lcms_cases(DATADIR)
        if quick:
            lcms.pop("lcms_full", None)
        cases.update(lcms)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true", help="skip the full-size LC-MS case")
    args = ap.parse_args()

    results = {}
    for name, case in all_cases(args.quick).items():
        t0 = time.perf_counter()
        try:
            summary = H.run_case(case)
        except Exception as exc:  # a case that cannot run is itself a finding
            print(f"{name:24s} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        summary["seconds"] = round(time.perf_counter() - t0, 3)
        results[name] = summary
        print(
            f"{name:24s} cost={summary['total_cost']:.6g} "
            f"transport={summary['transport_cost']:.6g} "
            f"dropped={summary['emp_dropped_peaks']} "
            f"recall={summary.get('truth_recall', float('nan')):.4f} "
            f"[{summary['seconds']}s]"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
