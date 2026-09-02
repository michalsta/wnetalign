"""Shared pytest configuration.

The ``realdata`` marker is registered here rather than in ``pyproject.toml``
so that the marker travels with the tests that use it.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "realdata: exercises the committed LC-MS / NMR datasets and pins exact "
        "numbers against tests/baselines/. Deselect with -m 'not realdata'. CI "
        "runs these on one canonical job only: the pinned values are produced "
        "by floating-point arithmetic (distances, the scaler's sqrt) feeding an "
        "exact integer solver, so a last-ulp difference between architectures "
        "or compilers can tip a tie between two equally-optimal matchings and "
        "change the answer without anything being wrong.",
    )
