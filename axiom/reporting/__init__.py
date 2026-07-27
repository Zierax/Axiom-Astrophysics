"""Research-grade benchmark reporting for axiom-astrophysics.

This package turns the validation suites into a reproducible, publication-quality
artifact tree::

    benchmarks/
      reports/
        README.md            general executive summary (+ chart gallery)
        summary.json         master machine-readable record
        suite_*.md / .json   per-suite detailed reports
      charts/
        *.png                15+ scientific figures (300 dpi)

The public entry point is :func:`generate`, which collects every suite's real
computed data (no hard-coded numbers), renders the figures and writes the
reports. All computation is deterministic under a fixed seed.
"""

from .collect import RunData, collect_all
from .pipeline import generate

__all__ = ["RunData", "collect_all", "generate"]
