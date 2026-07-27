"""Top-level orchestration: collect -> render charts -> write reports."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from .charts import render_all

log = logging.getLogger(__name__)
from .collect import RunData, build_verdict, collect_all
from .report_writer import write_reports

DEFAULT_ROOT = "benchmarks"


def generate(root: str = DEFAULT_ROOT, *, verbose: bool = True,
             run: RunData | None = None) -> Dict[str, Any]:
    """Generate the full benchmark artifact tree under ``root``.

    Parameters
    ----------
    root:
        Output directory; ``<root>/reports`` and ``<root>/charts`` are created.
    verbose:
        Emit progress to stdout.
    run:
        Optional pre-collected :class:`RunData` (skips recomputation).

    Returns
    -------
    dict
        The master summary record (also written to ``reports/summary.json``).
    """
    reports_dir = os.path.join(root, "reports")
    charts_dir = os.path.join(root, "charts")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    if run is None:
        run = collect_all(verbose=verbose)
    verdict = build_verdict(run)
    specs = render_all(run, charts_dir, verbose=verbose)
    master = write_reports(run, verdict, specs, reports_dir, verbose=verbose)

    if verbose:
        log.info("Rendering %d charts...", len(specs))
    return master
