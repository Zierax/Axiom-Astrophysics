#!/usr/bin/env python3
"""Generate the full research-grade benchmark report tree.

Runs every validation suite deterministically, renders the scientific figure
gallery and writes the markdown + JSON reports under ``benchmarks/``::

    benchmarks/reports/README.md      executive summary + chart gallery
    benchmarks/reports/summary.json   master machine-readable record
    benchmarks/reports/suite_*.md/.json
    benchmarks/charts/*.png           15+ figures (300 dpi)

Usage::

    python3 scripts/generate_reports.py [--root benchmarks] [--quiet]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom.reporting import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="benchmarks",
                        help="output root directory (default: benchmarks)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress")
    args = parser.parse_args()

    master = generate(root=args.root, verbose=not args.quiet)
    verdict = master.get("verdict", {})
    n_charts = len(master.get("charts", []))
    print(f"\nReports written under {args.root}/reports "
          f"({n_charts} charts under {args.root}/charts).")
    print(f"Overall validated: {verdict.get('overall_validated')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
