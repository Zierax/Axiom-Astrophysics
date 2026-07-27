#!/usr/bin/env python3
"""Historical anomaly audit CLI — comprehensive real-signal verification.

Scores 65+ historically recorded anomalous signals through the HTRU2 manifold
and real catalogs (ATNF pulsars, CHIME/FRB) on the physical manifold, ranking
the most off-manifold objects by calibrated conformal p-value.

Usage
-----
    # Full audit: historical signals + real catalogs
    python3 scripts/historical_audit.py

    # Historical signals only (no catalog fetch)
    python3 scripts/historical_audit.py --historical-only

    # Real catalogs only (no historical signals)
    python3 scripts/historical_audit.py --catalogs-only

    # Also re-pin catalog digest locks after a verified upstream release
    python3 scripts/historical_audit.py --repin

    # Write a markdown report
    python3 scripts/historical_audit.py --report paper/historical/historical_report.md

An off-manifold verdict is a statistical triage signal for human follow-up --
NOT a claim of artificial origin or a new discovery.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom.historical import (
    run_curated_audit,
    run_historical_verification,
    run_real_catalog_audit,
    run_unclassified_htru2_audit,
)


def _fmt_historical_verification(res) -> str:
    """Format the comprehensive historical signal verification lane."""
    top = res.top(30)
    by_type = res.by_type()

    lines = [
        "## Lane D -- Historical Signal Verification (FULL AUDIT)",
        "",
        f"**{len(res.rows)}** historically recorded signals scored through "
        "the HTRU2 manifold. The density model is fit on HTRU2 pulsars + RFI "
        "noise candidates; signals with log-likelihood below the natural floor "
        f"minus {res.ood_margin} are flagged OOD.",
        "",
        f"Natural floor log-likelihood: `{res.natural_floor:.3f}`",
        "",
        "### Per-type summary",
        "",
        "| Type | Count | OOD | Fraction OOD | Mean loglik | Min loglik |",
        "|---|---|---|---|---|---|",
    ]
    for t, stats in sorted(by_type.items()):
        lines.append(
            f"| {t} | {stats['count']} | {stats['ood_count']} | "
            f"{stats['fraction_ood']:.1%} | {stats['mean_loglik']:.3f} | "
            f"{stats['min_loglik']:.3f} |"
        )

    lines.extend([
        "",
        "### Top 30 most off-manifold signals",
        "",
        "| Rank | Signal | Type | DM | S/N | loglik | Verdict | Telescope | Freq (MHz) | Reference |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for i, row in enumerate(top, 1):
        ref_short = row["reference"][:40] + "..." if len(row["reference"]) > 40 else row["reference"]
        lines.append(
            f"| {i} | {row['name']} | {row['type']} | {row['true_dm']:.1f} | "
            f"{row['peak_snr']:.1f} | {row['loglik']:.3f} | {row['verdict']} | "
            f"{row['telescope']} | {row['frequency_mhz']:.0f} | {ref_short} |"
        )

    return "\n".join(lines)


def _fmt_curated(res) -> str:
    """Format the illustrative curated lane."""
    lines = [
        "## Lane A -- Curated historical signals (ILLUSTRATIVE)",
        "",
        "> These signals are scored through the HTRU2 manifold to show where "
        "documented signal *classes* fall. This lane is a teaching/sanity-check "
        "illustration only.",
        "",
        f"Natural-floor log-likelihood: `{res.natural_floor:.3f}` "
        f"(OOD margin = {res.ood_margin}).",
        "",
        "| Rank | Signal | Class | log-lik | Verdict | placeholder? |",
        "|---|---|---|---|---|---|",
    ]
    for i, (name, kind, score, verdict, ph) in enumerate(res.ranked, 1):
        lines.append(f"| {i} | {name} | {kind} | {score:.3f} | {verdict} | "
                     f"{'YES' if ph else 'no'} |")
    return "\n".join(lines)


def _fmt_real(res) -> str:
    """Format the real catalog measurement lane."""
    top = res.top(25)
    lines = [
        "## Lane B -- Real catalogs (MEASUREMENT)",
        "",
        f"Objects scored: **{res.n_objects}** from "
        f"{', '.join(res.catalog_keys)}. The off-manifold p-value is the "
        "**minimum** per-class split-conformal p-value: an object is flagged if "
        "it is rare under *at least one* natural population (sensitive triage). "
        "Small p means statistically unusual *even among known astrophysical "
        "sources* -- a prioritized list for human review, **not** an "
        "artificiality claim.",
        "",
        "| Rank | Object | Class | DM | Period (s) | Width (ms) | |b| (deg) | S/N | log-lik | p_off |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in top.iterrows():
        name = row.get("object_id", f"obj{i}")
        cls = row.get("class_name", "?")
        dm = row.get("dm", float("nan"))
        per = row.get("period_s", float("nan"))
        wid = row.get("width_ms", float("nan"))
        gl = row.get("glat", float("nan"))
        snr = row.get("snr", float("nan"))
        lines.append(
            f"| {i + 1} | {name} | {cls} | {dm:.1f} | {per:.4g} | {wid:.3g} | "
            f"{gl:.1f} | {snr:.1f} | {row['manifold_logprob']:.3f} | "
            f"{row['off_manifold_pval']:.4f} |"
        )
    return "\n".join(lines)


def _fmt_unclassified(res) -> str:
    """Format the unclassified discovery protocol lane."""
    top = res.top(15)
    lines = [
        "## Lane C -- Unclassified survey candidates (DISCOVERY PROTOCOL)",
        "",
        f"Pool: **{len(res.indices)}** HTRU2 candidates the ensemble could not "
        "classify (pulsar probability in [0.4, 0.6]) -- the survey's own "
        "unclassified set. Ranked by **p_max** (rare under *both* pulsar and RFI "
        "manifolds). HTRU2 is labeled, so this lane *validates the protocol*; a "
        "real unlabeled survey (MeerKAT/ASKAP/TNS) would use the identical filter "
        "to surface genuine unknowns.",
        "",
        "| Rank | HTRU2 idx | ens. P(pulsar) | p_min | p_max | true_label |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in top.iterrows():
        lbl = row.get("true_label", "?")
        lines.append(
            f"| {i + 1} | {int(row['htru2_index'])} | {row['ensemble_p_pulsar']:.3f} "
            f"| {row['p_min']:.4f} | {row['p_max']:.4f} | {lbl} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repin", action="store_true",
                    help="deliberately re-pin catalog digest locks to upstream releases")
    ap.add_argument("--historical-only", action="store_true",
                    help="only run the historical signal verification lane")
    ap.add_argument("--catalogs-only", action="store_true",
                    help="only run the real catalog lane")
    ap.add_argument("--unclassified", action="store_true",
                    help="run Lane C: HTRU2 unclassified-candidate discovery protocol")
    ap.add_argument("--report", default=None,
                    help="write a markdown report to this path")
    args = ap.parse_args()

    parts = [
        "# Historical Anomaly Audit",
        "",
        "_Generated by `scripts/historical_audit.py`. An off-manifold verdict is a "
        "statistical triage signal for human follow-up -- **not** a claim of "
        "artificial origin or a new discovery._",
        "",
    ]

    if not args.catalogs_only:
        print("[historical] running full historical signal verification...")
        hist = run_historical_verification()
        print(f"[historical] scored {len(hist.rows)} historical signals; "
              f"OOD count: {sum(1 for v in hist.verdict if v == 'OOD')}")
        parts.append(_fmt_historical_verification(hist))
        parts.append("")

    if not args.historical_only:
        print("[historical] running real-catalog lane (this fetches ATNF + CHIME/FRB)...")
        real = run_real_catalog_audit(repin=args.repin)
        print(f"[historical] scored {real.n_objects} real objects; "
              f"top p_off={real.top(1)['off_manifold_pval'].iloc[0]:.4f}")
        parts.append(_fmt_real(real))
        parts.append("")

    if args.unclassified:
        print("[historical] running unclassified-candidate protocol (Lane C)...")
        unc = run_unclassified_htru2_audit()
        print(f"[historical] unclassified pool size {len(unc.indices)}; "
              f"top p_max={unc.top(1)['p_max'].iloc[0]:.4f}")
        parts.append(_fmt_unclassified(unc))
        parts.append("")

    print("[historical] running curated lane (illustrative)...")
    curated = run_curated_audit()
    parts.append(_fmt_curated(curated))
    parts.append("")

    text = "\n".join(parts)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            fh.write(text)
        print(f"[historical] wrote {args.report}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
