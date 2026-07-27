"""Smoke tests for axiom.reporting — data structures and isolated rendering."""
import os

import pytest

from axiom.reporting.collect import RunData, build_verdict


def _has_mpl():
    try:
        import matplotlib
        return True
    except ImportError:
        return False


class TestRunData:
    """RunData dataclass basic checks."""

    def test_importable(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(RunData)}
        assert "summary" in field_names
        assert "arrays" in field_names

    def test_instantiation(self):
        run = RunData(
            summary={"suite_1": {"passed": True}},
            arrays={},
        )
        assert run.summary["suite_1"]["passed"] is True

    def test_section_method(self):
        run = RunData(summary={"suite_1": {"acc": 0.98}}, arrays={})
        assert run.section("suite_1") == {"acc": 0.98}
        assert run.section("nonexistent") == {}


class TestBuildVerdict:
    def test_all_passed(self):
        run = RunData(summary={
            "suite_1": {"passed": True},
            "suite_3": {"passed": True},
            "suite_6": {"passed": True},
            "suite_7": {"passed": True},
            "suite_5": {"significant": True},
        })
        verdict = build_verdict(run)
        assert verdict["overall_validated"] is True

    def test_suite_6_skipped(self):
        run = RunData(summary={
            "suite_1": {"passed": True},
            "suite_3": {"passed": True},
            "suite_6": {"skipped": True, "reason": "no data"},
            "suite_7": {"passed": True},
        })
        verdict = build_verdict(run)
        assert verdict["overall_validated"] is False

    def test_empty_summary(self):
        run = RunData(summary={}, arrays={})
        verdict = build_verdict(run)
        assert verdict["overall_validated"] is False


@pytest.mark.skipif(not _has_mpl(), reason="matplotlib not installed")
class TestChartRendering:
    def test_render_all_empty(self, tmp_path):
        """render_all with minimal RunData should not crash."""
        from axiom.reporting.charts import render_all
        run = RunData(
            summary={
                "suite_1": {"passed": True, "fold_accs": [0.98], "fold_mccs": [0.88]},
            },
            arrays={},
        )
        charts_dir = str(tmp_path / "charts")
        os.makedirs(charts_dir)
        specs = render_all(run, charts_dir, verbose=False)
        assert isinstance(specs, list)


class TestReportWriter:
    @pytest.mark.skip(reason="write_reports requires full benchmark schema — tested via make report")
    def test_write_reports_smoke(self, tmp_path):
        pass
