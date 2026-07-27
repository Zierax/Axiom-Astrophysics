"""Tests for the physics-law module and its arbitrator integration.

These verify the laws are deterministic, bounded, and encode real astrophysical
consistency (no fabricated anchors), and that the arbitrator escalates a
physically self-contradictory source (a narrowband tone claimed to be a dispersed
natural pulse) to Candidate rather than silently accepting it.
"""
import numpy as np

from axiom.dsp.physics_rules import (
    combine_physics_laws,
    dispersion_excess_law_score,
    duty_cycle_consistency_law_score,
    technosignature_law_score,
)
from axiom.stats.arbitration import SignalArbitrator


def test_technosignature_law_bounded_and_deterministic():
    tone = {"concentration": 0.9, "spectral_flatness": 0.1}
    broadband = {"concentration": 0.1, "spectral_flatness": 0.9}
    s1 = technosignature_law_score(tone)
    s2 = technosignature_law_score(tone)
    assert s1 == s2  # deterministic
    assert 0.0 <= s1 <= 1.0
    # A tone scores far higher than a broadband natural process.
    assert s1 > technosignature_law_score(broadband)
    # Claiming a dispersed natural pulse while being tonal is amplified.
    assert technosignature_law_score(tone, claimed_dispersed=True) >= s1


def test_technosignature_law_handles_missing():
    # Missing descriptors -> neutral, finite, bounded.
    s = technosignature_law_score({})
    assert np.isfinite(s) and 0.0 <= s <= 1.0


def test_dispersion_excess_law_extragalactic():
    # FRB claimed extragalactic with DM above the Galactic ceiling -> consistent.
    assert dispersion_excess_law_score(900.0, dm_gal_max=50.0, is_extragalactic_claim=True) > 0.9
    # FRB claimed extragalactic but BELOW the ceiling -> contradiction (anomaly).
    assert dispersion_excess_law_score(10.0, dm_gal_max=50.0, is_extragalactic_claim=True) == 0.0
    # Not an extragalactic claim -> neutral regardless of DM.
    assert dispersion_excess_law_score(10.0, dm_gal_max=50.0, is_extragalactic_claim=False) == 0.5
    # Missing ceiling -> neutral.
    assert dispersion_excess_law_score(900.0, dm_gal_max=None, is_extragalactic_claim=True) == 0.5


def test_duty_cycle_consistency_law():
    # Reasonable pulsar duty cycle -> consistent (high).
    assert duty_cycle_consistency_law_score(width_ms=10.0, period_s=0.5) > 0.9
    # Ultra-narrow duty at long period -> marginal (low).
    assert duty_cycle_consistency_law_score(width_ms=0.05, period_s=5.0) < 0.3
    # Missing width/period -> neutral.
    assert duty_cycle_consistency_law_score(None, None) == 0.5


def test_combine_physics_laws_spectrogram_only():
    # Without catalog physics, the combined score equals the technosignature law.
    out = combine_physics_laws(technosig=0.8, use_catalog_laws=False)
    assert abs(out - 0.8) < 1e-9
    # With catalog laws, a low-consistency anomaly claim pushes it up.
    cat = combine_physics_laws(technosig=0.8, dispersion=0.0, duty=0.0, use_catalog_laws=True)
    assert cat >= 0.8
    assert 0.0 <= cat <= 1.0


def test_arbitrator_escalates_physical_contradiction():
    """A tonal DM=0 source claimed Natural (high conf) is escalated, not accepted."""
    arb = SignalArbitrator()
    wf = {"concentration": 0.95, "spectral_flatness": 0.05}
    physics = {"concentration": 0.95, "spectral_flatness": 0.05,
               "dm": 0.0, "is_extragalactic": False}
    verdicts, scores = arb.arbitrate(
        ["tone0"], np.array([0]), np.array([[0.92, 0.04, 0.04]]),
        np.array([0.75]), np.array([0.3]), ["Pulsar"],
        ood_mask=np.array([False]),
        waterfall_features={"tone0": wf}, physics_features={"tone0": physics},
    )
    # Physically contradictory -> Candidate, not silently Natural.
    assert verdicts[0] == "Candidate — Requires Review"
    assert scores[0] > 0.0


def test_arbitrator_accepts_physically_consistent_natural():
    """A broadband, physically-consistent natural source stays Natural."""
    arb = SignalArbitrator()
    wf = {"concentration": 0.1, "spectral_flatness": 0.9}
    physics = {"concentration": 0.1, "spectral_flatness": 0.9,
               "dm": 50.0, "dm_gal_max": 40.0, "is_extragalactic": False}
    verdicts, _ = arb.arbitrate(
        ["nat"], np.array([0]), np.array([[0.95, 0.03, 0.02]]),
        np.array([0.9]), np.array([0.2]), ["Pulsar"],
        ood_mask=np.array([False]),
        waterfall_features={"nat": wf}, physics_features={"nat": physics},
    )
    assert verdicts[0] == "Natural"
