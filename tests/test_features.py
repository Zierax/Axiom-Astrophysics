"""Tests for the physics-based HTRU2 feature mapping and manifold placement.

These tests guard against the historical-audit degeneracy: natural / interference
sources must be placed ON the real HTRU2 manifold (natural / interference
clusters). Critically, narrowband / unknown carriers must NOT be placed on a
*hand-coded* off-manifold anchor -- that would make OOD separation a
by-construction guarantee rather than a measurement. The anomaly verdict for such
sources must come from their REAL measured spectrogram (descriptor-conformal /
chaos paths), so the mapping returns None for them.
"""
import numpy as np
import pytest

from axiom.data.loader import load_htru2
from axiom.dsp.features import physics_map_htru2_features
from axiom.ml.density import AnomalyDensityEstimator

SEED = 7


@pytest.fixture(scope="module")
def density():
    X, y, _ = load_htru2()
    d = AnomalyDensityEstimator(n_components=5)
    d.fit(X, y)
    return d, X, y


def test_feature_vector_shape_and_finite():
    vec = physics_map_htru2_features("Pulsar", dm=12.58, snr=20.0, seed=SEED)
    assert vec.shape == (8,)
    assert np.all(np.isfinite(vec))


def test_natural_signals_are_on_manifold(density):
    """Pulsars / FRBs must sit inside the natural population (high density)."""
    d, X, y = density
    natural_min = float(np.min(d.log_prob(X)))
    for sig_type, dm, snr in [("Pulsar", 56.0, 30.0), ("FRB", 558.0, 20.0),
                              ("Quasar", 0.0, 5.0), ("Transit", 0.0, 5.0)]:
        vec = physics_map_htru2_features(sig_type, dm=dm, snr=snr, seed=SEED)
        assert vec is not None
        score = d.log_prob(vec[None, :])[0]
        assert score >= natural_min - 5.0, f"{sig_type} fell off-manifold: {score:.1f}"


def test_interference_signals_are_on_manifold(density):
    d, X, y = density
    natural_min = float(np.min(d.log_prob(X)))
    for sig_type, dm, snr in [("RFI", 0.0, 10.0), ("Peryton", 0.0, 50.0)]:
        vec = physics_map_htru2_features(sig_type, dm=dm, snr=snr, seed=SEED)
        assert vec is not None
        score = d.log_prob(vec[None, :])[0]
        assert score >= natural_min - 5.0, f"{sig_type} fell off-manifold: {score:.1f}"


def test_narrowband_returns_none_not_planted_anchor():
    """Narrowband / unknown carriers must NOT be placed on a hand-coded
    off-manifold vector (that would make OOD trivially 'correct' by construction).
    The mapping returns None so the caller relies on the signal's real measured
    spectrogram for the anomaly verdict."""
    for sig_type in ["Narrowband", "Telemetry", "Transmitted", "Unknown"]:
        vec = physics_map_htru2_features(sig_type, dm=0.0, snr=20.0, seed=SEED)
        assert vec is None, f"{sig_type} must return None, got {vec}"


def test_mapping_is_deterministic():
    a = physics_map_htru2_features("Pulsar", dm=56.0, snr=30.0, seed=SEED)
    b = physics_map_htru2_features("Pulsar", dm=56.0, snr=30.0, seed=SEED)
    assert np.allclose(a, b)
