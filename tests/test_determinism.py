"""Determinism contracts for stochastic code paths.

Every stochastic generator in the codebase must satisfy two properties:
  1. Same seed -> bit-identical output (reproducibility).
  2. Seeded calls must not perturb the global RNG state (isolation), so
     pinning one component cannot silently change another component's draws.

These tests cover the paths flagged as unseeded or globally-seeding:
``simulate_htru2_features``, the self-healing cache generator, and the
``synthesis`` waveform generators.
"""
import numpy as np

from axiom.data.cache import generate_self_healing_cache
from axiom.dsp.features import simulate_htru2_features
from axiom.dsp.synthesis import generate_noise, synthesize_frb, synthesize_pulsar


def _waveform():
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 1.0, size=256) + 5.0


def test_simulate_htru2_features_seeded_deterministic():
    w = _waveform()
    a = simulate_htru2_features(w, true_dm=100.0, peak_snr=20.0, seed=42)
    b = simulate_htru2_features(w, true_dm=100.0, peak_snr=20.0, seed=42)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a))
    assert a.shape == (8,)


def test_simulate_htru2_features_seed_changes_draws():
    w = _waveform()
    a = simulate_htru2_features(w, true_dm=100.0, peak_snr=20.0, seed=42)
    b = simulate_htru2_features(w, true_dm=100.0, peak_snr=20.0, seed=43)
    assert not np.array_equal(a, b)


def test_self_healing_cache_seeded_reproducible():
    def _science_fields(records):
        return [{k: v for k, v in r.items() if k != "signal_id"}
                for r in records]

    a = generate_self_healing_cache(seed=0)
    b = generate_self_healing_cache(seed=0)
    assert len(a) == len(b) == 200
    assert _science_fields(a) == _science_fields(b)


def test_synthesis_seeded_deterministic():
    a = synthesize_pulsar(seed=1)
    b = synthesize_pulsar(seed=1)
    np.testing.assert_array_equal(a, b)
    c = generate_noise(256, amplitude=0.5, seed=9)
    d = generate_noise(256, amplitude=0.5, seed=9)
    np.testing.assert_array_equal(c, d)


def test_synthesis_does_not_pollute_global_rng():
    np.random.seed(123)
    before = np.random.rand(3)
    np.random.seed(123)
    synthesize_frb(seed=7)
    synthesize_pulsar(seed=8)
    after = np.random.rand(3)
    np.testing.assert_array_equal(before, after)
