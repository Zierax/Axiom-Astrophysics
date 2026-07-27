"""Smoke/accuracy tests for the stacking ensemble and density estimator."""
import numpy as np
import pytest

from axiom.data.loader import load_htru2, split_htru2
from axiom.ml.density import AnomalyDensityEstimator
from axiom.ml.ensemble import AxiomEnsemble


@pytest.fixture(scope="module")
def data():
    X, y, _ = load_htru2()
    return split_htru2(X, y, train_ratio=0.7, val_ratio=0.15, seed=42)


def test_ensemble_accuracy(data):
    splits = data
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    ens = AxiomEnsemble(n_classes=2, random_state=42)
    ens.fit(X_train, y_train)
    preds = ens.predict(X_test)
    acc = float(np.mean(preds == y_test))
    # HTRU2 is heavily imbalanced (10:1 RFI:pulsar); the classifier should still
    # reach the ~0.98 accuracy the project targets.
    assert acc >= 0.97, f"ensemble accuracy {acc:.4f} below 0.97"


def test_density_per_class_scores_higher_for_own_class():
    """A pulsar point scores higher under the pulsar density than an RFI point
    does; conversely the RFI point (off-manifold for the pulsar class) scores
    LOWER under the pulsar density than the pulsar point does. (Narrowband /
    unknown carriers intentionally have no HTRU2 anchor and are routed through
    the descriptor-conformal path, so the density test uses RFI as the OOD
    example.)"""
    X, y, _ = load_htru2()
    d = AnomalyDensityEstimator(n_components=5)
    d.fit(X, y)

    from axiom.dsp.features import physics_map_htru2_features
    pulsar = physics_map_htru2_features("Pulsar", dm=56.0, snr=30.0, seed=3)[None, :]
    rfi = physics_map_htru2_features("RFI", dm=0.0, snr=20.0, seed=3)[None, :]
    assert pulsar is not None and rfi is not None

    # A pulsar point scores higher under the pulsar density (class 1) than an
    # RFI point does, and higher under its own class than under the RFI density
    # (class 0): the per-class estimator distinguishes the two populations.
    assert d.log_prob_per_class(pulsar, 1)[0] > d.log_prob_per_class(rfi, 1)[0]
    assert d.log_prob_per_class(pulsar, 1)[0] > d.log_prob_per_class(pulsar, 0)[0]
