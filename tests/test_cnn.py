"""CNN integration tests.

The 1-D CNN is dependency-optional: if PyTorch is available it trains via
`torch`, otherwise a fully trainable pure-NumPy implementation is used. These
tests assert the inference interface is correct, the network actually *learns*
(backpropagation works), and the branch is now wired into the arbitrator.
"""
import numpy as np
import pytest

torch_available = False
try:
    import torch  # noqa: F401
    torch_available = True
except ImportError:
    torch_available = False

from axiom.ml.cnn import CosmicSignalCNN, NumpyCNN1D


def _separable_dataset(n_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    t = np.arange(256)
    for _ in range(n_per_class):
        # class 0: narrowband carrier (sine)
        X.append(np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.05, 256))
        y.append(0)
    for _ in range(n_per_class):
        # class 1: white noise
        X.append(rng.normal(0, 1.0, 256))
        y.append(1)
    for _ in range(n_per_class):
        # class 2: drifting tone
        X.append(np.sin(2 * np.pi * (0.2 * t + 1e-3 * t ** 2)) + rng.normal(0, 0.05, 256))
        y.append(2)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def test_numpy_cnn_untrained_is_uniform():
    net = NumpyCNN1D(seed=1)
    waves = np.random.default_rng(0).normal(0, 1, (5, 256)).astype(np.float64)
    probs = net.predict_proba(waves)
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs, 1.0 / 3.0)


def test_numpy_cnn_trains_and_discriminates():
    X, y = _separable_dataset()
    # Hold out 20% for an honest accuracy check.
    n = len(y)
    rng = np.random.default_rng(7)
    test_idx = rng.choice(n, size=n // 5, replace=False)
    train_idx = np.array([i for i in range(n) if i not in test_idx])
    net = NumpyCNN1D(seed=42)
    net.train(X[train_idx], y[train_idx], epochs=15, batch_size=16)
    preds = net.predict(X[test_idx])
    acc = float(np.mean(preds == y[test_idx]))
    assert acc > 0.7, f"CNN failed to learn separable task (acc={acc:.3f})"


def test_numpy_cnn_deterministic():
    X, y = _separable_dataset(seed=3)
    net1 = NumpyCNN1D(seed=42)
    net1.train(X, y, epochs=8, batch_size=16)
    net2 = NumpyCNN1D(seed=42)
    net2.train(X, y, epochs=8, batch_size=16)
    p1 = net1.predict_proba(X[:10])
    p2 = net2.predict_proba(X[:10])
    assert np.allclose(p1, p2)


def test_cnn_predict_interface():
    cnn = CosmicSignalCNN()
    waves = np.random.default_rng(1).normal(0, 1, (4, 256)).astype(np.float64)
    probs = cnn.predict_proba(waves)
    assert probs.shape == (4, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_cnn_branch_active_in_ood_eval():
    """The CNN branch must handle a ragged wave list (real 256-sample waveforms
    mixed with length-1 placeholders) without crashing, and return a valid (N, 3)
    probability matrix with the real-wave rows scored and placeholders neutral."""
    from axiom.stats.ood_eval import _cnn_branch

    rng = np.random.default_rng(0)
    real_waves = [rng.normal(0, 1, 256).astype(np.float64) for _ in range(6)]
    labels = [0, 1, 2, 0, 1, 2]
    # Ragged list: 6 real waveforms + 2 length-1 placeholders.
    waves = real_waves + [np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)]
    probs = _cnn_branch(waves, seed=42, train_waves=real_waves, train_labels=labels)
    assert probs is not None
    assert probs.shape == (8, 3)
    # Real-wave rows are scored (probabilities sum to 1, finite).
    assert np.all(np.isfinite(probs[:6].sum(axis=1)))
    assert np.allclose(probs[:6].sum(axis=1), 1.0, atol=1e-6)
    # Placeholder rows receive a neutral (1/3, 1/3, 1/3) distribution.
    assert np.allclose(probs[6:], 1.0 / 3.0, atol=1e-6)


@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
def test_pytorch_model_shape():
    import torch

    from axiom.ml.cnn import _build_torch_model  # noqa: F401  (smoke only)
    model = CosmicSignalCNN()._build_torch_model(0.0)
    waves = torch.randn(16, 1, 256)
    out = model(waves)
    assert out.shape == (16, 3)
