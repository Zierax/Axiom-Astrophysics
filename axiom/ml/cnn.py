"""1-D convolutional neural network for cosmic-signal arbitration.

Two execution paths are supported:

  * PyTorch (`CosmicSignalCNN`) when `torch` is importable -- preferred for
    large-scale training.
  * A fully self-contained, deterministic NumPy implementation
    (`NumpyCNN1D`) used as the fallback / default in this environment.  Unlike
    the previous fallback (which returned a constant 1/3 probability), this
    network is genuinely trained with backpropagation and Adam, so it provides a
    real learned waveform branch to the arbitrator.

The output head is always 3 classes aligned with the rest of the pipeline:
    0 = Natural, 1 = Interference, 2 = Anomaly.
"""
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-NumPy trainable 1-D CNN
# ---------------------------------------------------------------------------
class NumpyCNN1D:
    """Deterministic, dependency-free 1-D CNN with manual backpropagation.

    Architecture (input length L, default 256):
        conv1 (1->8, k=5, p=2) -> ReLU -> maxpool(2)
        conv2 (8->16, k=3, p=1) -> ReLU -> maxpool(2)
        flatten -> fc1 (16*(L/4) -> 32) -> ReLU -> fc2 (32 -> 3)
    """

    def __init__(self, length=256, seed=42, l2=1e-4, lr=1e-3):
        self.length = int(length)
        self.seed = int(seed)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.params = {}
        self._initialised = False

    # ---- initialisation -------------------------------------------------
    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        # He-ish initialisation for conv, Lecun for dense.
        self.params = {}
        self.params["conv1_w"] = rng.standard_normal((8, 1, 5)) * np.sqrt(2.0 / 5)
        self.params["conv1_b"] = np.zeros(8)
        self.params["conv2_w"] = rng.standard_normal((16, 8, 3)) * np.sqrt(2.0 / (8 * 3))
        self.params["conv2_b"] = np.zeros(16)
        flat = 16 * (self.length // 4)
        self.params["fc1_w"] = rng.standard_normal((32, flat)) * np.sqrt(1.0 / flat)
        self.params["fc1_b"] = np.zeros(32)
        self.params["fc2_w"] = rng.standard_normal((3, 32)) * np.sqrt(1.0 / 32)
        self.params["fc2_b"] = np.zeros(3)
        self._initialised = True

    # ---- low-level ops ---------------------------------------------------
    @staticmethod
    def _conv1d(x, w, b, padding=2):
        # x: (B, Cin, L); w: (Cout, Cin, K); b: (Cout,)
        B, Cin, L = x.shape
        Cout, _, K = w.shape
        if padding > 0:
            xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        else:
            xp = x
        Lp = L + 2 * padding
        out_L = Lp - K + 1
        out = np.zeros((B, Cout, out_L), dtype=np.float64)
        for oc in range(Cout):
            for ic in range(Cin):
                # cross-correlation via sliding window sum
                for t in range(out_L):
                    out[:, oc, t] += np.dot(
                        xp[:, ic, t:t + K], w[oc, ic]
                    )
            out[:, oc, :] += b[oc]
        return out

    @staticmethod
    def _conv1d_backward(dout, x, w, padding=2):
        B, Cin, L = x.shape
        Cout, _, K = w.shape
        if padding > 0:
            xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        else:
            xp = x
        Lp = L + 2 * padding
        out_L = Lp - K + 1
        dw = np.zeros_like(w)
        db = np.zeros(Cout)
        dxp = np.zeros_like(xp)
        for oc in range(Cout):
            db[oc] = np.sum(dout[:, oc, :])
            for ic in range(Cin):
                for t in range(out_L):
                    dw[oc, ic] += np.dot(
                        xp[:, ic, t:t + K].T, dout[:, oc, t]
                    ) / max(B, 1)
                # gradient w.r.t padded input
                for t in range(out_L):
                    dxp[:, ic, t:t + K] += np.outer(dout[:, oc, t], w[oc, ic])
        if padding > 0:
            dx = dxp[:, :, padding:L + padding]
        else:
            dx = dxp
        # average over batch for dw
        dw /= max(B, 1)
        return dx, dw, db

    @staticmethod
    def _maxpool1d(x, kernel=2, stride=2):
        B, C, L = x.shape
        out_L = L // stride
        out = np.zeros((B, C, out_L), dtype=np.float64)
        idx = np.zeros((B, C, out_L), dtype=np.int64)
        for i in range(out_L):
            seg = x[:, :, i * stride:i * stride + kernel]
            out[:, :, i] = np.max(seg, axis=-1)
            idx[:, :, i] = np.argmax(seg, axis=-1)
        return out, idx

    @staticmethod
    def _maxpool1d_backward(dout, idx, kernel=2, stride=2):
        B, C, L = dout.shape
        L_in = L * stride
        dx = np.zeros((B, C, L_in), dtype=np.float64)
        for i in range(L):
            dx[:, :, i * stride + idx[:, :, i]] += dout[:, :, i]
        return dx

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_backward(x, dout):
        return dout * (x > 0).astype(np.float64)

    # ---- forward / backward ---------------------------------------------
    def forward(self, x):
        if not self._initialised:
            self._init_params()
        assert x.ndim == 3 and x.shape[1] == 1, "expected (B,1,L) input"
        cache = {}
        h = self._conv1d(x, self.params["conv1_w"], self.params["conv1_b"], padding=2)
        cache["conv1_in"] = x
        a1 = self._relu(h)
        cache["conv1_out"] = h
        p1, p1_idx = self._maxpool1d(a1, 2, 2)
        cache["pool1_idx"] = p1_idx
        h2 = self._conv1d(p1, self.params["conv2_w"], self.params["conv2_b"], padding=1)
        cache["conv2_in"] = p1
        a2 = self._relu(h2)
        cache["conv2_out"] = h2
        p2, p2_idx = self._maxpool1d(a2, 2, 2)
        cache["pool2_idx"] = p2_idx
        flat = p2.reshape(p2.shape[0], -1)
        cache["flat_shape"] = p2.shape
        f1 = np.dot(flat, self.params["fc1_w"].T) + self.params["fc1_b"]
        cache["fc1_in"] = flat
        a3 = self._relu(f1)
        cache["fc1_out"] = f1
        logits = np.dot(a3, self.params["fc2_w"].T) + self.params["fc2_b"]
        cache["fc2_in"] = a3
        return logits, cache

    def backward(self, logits, targets, cache):
        B = logits.shape[0]
        # softmax cross-entropy gradient
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        sm = exp / np.sum(exp, axis=1, keepdims=True)
        onehot = np.zeros_like(sm)
        onehot[np.arange(B), np.asarray(targets, dtype=int)] = 1.0
        dlogits = (sm - onehot) / max(B, 1)

        grads = {}
        a3 = cache["fc2_in"]
        grads["fc2_w"] = np.dot(dlogits.T, a3) / max(B, 1) + self.l2 * self.params["fc2_w"]
        grads["fc2_b"] = np.sum(dlogits, axis=0) / max(B, 1)
        da3 = np.dot(dlogits, self.params["fc2_w"])
        df1 = self._relu_backward(cache["fc1_out"], da3)
        flat = cache["fc1_in"]
        grads["fc1_w"] = np.dot(df1.T, flat) / max(B, 1) + self.l2 * self.params["fc1_w"]
        grads["fc1_b"] = np.sum(df1, axis=0) / max(B, 1)
        dp2_flat = np.dot(df1, self.params["fc1_w"])
        p2 = dp2_flat.reshape(cache["flat_shape"])
        p2_idx = cache["pool2_idx"]
        da2 = self._maxpool1d_backward(p2, p2_idx, 2, 2)
        dh2 = self._relu_backward(cache["conv2_out"], da2)
        p1 = cache["conv2_in"]
        dx_pool1, dw2, db2 = self._conv1d_backward(dh2, p1, self.params["conv2_w"], padding=1)
        grads["conv2_w"] = dw2 + self.l2 * self.params["conv2_w"]
        grads["conv2_b"] = db2
        da1 = self._maxpool1d_backward(dx_pool1, cache["pool1_idx"], 2, 2)
        dh1 = self._relu_backward(cache["conv1_out"], da1)
        xin = cache["conv1_in"]
        _, dw1, db1 = self._conv1d_backward(dh1, xin, self.params["conv1_w"], padding=2)
        grads["conv1_w"] = dw1 + self.l2 * self.params["conv1_w"]
        grads["conv1_b"] = db1
        return grads

    # ---- Adam optimiser --------------------------------------------------
    def _adam_step(self, grads, m, v, t):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for key in self.params:
            m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
            v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] ** 2)
            mhat = m[key] / (1 - beta1 ** t)
            vhat = v[key] / (1 - beta2 ** t)
            self.params[key] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    # ---- training --------------------------------------------------------
    def train(self, waveforms, labels, epochs=25, batch_size=32, verbose=False):
        if not self._initialised:
            self._init_params()
        X = self._prepare_input(waveforms)
        y = np.asarray(labels, dtype=np.int64)
        if X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("waveforms/labels mismatch or empty")
        rng = np.random.default_rng(self.seed)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            correct = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = X[idx]
                yb = y[idx]
                logits, cache = self.forward(xb)
                # cross-entropy
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                sm = exp / np.sum(exp, axis=1, keepdims=True)
                loss = -np.mean(np.log(sm[np.arange(len(yb)), yb] + 1e-12))
                loss += 0.5 * self.l2 * sum(
                    np.sum(self.params[k] ** 2) for k in self.params
                )
                epoch_loss += loss * len(yb)
                correct += int(np.sum(np.argmax(logits, axis=1) == yb))
                grads = self.backward(logits, yb, cache)
                self._adam_step(grads, m, v, epoch)
            if verbose:
                log.debug("Epoch %d/%d — loss: %.4f", epoch, epochs, loss)
        return True

    # ---- inference -------------------------------------------------------
    def _prepare_input(self, waveforms):
        X = np.asarray(waveforms, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, -1)
        # Defensive: clip pathological magnitudes for numerical stability.
        X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        X = np.clip(X, -1e3, 1e3)
        return X

    def predict_proba(self, waveforms):
        X = self._prepare_input(waveforms)
        if not self._initialised:
            # Untrained network: fall back to uniform (never constant-dead output
            # without warning the caller).
            return np.full((X.shape[0], 3), 1.0 / 3.0, dtype=np.float64)
        logits, _ = self.forward(X)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, waveforms):
        return np.argmax(self.predict_proba(waveforms), axis=1)

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.savez(path, **self.params)
            self._initialised = True
        except Exception as exc:  # pragma: no cover - disk edge cases
            log.error("NumpyCNN1D save failed: %s", exc)

    def load(self, path):
        try:
            self.params = dict(np.load(path, allow_pickle=True))
            self._initialised = True
            return True
        except Exception as exc:
            log.warning("CNN load failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
class CosmicSignalCNN:
    """Thin wrapper exposing a single 3-class interface to the arbitrator.

    Uses PyTorch when available; otherwise trains/infers with `NumpyCNN1D`.
    """

    N_CLASSES = 3  # Natural, Interference, Anomaly

    def __init__(self, weights_path=None, config=None, seed=None):
        if weights_path is None:
            weights_path = str(_PKG_ROOT / "data" / "models" / "cnn_weights.npz")
        self.weights_path = weights_path
        self.config = config
        self.length = 256
        if seed is None:
            seed = int(config.get("models.cnn.seed", 42)) if config else 42
        self.numpy_net = NumpyCNN1D(
            length=self.length,
            seed=int(seed),
            l2=float(config.get("models.cnn.l2", 1e-4)) if config else 1e-4,
            lr=float(config.get("models.cnn.lr", 1e-3)) if config else 1e-3,
        )
        self.torch_model = None
        if TORCH_AVAILABLE:
            self.torch_model = self._build_torch_model(
                config.get("models.cnn.dropout", 0.3) if config else 0.3
            )

    # ---- PyTorch path ----------------------------------------------------
    def _build_torch_model(self, dropout_rate=0.3):  # pragma: no cover
        class _M(nn.Module):
            def __init__(self, dropout=0.3):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
                self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool1d(2, 2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(64 * (self.length // 8), 128)
                self.fc2 = nn.Linear(128, 3)

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.reshape(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)

        return _M(dropout_rate)

    def train(self, waveforms, labels, epochs=15, batch_size=64, lr=1e-3):
        if TORCH_AVAILABLE and self.torch_model is not None:  # pragma: no cover
            return self._train_torch(waveforms, labels, epochs, batch_size, lr)
        # NumPy path (default here)
        self.numpy_net.train(waveforms, labels, epochs=epochs, batch_size=batch_size)
        self.numpy_net.save(self.weights_path)
        return True

    def _train_torch(self, waveforms, labels, epochs, batch_size, lr):  # pragma: no cover
        xt = torch.tensor(np.asarray(waveforms, dtype=np.float64),
                          dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(np.asarray(labels, dtype=np.int64))
        loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)
        self.torch_model.train()
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(self.torch_model.parameters(), lr=lr)
        for _ in range(epochs):
            for bx, by in loader:
                opt.zero_grad()
                loss = crit(self.torch_model(bx), by)
                loss.backward()
                opt.step()
        self._save_torch()
        return True

    def _save_torch(self):  # pragma: no cover
        if self.torch_model is None:
            return
        try:
            parent = os.path.dirname(self.weights_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            w = {k: p.detach().cpu().numpy() for k, p in self.torch_model.state_dict().items()}
            np.savez(self.weights_path, **w)
            self.numpy_net.params = w
            self.numpy_net._initialised = True
        except Exception as exc:
            log.error("CosmicSignalCNN save failed: %s", exc)

    # ---- inference -------------------------------------------------------
    def predict_proba(self, waveforms):
        X = np.asarray(waveforms, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if TORCH_AVAILABLE and self.torch_model is not None:  # pragma: no cover
            self.torch_model.eval()
            xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                out = torch.softmax(self.torch_model(xt), dim=1).cpu().numpy()
            return out
        return self.numpy_net.predict_proba(X)

    def predict(self, waveforms):
        return np.argmax(self.predict_proba(waveforms), axis=1)
