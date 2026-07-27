"""Tests for axiom.data.real_loaders (no network required).

The download path is exercised manually; here we validate the deterministic
record builder and manifest so regressions in the real-data OOD augmentation
are caught without hitting Zenodo.
"""
import os

import numpy as np
import pytest

from axiom.data import real_loaders  # noqa: E402  (used by monkeypatch in tests)
from axiom.data.real_loaders import (
    DEFAULT_CACHE,
    VOYAGER_FIL_CACHE,
    RealOODManifest,
    load_voyager_anomaly,
    sample_real_frb_records,
)


def test_sample_real_frb_records_shape_and_roles():
    dm = np.array([134.0, 400.0, 900.0, 2000.0, 3966.0])
    recs = sample_real_frb_records(dm, n=5, seed=42)
    assert len(recs) == 5
    for _name, origin, stype, d, snr, role in recs:
        assert origin == "FRB"
        assert stype == "FRB"
        assert role == "Natural"
        assert d in dm  # real DM must come verbatim from the supplied array
        assert 8.0 <= snr <= 40.0


def test_sample_real_frb_records_deterministic():
    dm = np.linspace(100.0, 1000.0, 50)
    a = sample_real_frb_records(dm, n=10, seed=7)
    b = sample_real_frb_records(dm, n=10, seed=7)
    assert [r[3] for r in a] == [r[3] for r in b]
    assert [r[4] for r in a] == [r[4] for r in b]


def test_load_voyager_anomaly_disabled():
    recs, waves, wf, man = load_voyager_anomaly(0)
    assert recs == [] and waves == {} and wf == {}
    assert man.retrieved_ok is False


_VOYAGER_CACHED = os.path.exists(
    os.path.join(DEFAULT_CACHE, VOYAGER_FIL_CACHE))


def _has_blimpy() -> bool:
    try:
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_VOYAGER_CACHED and _has_blimpy()),
                    reason="Voyager filterbank not cached or blimpy not installed")
def test_load_voyager_anomaly_real_carrier():
    from axiom.stats.chaos import compute_chaos_descriptor

    recs, waves, wf, man = load_voyager_anomaly(25)
    assert man.retrieved_ok is True
    assert len(recs) >= 1
    names = [r[0] for r in recs]
    assert names[0] == "Voyager1_carrier"
    for name, origin, stype, _d, snr, role in recs:
        assert origin == "Narrowband" and stype == "Narrowband"
        assert role == "Anomaly"
        assert snr >= 3.0
        w = waves[name]
        assert w.shape == (256,)
        assert np.isfinite(w).all()
        # Each real carrier carries genuine spectrogram descriptors so the
        # descriptor-conformal detector can measure its actual morphology.
        assert name in wf and wf[name]
        # A real carrier is a highly-ordered narrowband tone: low entropy.
        entropy = float(np.atleast_1d(compute_chaos_descriptor(w))[0])
        assert entropy < 0.6


def test_load_voyager_anomaly_deterministic():
    if not _VOYAGER_CACHED:
        pytest.skip("Voyager filterbank not cached")
    a = load_voyager_anomaly(25)[0]
    b = load_voyager_anomaly(25)[0]
    assert [r[0] for r in a] == [r[0] for r in b]
    assert [r[4] for r in a] == [r[4] for r in b]


def test_manifest_serialises():
    m = RealOODManifest(
        source="x", doi="10.1234/abc", url="https://example/foo",
        retrieved_ok=True, n_real_frb=3369, dm_range=(134.0, 3966.0),
    )
    d = m.to_dict()
    assert d["n_real_frb"] == 3369
    assert d["retrieved_ok"] is True
    assert d["dm_range"] == (134.0, 3966.0)


# ---------------------------------------------------------------------------
# Real SETI / Breakthrough Listen ingestion (Kaggle seti-data).
# ---------------------------------------------------------------------------

def _make_fake_zip(path, n_signals=6):
    """Write a small zip with fabricated GUPPI-style .h5 spectrograms.

    Mirrors the real Breakthrough Listen archive layout: each entry is an HDF5
    file whose ``data`` dataset is a (channels, 1, time) float cube, exactly the
    shape produced by ``spliced_*.gpuspec.h5`` files.
    """
    import io
    import zipfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n_signals):
            # 3-D GUPPI cube (16 freq channels x 1 integration x time) with a
            # narrowband hit on one channel (high variance -> chosen by _as_waveform).
            cube = rng.standard_normal((16, 1, 200)).astype(np.float32) * 0.1
            cube[8, 0, :] += 5.0
            buf = io.BytesIO()
            import h5py
            with h5py.File(buf, "w") as hf:
                hf.create_dataset("data", data=cube)
                hf.create_dataset("mask", data=np.zeros((16, 1, 200), dtype="u1"))
            zf.writestr(f"HIP0000{i}/HIP0000{i}/spliced_guppi_HIP0000{i}_000{i}.gpuspec.0000.h5",
                        buf.getvalue())


def _has_h5py() -> bool:
    try:
        import h5py
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_h5py(), reason="h5py not installed")
def test_real_seti_parse_fake_archive(tmp_path):
    from axiom.data import real_seti

    _make_fake_zip(tmp_path / "seti-data.zip")
    signals = real_seti.parse_seti_archive(str(tmp_path / "seti-data.zip"),
                                            dest=str(tmp_path / "_ext"))
    # All fabricated GUPPI spectrograms parse into (1-D waveform, 2-D spec) pairs.
    assert len(signals) == 6
    for wave, _spec in signals.values():
        assert wave.shape == (real_seti.SIGNAL_LEN,)
        assert np.isfinite(wave).all()


@pytest.mark.skipif(not _has_h5py(), reason="h5py not installed")
def test_real_seti_build_ood_fake_archive(tmp_path):
    from axiom.data import real_seti

    _make_fake_zip(tmp_path / "seti-data.zip")
    recs, waves, wf, man = real_seti.build_real_seti_ood(
        n_anom=5, archive=str(tmp_path / "seti-data.zip"))
    assert man.retrieved_ok is True
    assert len(recs) == 5
    for name, _origin, _stype, dm, snr, role in recs:
        assert role == "Anomaly"
        assert dm == 0.0
        assert snr >= 1.0
        assert name in waves
        assert waves[name].shape == (real_seti.SIGNAL_LEN,)
        # Frequency-resolved descriptors are computed for every real signal.
        assert name in wf
        assert wf[name]  # non-empty dict of descriptors


def test_real_seti_missing_archive_is_graceful(monkeypatch):
    from axiom.data import real_seti

    # Force every lookup to a missing path so no real archive is discovered.
    monkeypatch.setattr(real_seti, "DEFAULT_LOCAL", "/nonexistent/default.zip")
    monkeypatch.setattr(real_loaders, "DEFAULT_CACHE",
                        "/nonexistent/cache")

    # A genuinely-unavailable archive must degrade to empty, never raise.
    signals = real_seti.parse_seti_archive("/nonexistent/path/seti-data.zip")
    assert signals == {}
    recs, waves, wf, man = real_seti.build_real_seti_ood(
        archive="/nonexistent/path/seti-data.zip")
    assert recs == [] and waves == {} and wf == {}
    assert man.retrieved_ok is False


@pytest.mark.skipif(not _has_h5py(), reason="h5py not installed")
def test_real_seti_wired_into_full_builder(tmp_path, monkeypatch):
    """When the real SETI archive is present it is preferred over synthetic tones."""
    from axiom.data.loader import load_htru2

    _make_fake_zip(tmp_path / "seti-data.zip")

    X, y, _ = load_htru2()
    records, real_features, real_waves, waterfall_features, manifest = real_loaders.get_full_real_ood(
        X, y, n_anom=5, n_frb=5, n_rfi=5,
        seti_archive=str(tmp_path / "seti-data.zip"))
    # The only genuine artificial-technosignature ground truth is Voyager; the
    # real SETI stellar observations are held as an Unlabeled discovery pool
    # (they have no confirmed narrowband signal), so they are NOT anomaly-labeled.
    anom = [r for r in records if r[5] == "Anomaly"]
    assert anom and all(r[0].lower().startswith("voyager1") for r in anom)
    unlabeled = [r for r in records if r[5] == "Unlabeled"]
    seti = [r for r in unlabeled if r[0].startswith("SETI__")]
    assert len(seti) >= 1  # real SETI signals were ingested (not all synthetic)
    # Native frequency-resolved descriptors are available for the real signals.
    assert any(s in waterfall_features for s in [r[0] for r in seti])


def test_waterfall_features_narrowband_scores_higher_than_broadband():
    """Native frequency-resolved descriptors must discriminate a tone from noise."""
    from axiom.dsp.waterfall_features import (
        FrequencyResolvedScorer,
        compute_waterfall_features,
        waterfall_narrowband_score,
    )

    rng = np.random.default_rng(0)
    n_chan, n_time = 64, 256
    noise = np.abs(rng.normal(0.0, 1.0, size=(n_chan, n_time))) + 1.0

    # Narrowband tone: a single flat channel above the background.
    narrow = noise.copy()
    narrow[10, :] += 30.0
    # Broadband: raise every channel uniformly (no concentration).
    broad = noise.copy() + 8.0

    f_narrow = compute_waterfall_features(narrow)
    f_broad = compute_waterfall_features(broad)
    for f in (f_narrow, f_broad):
        assert all(np.isfinite(v) for v in f.values())

    assert f_narrow["concentration"] > f_broad["concentration"]
    assert f_narrow["spectral_flatness"] < f_broad["spectral_flatness"]

    scorer = FrequencyResolvedScorer({"n": f_narrow, "b": f_broad})
    assert scorer.score("n") > scorer.score("b")
    assert 0.0 <= waterfall_narrowband_score(f_narrow) <= 1.0
    assert scorer.score("missing") == 0.5  # neutral default
