import logging
import os
import urllib.request
import zipfile

import numpy as np

log = logging.getLogger(__name__)

HTRU2_ZIP_URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
HTRU2_CSV_FILENAME = "HTRU_2.csv"
# Pinned SHA-256 of the canonical HTRU_2.csv (17,898 rows, 9 cols) so results are
# reproducible from a fixed byte stream. Computed from the real UCI release present
# in this repo at data/HTRU_2.csv.
HTRU2_SHA256 = "b13b4d8929e96ecd196e464c1c8a454c3ac2ffa631015f6388957531a9923f59"

HTRU2_COLUMNS = [
    "profile_mean",
    "profile_std",
    "profile_kurtosis",
    "profile_skewness",
    "dmsnr_mean",
    "dmsnr_std",
    "dmsnr_kurtosis",
    "dmsnr_skewness",
]


def _find_csv(dest_dir):
    """Locate HTRU_2.csv in dest_dir or the package data directory."""
    candidates = [
        os.path.join(dest_dir, HTRU2_CSV_FILENAME),
        os.path.join(os.path.dirname(__file__), HTRU2_CSV_FILENAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def download_and_extract_htru2(dest_dir="data"):
    """
    Downloads the HTRU2 dataset from UCI ML Repository and extracts HTRU_2.csv.
    Returns the absolute path to the CSV file.
    Raises RuntimeError if download or extraction fails.
    """
    os.makedirs(dest_dir, exist_ok=True)
    existing = _find_csv(dest_dir)
    if existing is not None:
        _verify_htru2_sha256(existing)
        log.info("[Data] HTRU_2.csv found at %s", existing)
        return existing

    zip_path = os.path.join(dest_dir, "htru2.zip")
    log.info("[Data] Downloading HTRU2 from %s ...", HTRU2_ZIP_URL)

    try:
        req = urllib.request.Request(
            HTRU2_ZIP_URL,
            headers={"User-Agent": "AxiomAstro/2.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())
        log.info("[Data] Download complete (%d bytes).", os.path.getsize(zip_path))
    except Exception as exc:
        raise RuntimeError(f"HTRU2 download failed: {exc}") from exc

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        log.info("[Data] Extraction complete.")
    except Exception as exc:
        raise RuntimeError(f"HTRU2 zip extraction failed: {exc}") from exc
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    csv_path = os.path.join(dest_dir, HTRU2_CSV_FILENAME)
    if not os.path.exists(csv_path):
        raise RuntimeError("HTRU_2.csv not found after extraction.")
    _verify_htru2_sha256(csv_path)
    return csv_path


def _verify_htru2_sha256(csv_path: str) -> None:
    """Verify HTRU_2.csv integrity against the pinned SHA-256.

    Warns (does not crash) on mismatch so an environment with a different-but-real
    HTRU2 copy still runs, while a corrupted/truncated file is surfaced loudly.
    """
    if HTRU2_SHA256 is None:
        return
    import hashlib
    h = hashlib.sha256()
    try:
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return
    actual = h.hexdigest()
    if actual != HTRU2_SHA256:
        log.warning(
            "[Data] HTRU_2.csv SHA-256 mismatch (got %s, expected %s). "
            "Results may not be reproducible against the pinned release.",
            actual, HTRU2_SHA256,
        )


def load_htru2(dest_dir="data"):
    """
    Load the full HTRU2 dataset. Returns (features, labels, column_names).
    features: ndarray shape (17898, 8) float64
    labels:   ndarray shape (17898,) int64 — 0=RFI/noise, 1=pulsar
    """
    csv_path = download_and_extract_htru2(dest_dir)

    try:
        raw = np.loadtxt(csv_path, delimiter=",")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {csv_path}: {exc}") from exc

    if raw.shape[1] != 9:
        raise ValueError(
            f"Expected 9 columns in HTRU_2.csv, got {raw.shape[1]}. File may be corrupted."
        )

    features = raw[:, :8].astype(np.float64)
    labels = raw[:, 8].astype(np.int64)

    n_pulsar = int(np.sum(labels == 1))
    n_rfi = int(np.sum(labels == 0))
    log.info(
        "[Data] HTRU2 loaded: %d total (%d pulsars, %d RFI/noise)",
        len(labels), n_pulsar, n_rfi,
    )

    # Sanity checks
    if n_pulsar < 1000 or n_rfi < 10000:
        log.warning("[Data] Unexpected class distribution — file may be incomplete.")

    return features, labels, HTRU2_COLUMNS


def split_htru2(features, labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Stratified train / validation / test split.
    Returns dict with keys 'train', 'val', 'test', each containing (X, y).
    """
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: train vs val
    val_fraction_of_trainval = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=seed,
    )

    log.info(
        "[Data] Split: train=%d, val=%d, test=%d",
        len(y_train), len(y_val), len(y_test),
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
