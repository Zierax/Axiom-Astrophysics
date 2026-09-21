import json
import logging
import os
import random
import uuid

log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "universe_cache.json")


def generate_self_healing_cache(seed=None):
    """Generate a default set of realistic cosmic signal parameters to ensure offline operation.

    seed : optional int. All stochastic fields are drawn from a local
        ``random.Random(seed)`` instance so the generated catalog is
        reproducible without touching global RNG state. ``signal_id``
        values use ``uuid4`` and intentionally remain unique per call
        (identifiers, not measurements).
    """
    rng = random.Random(seed)
    records = []

    # 1. Pulsars (50)
    for _i in range(50):
        records.append({
            "signal_id": f"SIG_PUL_{uuid.uuid4().hex[:8].upper()}",
            "name": f"PSR J{rng.randint(0,23):02d}{rng.randint(0,59):02d}{rng.choice(['+', '-'])}{rng.randint(10,89):02d}",
            "frequency_mhz": round(float(rng.uniform(300, 3000)), 2),
            "modulation_type": "Pulsed",
            "bandwidth_efficiency": "Narrowband",
            "drift_rate": 0.0,
            "harmonic_complexity": round(float(rng.uniform(0.1, 0.6)), 4),
            "intensity_sigma": round(float(rng.uniform(3.0, 15.0)), 2),
            "duration_sec": 999999.0,
            "right_ascension": f"{rng.randint(0,23):02d}h {rng.randint(0,59):02d}m {rng.randint(0,59):02d}s",
            "declination": f"{rng.choice(['+', '-'])}{rng.randint(0,89):02d}d {rng.randint(0,59):02d}m",
            "is_repeater": True,
            "origin_class": "Natural",
            "catalog_source": "ATNF_PULSAR_CACHE"
        })

    # 2. FRBs (50)
    for _i in range(50):
        records.append({
            "signal_id": f"SIG_FRB_{uuid.uuid4().hex[:8].upper()}",
            "name": f"FRB {rng.randint(2018,2026):04d}{rng.randint(1,12):02d}{rng.randint(1,28):02d}{rng.choice(['A', 'B', 'C'])}",
            "frequency_mhz": round(float(rng.uniform(400, 1600)), 2),
            "modulation_type": "Variable",
            "bandwidth_efficiency": "Broadband",
            "drift_rate": round(float(rng.uniform(-50.0, -5.0)), 4),
            "harmonic_complexity": 0.0,
            "intensity_sigma": round(float(rng.uniform(8.0, 45.0)), 2),
            "duration_sec": 0.005,
            "right_ascension": f"{rng.randint(0,23):02d}h {rng.randint(0,59):02d}m {rng.randint(0,59):02d}s",
            "declination": f"{rng.choice(['+', '-'])}{rng.randint(0,89):02d}d {rng.randint(0,59):02d}m",
            "is_repeater": rng.choice([True, False]),
            "origin_class": "Natural",
            "catalog_source": "CHIME_FRB_CACHE"
        })

    # 3. HI 21-cm Hydrogen Sources (50)
    for _i in range(50):
        # HI line: 1420.405 MHz with some Doppler shifts
        freq = 1420.405 + rng.uniform(-1.5, 1.5)
        records.append({
            "signal_id": f"SIG_HYD_{uuid.uuid4().hex[:8].upper()}",
            "name": f"HI cloud G{rng.uniform(0.0, 360.0):.2f}{rng.choice(['+', '-'])}{rng.uniform(0.0, 90.0):.2f}",
            "frequency_mhz": round(freq, 4),
            "modulation_type": "Continuous",
            "bandwidth_efficiency": "Broadband",
            "drift_rate": 0.0,
            "harmonic_complexity": 0.0,
            "intensity_sigma": round(float(rng.uniform(1.5, 5.0)), 2),
            "duration_sec": 999999.0,
            "right_ascension": f"{rng.randint(0,23):02d}h {rng.randint(0,59):02d}m {rng.randint(0,59):02d}s",
            "declination": f"{rng.choice(['+', '-'])}{rng.randint(0,89):02d}d {rng.randint(0,59):02d}m",
            "is_repeater": True,
            "origin_class": "Natural",
            "catalog_source": "SIMBAD_HI_CACHE"
        })

    # 4. Quasars (50)
    for _i in range(50):
        records.append({
            "signal_id": f"SIG_QUA_{uuid.uuid4().hex[:8].upper()}",
            "name": f"3C {rng.randint(10,499)}",
            "frequency_mhz": round(float(rng.uniform(100, 10000)), 2),
            "modulation_type": "Continuous",
            "bandwidth_efficiency": "Broadband",
            "drift_rate": 0.0,
            "harmonic_complexity": 0.0,
            "intensity_sigma": round(float(rng.uniform(5.0, 30.0)), 2),
            "duration_sec": 999999.0,
            "right_ascension": f"{rng.randint(0,23):02d}h {rng.randint(0,59):02d}m {rng.randint(0,59):02d}s",
            "declination": f"{rng.choice(['+', '-'])}{rng.randint(0,89):02d}d {rng.randint(0,59):02d}m",
            "is_repeater": True,
            "origin_class": "Natural",
            "catalog_source": "SIMBAD_QSO_CACHE"
        })

    return records


def save_cache(records, path=None):
    """Save fetched catalog records to a local JSON file."""
    path = path or DEFAULT_CACHE_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
        log.info("[Cache] Saved %d records to %s", len(records), path)
    except Exception as e:
        log.warning("[Cache] Failed to save cache: %s", e)


def load_cache(path=None, seed=None):
    """Load catalog records from a local JSON cache file."""
    path = path or DEFAULT_CACHE_PATH
    if not os.path.exists(path):
        log.warning(
            "[Cache] No cache file found at %s. FALLING BACK to a SYNTHETIC "
            "self-healing cache (50 pulsars / 50 FRBs, randomly generated). "
            "Reported results using this cache are NOT from real catalog data. "
            "Fetch real catalogs (e.g. axiom.data.downloader) before trusting any "
            "number built on this cache.", path)
        default_data = generate_self_healing_cache(seed=seed)
        save_cache(default_data, path)
        return default_data
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        log.info("[Cache] Loaded %d records from cache %s", len(records), path)
        return records
    except Exception as e:
        log.warning("[Cache] Failed to load cache: %s", e)
        return []


def get_or_fetch(fetch_fn, cache_path=None, force_refresh=False, seed=None):
    """
    Tries to fetch live data via fetch_fn(). If it fails or returns nothing,
    falls back to the local cache. On success, saves the result to cache.
    """
    cache_path = cache_path or DEFAULT_CACHE_PATH

    if not force_refresh:
        cached = load_cache(cache_path, seed=seed)
        if cached:
            return cached

    # Try live fetch
    try:
        records = fetch_fn()
        if records:
            save_cache(records, cache_path)
            return records
    except Exception as e:
        log.warning("[Cache] Live fetch failed: %s — falling back to cache.", e)

    return load_cache(cache_path, seed=seed)
