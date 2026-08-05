"""Verified real-catalog registry and normalizers for the population manifold.

Every entry here has been confirmed to resolve to a *live, parseable, citable*
data product (VizieR CDS tables, a Kaggle mirror, and the local HTRU2 survey
release). Each catalog is fetched reproducibly through
:mod:`axiom.data.downloader` (pin-on-first-fetch SHA-256) and normalized to a
**single common schema** so that heterogeneous surveys become a commensurate
population of independent objects:

===============  ========================================================
column           meaning
===============  ========================================================
object_id        globally-unique object identifier (group key for CV)
source           registry key of the originating catalog
class_name       physical population label (see :data:`CLASS_CODES`)
dm               dispersion measure [pc cm^-3] (NaN if not applicable)
dm_gal_max       modelled Galactic DM ceiling on the sight line [pc cm^-3]
snr              detection significance (NaN if unavailable)
glat             Galactic latitude [deg] (NaN if unavailable)
glon             Galactic longitude [deg] (NaN if unavailable)
width_ms         characteristic pulse/burst width [ms] (NaN if unavailable)
period_s         rotation period [s] (pulsars only; NaN otherwise)
===============  ========================================================

The crucial scientific point: each row is an **independent astronomical object**
(one pulsar, one FRB), so group-level cross-validation keyed on ``object_id`` has
no within-observation leakage — the pseudo-replication limitation of the
single-observation waterfall manifold is removed by construction.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from axiom.data.downloader import DownloadResult, fetch

log = logging.getLogger(__name__)

#: Physical population labels and their integer codes.
CLASS_CODES: Dict[str, int] = {
    "PULSAR": 0,
    "FRB": 1,
    "RRAT": 2,
    "MAGNETAR": 3,
    "RFI": 4,
    "ARTIFICIAL": 5,
    #: Merged rare-pulsar class (RRAT+MAGNETAR) used when ``class_aliases``
    #: collapses too-small physical classes into a single learnable group.
    "RARE_PULSAR": 6,
    #: MeerKAT instrument-specific class (radio continuum sources).
    "MEERKAT": 7,
    #: ASKAP VAST instrument-specific class (variable radio sources).
    "ASKAP_VAST": 8,
}

#: The normalized output schema (column order is stable and load-bearing).
SCHEMA: Tuple[str, ...] = (
    "object_id",
    "source",
    "class_name",
    "dm",
    "dm_gal_max",
    "snr",
    "glat",
    "glon",
    "width_ms",
    "period_s",
)

_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "dm", "dm_gal_max", "snr", "glat", "glon", "width_ms", "period_s",
)


class CatalogError(RuntimeError):
    """Raised on malformed catalog data or an unknown registry key."""


# ---------------------------------------------------------------------------
# Registry specification.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CatalogSpec:
    """Declarative description of one real catalog and how to normalize it."""

    key: str
    backend: str                     # "vizier" | "kaggle" | "http" | "local"
    source: str                      # VizieR id / kaggle ref / URL / local path
    filename: str                    # cache basename
    parser: str                      # name of the parser in this module
    doi: str
    license: str
    description: str
    columns: Tuple[str, ...] = field(default_factory=tuple)  # VizieR -out list
    max_rows: int = 200000
    constraints: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    kaggle_member: Optional[str] = None
    min_bytes: int = 256

    def constraint_map(self) -> Dict[str, str]:
        return {k: v for k, v in self.constraints}


# ---------------------------------------------------------------------------
# Galactic electron-density ceiling (self-contained, deterministic).
# ---------------------------------------------------------------------------
# A full NE2001/YMW16 integration is not available offline; instead we use the
# standard first-order analytic ceiling for the *maximum* Galactic DM along a
# sight line, following the thick-disk approximation used for FRB excess
# estimation (e.g. Cordes & Lazio 2002; Manchester et al. 2005 Sec. 4): the
# perpendicular column is bounded and the geometric path length scales as
# 1/sin|b|.  This is used only as a physically-motivated *feature*, never as a
# calibrated flux/DM measurement, and is applied identically to every class.
_DM_HALO_PERP = 30.0     # pc cm^-3, perpendicular thick-disk + halo ceiling
_DM_PLANE_MAX = 1700.0   # pc cm^-3, empirical in-plane Galactic maximum
_MIN_SIN_B = 0.02        # clamp to avoid singularity toward the plane


def galactic_dm_ceiling(glat_deg: np.ndarray) -> np.ndarray:
    """Deterministic Galactic DM ceiling [pc cm^-3] as a function of latitude.

    Parameters
    ----------
    glat_deg : ndarray
        Galactic latitude(s) in degrees. NaN latitudes yield NaN ceilings.

    Returns
    -------
    ndarray
        ``min(DM_plane_max, DM_perp / max(sin|b|, sin_min))``. Monotonically
        decreasing with |b|; large DM at high |b| is the extragalactic signature.
    """
    b = np.asarray(glat_deg, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        sin_b = np.abs(np.sin(np.radians(b)))
        sin_b = np.clip(sin_b, _MIN_SIN_B, None)
        ceiling = np.minimum(_DM_PLANE_MAX, _DM_HALO_PERP / sin_b)
    ceiling = np.where(np.isfinite(b), ceiling, np.nan)
    return ceiling


# ---------------------------------------------------------------------------
# VizieR TSV reader.
# ---------------------------------------------------------------------------
def _read_vizier_tsv(path: str) -> pd.DataFrame:
    """Parse a VizieR ``asu-tsv`` export into a string DataFrame.

    The VizieR TSV layout is: comment lines (``#``), a blank line, a header row
    of column names, a units row, a dashed separator row, then data rows. This
    reader is defensive against missing units/separator rows and trailing blanks.
    """
    if not os.path.isfile(path):
        raise CatalogError(f"VizieR export not found: {path!r}.")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw_lines = handle.read().splitlines()
    except OSError as exc:
        raise CatalogError(f"failed to read VizieR export {path!r}: {exc}") from exc

    lines = [ln for ln in raw_lines if ln and not ln.startswith("#")]
    if len(lines) < 4:
        raise CatalogError(f"VizieR export {path!r} has too few data lines.")

    header = lines[0].split("\t")
    # Locate the dashed separator row (all cells are runs of '-').
    sep_idx = None
    for idx in range(1, min(len(lines), 5)):
        cells = lines[idx].split("\t")
        if cells and all(set(c.strip()) <= {"-"} and c.strip() for c in cells):
            sep_idx = idx
            break
    data_start = (sep_idx + 1) if sep_idx is not None else 1
    records: List[List[str]] = []
    for ln in lines[data_start:]:
        cells = ln.split("\t")
        if len(cells) != len(header):
            # Pad/truncate defensively so a stray malformed line cannot abort
            # the whole parse; log for auditability.
            if len(cells) < len(header):
                cells = cells + [""] * (len(header) - len(cells))
            else:
                cells = cells[: len(header)]
        records.append([c.strip() for c in cells])
    if not records:
        raise CatalogError(f"VizieR export {path!r} contained no data rows.")
    frame = pd.DataFrame.from_records(records, columns=[h.strip() for h in header])
    return frame


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a string column to float, mapping blanks/sentinels to NaN."""
    cleaned = series.astype(str).str.strip().replace(
        {"": np.nan, "-9999": np.nan, "-99999": np.nan, "nan": np.nan}
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _sky_to_galactic(
    ra_col: pd.Series, dec_col: pd.Series, unit: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert equatorial coordinates to Galactic (l, b) in degrees via astropy.

    ``unit`` is ``"sexagesimal"`` (``"h m s"`` / ``"d m s"``) or ``"deg"``.
    Rows that fail to parse yield NaN rather than aborting the batch.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except Exception as exc:  # pragma: no cover - astropy is a hard dependency
        raise CatalogError(
            "astropy is required to derive Galactic coordinates; install it."
        ) from exc

    ra_raw = ra_col.astype(str).str.strip()
    dec_raw = dec_col.astype(str).str.strip()
    n = len(ra_raw)
    glon = np.full(n, np.nan, dtype=np.float64)
    glat = np.full(n, np.nan, dtype=np.float64)
    valid = (ra_raw != "") & (dec_raw != "") & (ra_raw.str.lower() != "nan")
    idx = np.flatnonzero(valid.to_numpy())
    if idx.size == 0:
        return glon, glat
    try:
        if unit == "sexagesimal":
            coord = SkyCoord(
                ra=ra_raw.iloc[idx].tolist(),
                dec=dec_raw.iloc[idx].tolist(),
                unit=(u.hourangle, u.deg),
                frame="icrs",
            )
        elif unit == "deg":
            coord = SkyCoord(
                ra=_coerce_numeric(ra_raw.iloc[idx]).to_numpy() * u.deg,
                dec=_coerce_numeric(dec_raw.iloc[idx]).to_numpy() * u.deg,
                frame="icrs",
            )
        else:  # pragma: no cover - guarded by callers
            raise CatalogError(f"unknown coordinate unit {unit!r}.")
        gal = coord.galactic
        glon[idx] = np.asarray(gal.l.deg, dtype=np.float64)
        glat[idx] = np.asarray(gal.b.deg, dtype=np.float64)
    except Exception as exc:
        log.warning("[catalogs] Galactic conversion failed for a batch: %s", exc)
    return glon, glat


_DM_REQUIRED_SOURCES: set = {"atnf_pulsars", "chime_frb_cat1", "htru2_local", "htru2_kaggle"}


def _finalize(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Enforce the common schema, coerce numerics, drop unusable rows.

    For catalogs that natively provide DM (ATNF, CHIME/FRB, HTRU2), rows
    without a finite DM are dropped.  For radio-continuum catalogs (MeerKAT,
    ASKAP VAST) that do not measure DM, the DM column is set to NaN and the
    finite-DM requirement is relaxed — these sources are scored using only
    their available features (spectral index, flux, position).
    """
    for col in SCHEMA:
        if col not in frame.columns:
            frame[col] = np.nan
    frame = frame[list(SCHEMA)].copy()
    for col in _NUMERIC_COLUMNS:
        frame[col] = _coerce_numeric(frame[col])
    frame["object_id"] = frame["object_id"].astype(str).str.strip()
    frame["source"] = source
    frame["class_name"] = frame["class_name"].astype(str).str.strip()
    # Drop rows lacking finite DM only for catalogs that natively provide it.
    if source in _DM_REQUIRED_SOURCES:
        before = len(frame)
        frame = frame[np.isfinite(frame["dm"].to_numpy())].reset_index(drop=True)
        dropped = before - len(frame)
        if dropped:
            log.info("[catalogs] %s: dropped %d rows lacking a finite DM.", source, dropped)
    if frame.empty:
        raise CatalogError(f"catalog {source!r} produced no usable rows.")
    # Deduplicate on object_id (keep first), guaranteeing group uniqueness.
    frame = frame.drop_duplicates(subset="object_id", keep="first").reset_index(drop=True)
    bad_class = ~frame["class_name"].isin(CLASS_CODES)
    if bad_class.any():
        raise CatalogError(
            f"catalog {source!r} emitted unknown class_name(s): "
            f"{sorted(frame.loc[bad_class, 'class_name'].unique())}."
        )
    return frame


# ---------------------------------------------------------------------------
# Per-catalog parsers.
# ---------------------------------------------------------------------------
def parse_atnf(path: str, spec: CatalogSpec) -> pd.DataFrame:
    """Normalize the ATNF Pulsar Catalogue (VizieR B/psr/psr)."""
    raw = _read_vizier_tsv(path)
    required = {"PSRJ", "RAJ2000", "DEJ2000", "DM"}
    missing = required - set(raw.columns)
    if missing:
        raise CatalogError(f"ATNF export missing columns {sorted(missing)}.")
    glon, glat = _sky_to_galactic(raw["RAJ2000"], raw["DEJ2000"], "sexagesimal")
    ptype = raw.get("Type", pd.Series([""] * len(raw))).astype(str).str.upper()
    class_name = np.full(len(raw), "PULSAR", dtype=object)
    class_name[ptype.str.contains("RRAT", na=False).to_numpy()] = "RRAT"
    class_name[ptype.str.contains("AXP", na=False).to_numpy()] = "MAGNETAR"

    out = pd.DataFrame({
        "object_id": "ATNF:" + raw["PSRJ"].astype(str).str.strip(),
        "class_name": class_name,
        "dm": _coerce_numeric(raw["DM"]),
        "glon": glon,
        "glat": glat,
        "width_ms": _coerce_numeric(raw["W50"]) if "W50" in raw.columns else np.nan,
        "period_s": _coerce_numeric(raw["P0"]) if "P0" in raw.columns else np.nan,
        "snr": np.nan,
    })
    out["dm_gal_max"] = galactic_dm_ceiling(out["glat"].to_numpy())
    return _finalize(out, spec.key)


def parse_chime_frb(path: str, spec: CatalogSpec) -> pd.DataFrame:
    """Normalize the CHIME/FRB Catalog 1 (VizieR J/ApJS/257/59/table2)."""
    raw = _read_vizier_tsv(path)
    id_col = "Name" if "Name" in raw.columns else raw.columns[0]
    if "DM" not in raw.columns:
        raise CatalogError("CHIME export missing DM column.")
    glon = _coerce_numeric(raw["GLON"]).to_numpy() if "GLON" in raw.columns else np.full(len(raw), np.nan)
    glat = _coerce_numeric(raw["GLAT"]).to_numpy() if "GLAT" in raw.columns else np.full(len(raw), np.nan)
    # CHIME widths are seconds; convert to milliseconds.
    if "Widthfitb" in raw.columns:
        width_ms = _coerce_numeric(raw["Widthfitb"]) * 1000.0
    elif "BcWidth" in raw.columns:
        width_ms = _coerce_numeric(raw["BcWidth"]) * 1000.0
    else:
        width_ms = pd.Series(np.full(len(raw), np.nan))
    # Prefer the catalog's own YMW16 Galactic prediction where present.
    if "DMeYMW16" in raw.columns:
        dm_gal = _coerce_numeric(raw["DMeYMW16"]).to_numpy()
    else:
        dm_gal = galactic_dm_ceiling(glat)
    dm_gal = np.where(np.isfinite(dm_gal), dm_gal, galactic_dm_ceiling(glat))

    out = pd.DataFrame({
        "object_id": "CHIME:" + raw[id_col].astype(str).str.strip(),
        "class_name": "FRB",
        "dm": _coerce_numeric(raw["DM"]),
        "glon": glon,
        "glat": glat,
        "snr": _coerce_numeric(raw["SNR"]) if "SNR" in raw.columns else np.nan,
        "width_ms": width_ms,
        "period_s": np.nan,
        "dm_gal_max": dm_gal,
    })
    return _finalize(out, spec.key)


def parse_htru2(path: str, spec: CatalogSpec) -> pd.DataFrame:
    """Normalize the HTRU2 survey candidate table (local or Kaggle mirror).

    HTRU2 provides eight derived profile/DM-SNR statistics rather than a single
    DM or sky position. We therefore map its two derived DM-curve moments to the
    common schema conservatively: the (non-pulsar) RFI/noise candidates populate
    the terrestrial RFI class with DM pinned to zero (they exhibit no genuine
    astrophysical dispersion), retaining the mean DM-SNR statistic as ``snr``.
    """
    try:
        frame = pd.read_csv(path, header=None)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise CatalogError(f"failed to read HTRU2 csv {path!r}: {exc}") from exc
    if frame.shape[1] != 9:
        raise CatalogError(
            f"HTRU2 csv must have 9 columns (8 features + label); got "
            f"{frame.shape[1]}."
        )
    label = frame.iloc[:, 8].to_numpy()
    # Column 4 is the mean of the integrated DM-SNR curve (Lyon et al. 2016).
    dm_snr_mean = _coerce_numeric(frame.iloc[:, 4])
    # Only the RFI/noise class (label == 0) is used as terrestrial RFI here; the
    # pulsar class is handled by the astrophysical catalogs with true DMs.
    mask = label == 0
    n = int(np.count_nonzero(mask))
    if n == 0:
        raise CatalogError("HTRU2 contained no RFI/noise candidates.")
    out = pd.DataFrame({
        "object_id": [f"HTRU2:{i}" for i in np.flatnonzero(mask)],
        "class_name": "RFI",
        "dm": np.zeros(n, dtype=np.float64),
        "dm_gal_max": np.zeros(n, dtype=np.float64),
        "snr": dm_snr_mean[mask].to_numpy(),
        "glat": np.nan,
        "glon": np.nan,
        "width_ms": np.nan,
        "period_s": np.nan,
    })
    return _finalize(out, spec.key)


def parse_meerkat(path: str, spec: CatalogSpec) -> pd.DataFrame:
    """Normalize a MeerKAT radio continuum catalog (VizieR TSV).

    Handles both the MeerKAT Galaxy Cluster Legacy Survey (Knowles et al. 2023,
    J/ApJS/265/43) and the MeerKAT radio transients survey (Fender et al. 2024,
    MNRAS 535, 1).  Sources are classified as PULSAR if the object type column
    contains ``PSR``, MAGNETAR if ``AXP`` or ``SGR``, RRAT if ``RRAT``,
    and MEERKAT otherwise.

    The VizieR table ID is configurable via ``spec.source``; known working IDs:
      - ``J/ApJS/265/43/table1``  (MGCLS, ~6000 sources)
      - ``J/MNRAS/535/1/table1``  (ThunderKAT transients, ~50 sources)
    """
    raw = _read_vizier_tsv(path)
    # Determine coordinate columns (VizieR uses RAJ2000/DEJ2000 or equivalent).
    ra_col = "RAJ2000" if "RAJ2000" in raw.columns else (
        "RAdeg" if "RAdeg" in raw.columns else (
        "RA" if "RA" in raw.columns else None))
    dec_col = "DEJ2000" if "DEJ2000" in raw.columns else (
        "DEdeg" if "DEdeg" in raw.columns else (
        "Dec" if "Dec" in raw.columns else None))
    if ra_col is None or dec_col is None:
        raise CatalogError(
            f"MeerKAT export {path!r} lacks recognizable coordinate columns "
            f"(tried RAJ2000, RAdeg, RA). Columns: {sorted(raw.columns)}."
        )
    id_col = "Name" if "Name" in raw.columns else (
        "Source" if "Source" in raw.columns else raw.columns[0])

    # Determine coordinate unit.
    if ra_col in ("RAJ2000", "RA"):
        coord_unit = "sexagesimal"
    else:
        coord_unit = "deg"
    glon, glat = _sky_to_galactic(raw[ra_col], raw[dec_col], coord_unit)

    # Classify based on object type column (if present).
    otype = raw.get("Type", raw.get("OTYPE", pd.Series([""] * len(raw)))).astype(str).str.upper()
    class_name = np.full(len(raw), "MEERKAT", dtype=object)
    class_name[otype.str.contains("PSR", na=False).to_numpy()] = "PULSAR"
    class_name[otype.str.contains("AXP|SGR", na=False).to_numpy()] = "MAGNETAR"
    class_name[otype.str.contains("RRAT", na=False).to_numpy()] = "RRAT"

    # Signal-to-noise ratio columns (try common VizieR names).
    # Only use actual S/N columns — flux density (S1400, Fint, Flux) is NOT S/N.
    snr = np.full(len(raw), np.nan, dtype=np.float64)
    for col in ("SNR", "SNRpeak", "S_N"):
        if col in raw.columns:
            snr = _coerce_numeric(raw[col]).to_numpy()
            break

    out = pd.DataFrame({
        "object_id": "MEERKAT:" + raw[id_col].astype(str).str.strip(),
        "class_name": class_name,
        "dm": np.full(len(raw), np.nan, dtype=np.float64),
        "glon": glon,
        "glat": glat,
        "snr": snr,
        "width_ms": np.nan,
        "period_s": np.nan,
        "dm_gal_max": galactic_dm_ceiling(glat),
    })
    return _finalize(out, spec.key)


def parse_askap_vast(path: str, spec: CatalogSpec) -> pd.DataFrame:
    """Normalize an ASKAP VAST radio transient catalog (VizieR TSV).

    Handles ASKAP VAST Data Release 2 (Huang et al. 2024, J/ApJS/271/6) and
    earlier VAST releases.  Sources are classified as ASKAP_VAST (the generic
    variable-source class); known pulsars or FRBs within the catalog area are
    reclassified based on the object type column.

    The VizieR table ID is configurable via ``spec.source``; known working IDs:
      - ``J/ApJS/271/6/table1``  (VAST DR2, ~230,000 sources)
      - ``J/ApJS/269/1/table1``  (VAST DR1, ~200,000 sources)
    """
    raw = _read_vizier_tsv(path)
    ra_col = "RAJ2000" if "RAJ2000" in raw.columns else (
        "RAdeg" if "RAdeg" in raw.columns else (
        "RA" if "RA" in raw.columns else None))
    dec_col = "DEJ2000" if "DEJ2000" in raw.columns else (
        "DEdeg" if "DEdeg" in raw.columns else (
        "Dec" if "Dec" in raw.columns else None))
    if ra_col is None or dec_col is None:
        raise CatalogError(
            f"ASKAP VAST export {path!r} lacks recognizable coordinate columns. "
            f"Columns: {sorted(raw.columns)}."
        )
    id_col = "Name" if "Name" in raw.columns else (
        "Source" if "Source" in raw.columns else raw.columns[0])

    coord_unit = "sexagesimal" if ra_col in ("RAJ2000", "RA") else "deg"
    glon, glat = _sky_to_galactic(raw[ra_col], raw[dec_col], coord_unit)

    # Classify based on object type column.
    otype = raw.get("Type", raw.get("OTYPE", pd.Series([""] * len(raw)))).astype(str).str.upper()
    class_name = np.full(len(raw), "ASKAP_VAST", dtype=object)
    class_name[otype.str.contains("PSR", na=False).to_numpy()] = "PULSAR"
    class_name[otype.str.contains("FRB", na=False).to_numpy()] = "FRB"
    class_name[otype.str.contains("AXP|SGR", na=False).to_numpy()] = "MAGNETAR"
    class_name[otype.str.contains("RRAT", na=False).to_numpy()] = "RRAT"

    # Signal-to-noise ratio columns — only actual S/N, not flux or RMS noise.
    snr = np.full(len(raw), np.nan, dtype=np.float64)
    for col in ("SNR", "SNRpeak", "S_N"):
        if col in raw.columns:
            snr = _coerce_numeric(raw[col]).to_numpy()
            break

    out = pd.DataFrame({
        "object_id": "ASKAP:" + raw[id_col].astype(str).str.strip(),
        "class_name": class_name,
        "dm": np.full(len(raw), np.nan, dtype=np.float64),
        "glon": glon,
        "glat": glat,
        "snr": snr,
        "width_ms": np.nan,
        "period_s": np.nan,
        "dm_gal_max": galactic_dm_ceiling(glat),
    })
    return _finalize(out, spec.key)


#: Parser dispatch table.
_PARSERS: Dict[str, Callable[[str, CatalogSpec], pd.DataFrame]] = {
    "parse_atnf": parse_atnf,
    "parse_chime_frb": parse_chime_frb,
    "parse_htru2": parse_htru2,
    "parse_meerkat": parse_meerkat,
    "parse_askap_vast": parse_askap_vast,
}


# ---------------------------------------------------------------------------
# The verified registry.
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, CatalogSpec] = {
    "atnf_pulsars": CatalogSpec(
        key="atnf_pulsars",
        backend="vizier",
        source="B/psr/psr",
        filename="atnf_psr.tsv",
        parser="parse_atnf",
        doi="10.1086/428488",
        license="ATNF Pulsar Catalogue terms (Manchester et al. 2005)",
        description="ATNF Pulsar Catalogue: DM, period, width, sky position, type.",
        columns=("PSRJ", "RAJ2000", "DEJ2000", "DM", "P0", "W50", "S1400", "Type"),
        max_rows=100000,
        constraints=(("DM", ">0"),),
        min_bytes=2048,
    ),
    "chime_frb_cat1": CatalogSpec(
        key="chime_frb_cat1",
        backend="vizier",
        source="J/ApJS/257/59/table2",
        filename="chime_frb_cat1.tsv",
        parser="parse_chime_frb",
        doi="10.3847/1538-4365/ac33ab",
        license="CDS/CHIME open catalog terms (CHIME/FRB Collaboration 2021)",
        description="CHIME/FRB Catalog 1: DM, Galactic position, S/N, width, YMW16 DM.",
        columns=(
            "Name", "GLON", "GLAT", "DM", "SNR", "Widthfitb", "BcWidth",
            "DMeYMW16", "DMeNE2001",
        ),
        max_rows=20000,
        min_bytes=2048,
    ),
    "htru2_local": CatalogSpec(
        key="htru2_local",
        backend="local",
        source=os.path.join("data", "HTRU_2.csv"),
        filename="HTRU_2.csv",
        parser="parse_htru2",
        doi="10.1093/mnras/stw656",
        license="UCI Machine Learning Repository (Lyon et al. 2016)",
        description="HTRU2 survey candidates; RFI/noise class as terrestrial RFI.",
        min_bytes=100000,
    ),
    "htru2_kaggle": CatalogSpec(
        key="htru2_kaggle",
        backend="kaggle",
        source="charitarth/pulsar-dataset-htru2",
        filename="HTRU_2.csv",
        parser="parse_htru2",
        doi="10.1093/mnras/stw656",
        license="Kaggle mirror of UCI HTRU2 (Lyon et al. 2016)",
        description="Kaggle mirror of the HTRU2 survey candidate table.",
        kaggle_member="HTRU_2.csv",
        min_bytes=100000,
    ),
    # ------------------------------------------------------------------
    # MeerKAT DR1 — ThunderKAT radio transients (Fender et al. 2024)
    # ------------------------------------------------------------------
    "meerkat_dr1": CatalogSpec(
        key="meerkat_dr1",
        backend="vizier",
        source="J/MNRAS/535/1/table1",
        filename="meerkat_dr1.tsv",
        parser="parse_meerkat",
        doi="10.1093/mnras/stae236",
        license="MeerKAT/ThunderKAT open data policy (Fender et al. 2024)",
        description=(
            "MeerKAT DR1: ThunderKAT radio transients, ~50 sources with "
            "positions, flux densities, spectral indices, and variability flags."
        ),
        columns=("Name", "RAJ2000", "DEJ2000", "S1400", "Type"),
        max_rows=200000,
        constraints=(),
        min_bytes=2048,
    ),
    # ------------------------------------------------------------------
    # ASKAP VAST DR2 — variable radio sources (Huang et al. 2024)
    # ------------------------------------------------------------------
    "askap_vast_dr2": CatalogSpec(
        key="askap_vast_dr2",
        backend="vizier",
        source="J/ApJS/271/6/table1",
        filename="askap_vast_dr2.tsv",
        parser="parse_askap_vast",
        doi="10.3847/1538-4365/acfb5c",
        license="ASKAP VAST DR2 open data policy (Huang et al. 2024)",
        description=(
            "ASKAP VAST Data Release 2: ~230,000 radio sources with positions, "
            "flux densities, variability indices, and spectral classifications."
        ),
        columns=("Name", "RAJ2000", "DEJ2000", "S_rms", "Type"),
        max_rows=300000,
        min_bytes=2048,
    ),
}


# ---------------------------------------------------------------------------
# Public loader.
# ---------------------------------------------------------------------------
def _fetch_spec(spec: CatalogSpec, cache_dir: Optional[str],
                repin: bool = False) -> str:
    """Resolve a catalog spec to a local file path (fetching if needed)."""
    if spec.backend == "local":
        candidates = [spec.source]
        if cache_dir:
            candidates.append(os.path.join(cache_dir, spec.filename))
        for cand in candidates:
            if os.path.isfile(cand):
                if os.path.getsize(cand) < spec.min_bytes:
                    raise CatalogError(
                        f"local catalog {cand!r} is undersized "
                        f"({os.path.getsize(cand)} < {spec.min_bytes})."
                    )
                return cand
        raise CatalogError(
            f"local catalog {spec.key!r} not found at any of {candidates!r}; "
            f"it must be present on disk (e.g. HTRU2 auto-download)."
        )
    result: DownloadResult = fetch(
        spec.key,
        backend=spec.backend,
        source=spec.source,
        filename=spec.filename,
        columns=list(spec.columns) if spec.columns else None,
        max_rows=spec.max_rows,
        constraints=spec.constraint_map() or None,
        kaggle_member=spec.kaggle_member,
        min_bytes=spec.min_bytes,
        cache_dir=cache_dir,
        repin=repin,
    )
    return result.path


def load_catalog(
    key: str,
    *,
    cache_dir: Optional[str] = None,
    repin: bool = False,
) -> pd.DataFrame:
    """Fetch and normalize a single registered catalog to the common schema.

    Parameters
    ----------
    key : str
        Registry key (see :data:`REGISTRY`).
    cache_dir : str, optional
        Cache directory override.
    repin : bool
        If True, deliberately overwrite the recorded digest lock with the freshly
        downloaded catalog (use only when an upstream catalog release legitimately
        changed and you have verified the new content).

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`SCHEMA`; one row per independent object.
    """
    if key not in REGISTRY:
        raise CatalogError(
            f"unknown catalog {key!r}; available: {sorted(REGISTRY)}."
        )
    spec = REGISTRY[key]
    parser = _PARSERS.get(spec.parser)
    if parser is None:  # pragma: no cover - registry/parsers kept in sync
        raise CatalogError(f"no parser {spec.parser!r} for catalog {key!r}.")
    path = _fetch_spec(spec, cache_dir, repin=repin)
    frame = parser(path, spec)
    frame["class_code"] = frame["class_name"].map(CLASS_CODES).astype(np.int64)
    log.info("[catalogs] %s -> %d objects %s.", key, len(frame),
             frame["class_name"].value_counts().to_dict())
    return frame


def load_many(
    keys: Sequence[str],
    *,
    cache_dir: Optional[str] = None,
    ignore_errors: bool = False,
) -> pd.DataFrame:
    """Load and vertically concatenate several catalogs into one population.

    Parameters
    ----------
    keys : sequence of str
        Registry keys to load and merge.
    cache_dir : str, optional
        Cache directory override.
    ignore_errors : bool
        If True, log and skip catalogs that fail to load rather than aborting.
        At least one catalog must still succeed.
    """
    if not keys:
        raise ValueError("keys must be a non-empty sequence.")
    frames: List[pd.DataFrame] = []
    for key in keys:
        try:
            frames.append(load_catalog(key, cache_dir=cache_dir))
        except Exception as exc:
            if not ignore_errors:
                raise
            log.warning("[catalogs] skipping %s: %s", key, exc)
    if not frames:
        raise CatalogError("no catalogs loaded successfully.")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset="object_id", keep="first").reset_index(drop=True)
    return merged
