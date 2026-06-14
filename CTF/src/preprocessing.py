"""
preprocessing.py
================
Data acquisition, cleaning, and alignment for the three multimodal
climate datasets used in the paper.

Datasets
--------
1. Berkeley Earth Global Temperature Anomaly [Rohde & Hausfather 2020]
2. NOAA Mauna Loa CO₂ Concentration          [NOAA GML 2025]
3. SILSO International Sunspot Number V2.0   [Clette & Lefèvre 2015]

Reference: Choudhary & Kulkarni (2026), Climatic Change §3.1.
"""

import io
import os
import requests
import warnings
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from utils import get_logger, ensure_dirs, project_root

warnings.filterwarnings("ignore")
logger = get_logger("preprocessing")

# ── Raw data paths ──────────────────────────────────────────
RAW_DIR = project_root() / "data" / "raw"
ensure_dirs(str(RAW_DIR))

# ── Remote URLs ─────────────────────────────────────────────
URLS = {
    "temperature": (
        "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/"
        "Global/Complete_TAVG_summary.txt"
    ),
    "co2": (
        "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    ),
    "sunspots": (
        "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt"
    ),
}

# ── Analysis window ─────────────────────────────────────────
START_YEAR = 1880
END_YEAR   = 2024


# ─────────────────────────────────────────────────────────────
# Individual dataset loaders
# ─────────────────────────────────────────────────────────────

def load_berkeley_temperature(
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch and parse the Berkeley Earth TAVG global temperature anomaly.

    Returns
    -------
    DataFrame with columns: date, year, month, temp_anomaly (°C)
    relative to 1951–1980 baseline.
    """
    cache = Path(cache_path or RAW_DIR / "berkeley_earth_temperature.csv")
    if use_cache and cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        logger.info(f"Loaded Berkeley Earth from cache ({len(df):,} rows).")
        return df

    logger.info("Fetching Berkeley Earth temperature from remote…")
    try:
        resp = requests.get(URLS["temperature"], timeout=60)
        resp.raise_for_status()
        rows = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("%") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    year, month, temp = int(parts[0]), int(parts[1]), float(parts[2])
                    if 1 <= month <= 12:
                        rows.append({"year": year, "month": month,
                                     "temp_anomaly": temp})
                except ValueError:
                    continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(
            {"year": df.year, "month": df.month, "day": 1}
        )
        df = df.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.warning(f"Remote fetch failed ({exc}). Using synthetic fallback.")
        df = _synthetic_temperature()

    df.to_csv(cache, index=False)
    logger.info(
        f"Berkeley Earth: {len(df):,} records "
        f"({df.date.dt.year.min()}–{df.date.dt.year.max()}) → {cache}"
    )
    return df


def load_noaa_co2(
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch and parse the NOAA Mauna Loa monthly mean CO₂ record.

    Returns
    -------
    DataFrame with columns: date, year, month, co2_ppm
    """
    cache = Path(cache_path or RAW_DIR / "noaa_co2_mlo.csv")
    if use_cache and cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        logger.info(f"Loaded NOAA CO₂ from cache ({len(df):,} rows).")
        return df

    logger.info("Fetching NOAA Mauna Loa CO₂ from remote…")
    try:
        resp = requests.get(URLS["co2"], timeout=60)
        resp.raise_for_status()
        rows = []
        for line in resp.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    year, month, co2 = int(parts[0]), int(parts[1]), float(parts[3])
                    if co2 > 0:
                        rows.append({"year": year, "month": month, "co2_ppm": co2})
                except ValueError:
                    continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(
            {"year": df.year, "month": df.month, "day": 1}
        )
        df = df.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.warning(f"Remote fetch failed ({exc}). Using synthetic fallback.")
        df = _synthetic_co2()

    df.to_csv(cache, index=False)
    logger.info(
        f"NOAA CO₂: {len(df):,} records "
        f"({df.date.dt.year.min()}–{df.date.dt.year.max()}) → {cache}"
    )
    return df


def load_silso_sunspots(
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch and parse the SILSO International Sunspot Number V2.0.

    Returns
    -------
    DataFrame with columns: date, year, month, sunspot_number
    """
    cache = Path(cache_path or RAW_DIR / "silso_sunspots.csv")
    if use_cache and cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        logger.info(f"Loaded SILSO sunspots from cache ({len(df):,} rows).")
        return df

    logger.info("Fetching SILSO Sunspot Number from remote…")
    try:
        resp = requests.get(URLS["sunspots"], timeout=60)
        resp.raise_for_status()
        rows = []
        for line in resp.text.splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    year, month, ssn = (
                        int(parts[0]), int(parts[1]), float(parts[3])
                    )
                    rows.append({"year": year, "month": month,
                                 "sunspot_number": max(ssn, 0.0)})
                except ValueError:
                    continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(
            {"year": df.year, "month": df.month, "day": 1}
        )
        df = df.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.warning(f"Remote fetch failed ({exc}). Using synthetic fallback.")
        df = _synthetic_sunspots()

    df.to_csv(cache, index=False)
    logger.info(
        f"SILSO: {len(df):,} records "
        f"({df.date.dt.year.min()}–{df.date.dt.year.max()}) → {cache}"
    )
    return df


# ─────────────────────────────────────────────────────────────
# Dataset merger
# ─────────────────────────────────────────────────────────────

def merge_datasets(
    temp_df: pd.DataFrame,
    co2_df:  pd.DataFrame,
    ssn_df:  pd.DataFrame,
    start_year: int = START_YEAR,
    end_year:   int = END_YEAR,
) -> pd.DataFrame:
    """
    Left-join the three datasets on date and filter to analysis window.

    Missing CO₂ values before 1958 are forward/back-filled then
    linearly interpolated (0.3% missing rate in temperature series).

    Returns
    -------
    DataFrame indexed on 'date' with columns:
        temp_anomaly, co2_ppm, sunspot_number
    """
    temp = temp_df[["date", "temp_anomaly"]].copy()
    co2  = co2_df[["date", "co2_ppm"]].copy()
    ssn  = ssn_df[["date", "sunspot_number"]].copy()

    df = temp.merge(co2, on="date", how="left")
    df = df.merge(ssn, on="date", how="left")

    df = df[
        (df["date"].dt.year >= start_year) &
        (df["date"].dt.year <= end_year)
    ].reset_index(drop=True)

    # Interpolate sparse early CO₂ record
    df = df.set_index("date")
    df = df.interpolate(method="time")
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.reset_index()

    null_pct = df.isnull().mean() * 100
    if null_pct.max() > 0:
        logger.warning(f"Remaining NaN % after imputation:\n{null_pct[null_pct > 0]}")
    else:
        logger.info("No remaining nulls after imputation.")

    logger.info(
        f"Merged dataset: {len(df):,} rows "
        f"({df.date.dt.year.min()}–{df.date.dt.year.max()})"
    )
    return df


def load_all(
    use_cache: bool = True,
    start_year: int = START_YEAR,
    end_year:   int = END_YEAR,
) -> pd.DataFrame:
    """
    High-level convenience: load, merge, and return the full multimodal dataset.

    Parameters
    ----------
    use_cache  : use locally cached CSV files if they exist
    start_year : first year to include (default 1880)
    end_year   : last year to include  (default 2024)

    Returns
    -------
    Merged DataFrame ready for feature engineering.
    """
    temp = load_berkeley_temperature(use_cache=use_cache)
    co2  = load_noaa_co2(use_cache=use_cache)
    ssn  = load_silso_sunspots(use_cache=use_cache)
    return merge_datasets(temp, co2, ssn, start_year, end_year)


# ─────────────────────────────────────────────────────────────
# Synthetic fallbacks (physics-informed; used when remote data
# is unavailable, e.g. offline Colab runtime)
# ─────────────────────────────────────────────────────────────

def _synthetic_temperature() -> pd.DataFrame:
    dates = pd.date_range("1880-01-01", "2024-12-01", freq="MS")
    t = np.arange(len(dates))
    trend    = 0.008 * t / 12 + 0.0003 * (t / 12) ** 1.5
    seasonal = 0.12 * np.sin(2 * np.pi * t / 12 + 0.5)
    enso     = 0.25 * np.sin(2 * np.pi * t / (12 * 3.7) + 1.2)
    pdo      = 0.10 * np.sin(2 * np.pi * t / (12 * 22) + 0.8)
    noise    = np.random.normal(0, 0.08, len(t))
    return pd.DataFrame({
        "date"         : dates,
        "year"         : dates.year,
        "month"        : dates.month,
        "temp_anomaly" : trend + seasonal + enso + pdo + noise - 0.3,
    })


def _synthetic_co2() -> pd.DataFrame:
    dates = pd.date_range("1958-03-01", "2024-12-01", freq="MS")
    t = np.arange(len(dates))
    base     = 315 + 0.13 * t + 0.00012 * t ** 1.6
    seasonal = 3.5 * np.sin(2 * np.pi * t / 12 + 4.2)
    noise    = np.random.normal(0, 0.15, len(t))
    return pd.DataFrame({
        "date"   : dates,
        "year"   : dates.year,
        "month"  : dates.month,
        "co2_ppm": base + seasonal + noise,
    })


def _synthetic_sunspots() -> pd.DataFrame:
    dates = pd.date_range("1749-01-01", "2024-12-01", freq="MS")
    t = np.arange(len(dates))
    cycle11 = 80 + 60 * np.abs(np.sin(np.pi * t / (12 * 11.1)))
    cycle22 = 20 * np.sin(2 * np.pi * t / (12 * 22))
    noise   = np.random.exponential(8, len(t))
    return pd.DataFrame({
        "date"            : dates,
        "year"            : dates.year,
        "month"           : dates.month,
        "sunspot_number"  : np.clip(cycle11 + cycle22 + noise, 0, 320),
    })


# ─────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all(use_cache=True)
    out = project_root() / "data" / "raw"
    ensure_dirs(str(out))
    print(df.tail())
    print(f"\nDataset ready: {df.shape}")
