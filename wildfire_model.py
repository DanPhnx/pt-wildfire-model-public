"""Shared, importable logic used by more than one phase notebook.

Kept out of the notebooks per the PRD's technical decision ("Entry point:
one main.py; notebooks for EDA only") - phase notebooks are read as EDA and
diagnostics, not as the owner of logic that other phases also depend on.
Only 01_eda.ipynb reads raw source data; from here on, phases 2-4 work from
data/processed/wildfires_processed.csv and this module.

Import from a notebook (which runs with the notebook's own directory as
its working directory) with:

    import sys
    sys.path.insert(0, "..")
    from wildfire_model import build_fire_day_events, load_icnf_prdf_database
"""

import sqlite3
from pathlib import Path

import pandas as pd

# ICNF PRDF raw file, relative to this module (repo root)
_ICNF_PRDF_FILE = Path(__file__).parent / "data/raw/icnf_prdf_fogos.gpkg"

MIN_FIRE_AREA_HA = 30.0


def load_icnf_prdf_database(
    path: Path = _ICNF_PRDF_FILE,
    min_area_ha: float = MIN_FIRE_AREA_HA,
    start_year: int = 2009,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Load the ICNF Portuguese Rural Fire Database (PRDF, 1980-2025).

    Reads the GeoPackage attribute table via sqlite3 — no geopandas needed,
    the geometry column is ignored. Returns the same schema the downstream
    pipeline expects: Date, Location, Burned_Area_ha.

    Source: Zenodo DOI 10.5281/zenodo.21427772 (Lopes Almeida et al., 2026),
    committed to data/raw/icnf_prdf_fogos.gpkg as a static snapshot (no live
    API, no reproducibility drift).

    The PRDF covers mainland Portugal only (all 18 mainland districts appear;
    island districts do not); no geographic filter is needed. The NUTS2 field
    is NULL for 2009-2025 records — Distrito (district) is used for Location
    instead. Same MIN_FIRE_AREA_HA=30 threshold as the former EFFIS pipeline.

    Unlike the EFFIS export there is no sensor break (PRDF harmonises five
    historical databases into a unified series, so all years use the same
    definition). Exact-duplicate records from EFFIS's export artifact are
    also not present; no deduplication step is applied.

    Parameters
    ----------
    path : Path
        Location of the committed ICNF PRDF GeoPackage.
    min_area_ha : float
        Minimum AREATOTAL (ha) to include as a fire event.
    start_year, end_year : int
        Inclusive year window applied on ANO (alert year).

    Returns
    -------
    pd.DataFrame
        Columns: Date (datetime64[ns]), Location (str, NUTS2 region),
        Burned_Area_ha (float). One row per fire, sorted by Date.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download fogos.gpkg from Zenodo DOI "
            "10.5281/zenodo.21427772 and place it in data/raw/."
        )
    conn = sqlite3.connect(str(path))
    # Column names in the GeoPackage are mixed case: DHInicio, AreaTotal, Ano, Distrito.
    # SQLite3 column name matching is case-insensitive so these spellings work fine.
    # The PRDF covers mainland Portugal only (18 mainland districts; no island records
    # appear in the data), so no further geographic filter is needed.
    df = pd.read_sql_query(
        "SELECT DHInicio, AreaTotal, Distrito, Ano FROM Fogos"
        " WHERE CAST(Ano AS INTEGER) BETWEEN ? AND ?",
        conn,
        params=(start_year, end_year),
    )
    conn.close()

    df["Date"] = pd.to_datetime(df["DHInicio"], format="mixed", utc=True).dt.tz_localize(None)
    # Drop non-physical records (area <= 0 or NULL)
    df = df[df["AreaTotal"].notna() & (df["AreaTotal"] > 0)]
    # Apply size threshold
    df = df[df["AreaTotal"] >= min_area_ha]

    return (
        df.rename(columns={"Distrito": "Location", "AreaTotal": "Burned_Area_ha"})
        [["Date", "Location", "Burned_Area_ha"]]
        .sort_values("Date")
        .reset_index(drop=True)
    )

# Confirmed Phase 1 checkpoint decision (see docs/worklog.md): train on
# 2009-2020, hold out 2021-2025 for the Phase 4 backtest. Keeps 2017 - the
# main severity-tail anchor - in training; the hold-out mixes quiet years
# (2021, 2023) and severe ones (2022, 2024, 2025).
TRAIN_YEARS = (2009, 2020)
HOLDOUT_YEARS = (2021, 2025)


def build_fire_day_events(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fire records into fire-day events.

    Individual EFFIS polygons are not independent events for modelling
    purposes: multiple fires on the same day are often the same
    weather-driven outbreak rather than unrelated ignitions (e.g. 33
    polygons burned 196,476 ha on 2017-10-15 alone). Grouping by start date
    is the event definition chosen during the Phase 1 review (see
    docs/worklog.md and the "What the data shows" section of
    docs/phase1-writeup.md): on the full 2009-2025 record it cuts annual
    overdispersion from ~41 (per-polygon) to ~5.5 (fire-day) and removes
    the significant count-vs-size correlation (rho 0.57, p = 0.02
    per-polygon vs rho 0.25, p = 0.33 fire-day); a residual dependence
    remains (annual-area SD is still ~1.8x an independent model's).

    This is a simple first choice (same calendar start date, national
    scope, no distinction between simultaneous fires in different
    regions) and is flagged in the worklog as something to revisit later
    (e.g. multi-day windows, in the style of a treaty hours clause).

    Parameters
    ----------
    df : pd.DataFrame
        Per-fire records with Date, Burned_Area_ha, Estimated_Loss_EUR and
        Estimated_Loss_EUR_2025 (as produced by 01_eda.ipynb).

    Returns
    -------
    pd.DataFrame
        One row per calendar day with at least one fire event, sorted by
        date. Columns: Date, N_Fires, Burned_Area_ha, Estimated_Loss_EUR,
        Estimated_Loss_EUR_2025 (all but N_Fires summed across that day's
        fires). Estimated_Loss_EUR is summed in each fire's own-year
        euros, which is exact here since a fire-day never spans a year
        boundary.
    """
    day = df["Date"].dt.normalize()
    events = df.groupby(day).agg(
        N_Fires=("Burned_Area_ha", "size"),
        Burned_Area_ha=("Burned_Area_ha", "sum"),
        Estimated_Loss_EUR=("Estimated_Loss_EUR", "sum"),
        Estimated_Loss_EUR_2025=("Estimated_Loss_EUR_2025", "sum"),
    )
    events.index.name = "Date"
    return events.reset_index().sort_values("Date").reset_index(drop=True)


def split_train_holdout(events: pd.DataFrame, date_col: str = "Date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split events into the confirmed training and hold-out windows.

    Parameters
    ----------
    events : pd.DataFrame
        Event-level data with a date column (fire-day events or raw
        per-fire records both work).
    date_col : str
        Name of the date column.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        (train, holdout), split on TRAIN_YEARS / HOLDOUT_YEARS.
    """
    year = events[date_col].dt.year
    train = events[year.between(*TRAIN_YEARS)]
    holdout = events[year.between(*HOLDOUT_YEARS)]
    return train, holdout
