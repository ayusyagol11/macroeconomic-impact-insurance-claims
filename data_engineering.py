"""
Data Engineering Pipeline
Macroeconomic Impact on Insurance Claims
-----------------------------------------
This script reads raw Excel files from data/raw/, cleans and transforms them,
and writes three ready-to-use CSV files into data/processed/:

  economic_master.csv  - All macro indicators at monthly frequency (2015–2026)
  insurance_master.csv - APRA insurance metrics at quarterly frequency (2023–2025)
  model_ready.csv      - The two datasets merged together, plus lag features for modelling

Run this script first before opening the notebook or launching the dashboard.
"""

import os
import warnings
import pandas as pd
import numpy as np

# Suppress noisy pandas/openpyxl warnings that don't affect the output
warnings.filterwarnings("ignore")

# Folder paths — all raw Excel files live in data/raw/, outputs go to data/processed/
RAW  = "data/raw/"
PROC = "data/processed/"
os.makedirs(PROC, exist_ok=True)   # Create the processed folder if it doesn't already exist


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HELPER FUNCTIONS
# These handle the quirky structure of ABS and RBA Excel files, where the first
# several rows are metadata (units, frequency, dates) before the actual data begins.
# ─────────────────────────────────────────────────────────────────────────────

def _find_header_row(path: str, sheet: str) -> int | None:
    """
    Scan an Excel sheet row-by-row to find which row contains 'Series ID'.

    ABS and RBA files store metadata in the top rows before the actual column
    headers appear. The row labelled 'Series ID' marks where real column names are.
    We need its row number so we can tell pandas to use it as the header.

    Returns the row index (0-based) if found, or None if the file is unexpected.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    for i, row in raw.iterrows():
        if "Series ID" in row.values:
            return int(i)
    return None   # File structure was not what we expected


def load_abs(filename: str, series_id: str) -> pd.DataFrame:
    """
    Load a single data series from an ABS Excel publication.

    ABS files (CPI, WPI, PPI) all follow the same layout:
      - Rows 0–8: metadata (units, frequency, start/end dates, etc.)
      - Row 9:    'Series ID' row — this becomes the column header
      - Row 10+:  actual data, one row per time period

    Parameters
    ----------
    filename  : the Excel filename inside data/raw/  (e.g. 'abs_cpi_jan26.xlsx')
    series_id : the ABS series code to extract       (e.g. 'A130393720C')

    Returns a two-column DataFrame: ['Date', series_id], or empty if anything fails.
    """
    path = os.path.join(RAW, filename)

    # Guard: skip gracefully if the file hasn't been downloaded yet
    if not os.path.exists(path):
        print(f"  ⚠️  Missing: {filename}")
        return pd.DataFrame()

    # Locate the 'Series ID' row so we know where the real headers are
    header_row = _find_header_row(path, "Data1")
    if header_row is None:
        print(f"  ⚠️  No 'Series ID' row found in {filename}")
        return pd.DataFrame()

    # Re-read the file, this time telling pandas to use the Series ID row as column names
    # This automatically skips all the metadata rows above it
    df = pd.read_excel(path, sheet_name="Data1", header=header_row)

    # The first column is always the date, but its name varies — rename it consistently
    df = df.rename(columns={df.columns[0]: "Date"})

    # Parse dates; coerce=True turns anything unparseable into NaT rather than crashing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows where the date couldn't be parsed (e.g. empty trailing rows)
    df = df.dropna(subset=["Date"])

    # Check the requested series actually exists in this file
    if series_id not in df.columns:
        print(f"  ⚠️  Series '{series_id}' not found in {filename}")
        return pd.DataFrame()

    # Return only the two columns we need: date and the series value
    return df[["Date", series_id]].copy()


def load_rba(filename: str = "rba_f1_1_historical.xlsx") -> pd.DataFrame:
    """
    Load the RBA Cash Rate Target series from the RBA F1.1 Excel publication.

    The RBA file has the same metadata-rows structure as ABS files, but uses
    the sheet name 'Data' instead of 'Data1'.

    We extract the 'FIRMMCRT' series (Cash Rate Target) and rename it to
    'cash_rate' for readability throughout the rest of the pipeline.

    Returns a two-column DataFrame: ['Date', 'cash_rate'], or empty if anything fails.
    """
    path = os.path.join(RAW, filename)

    if not os.path.exists(path):
        print(f"  ⚠️  Missing: {filename}")
        return pd.DataFrame()

    header_row = _find_header_row(path, "Data")
    if header_row is None:
        print(f"  ⚠️  No 'Series ID' row found in RBA file")
        return pd.DataFrame()

    df = pd.read_excel(path, sheet_name="Data", header=header_row)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Keep only date + cash rate, and give it a human-readable column name
    return df[["Date", "FIRMMCRT"]].rename(columns={"FIRMMCRT": "cash_rate"}).copy()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BUILD economic_master.csv
# Load all four macro series, merge them on Date, clean duplicates,
# then engineer useful features (YoY changes, rolling averages, flags).
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Building economic_master.csv ──")

# Load each raw series — each returns a 2-column DataFrame (Date + value)
cpi_raw = load_abs("abs_cpi_jan26.xlsx", "A130393720C")  # CPI: Insurance & Financial Services subgroup
wpi_raw = load_abs("abs_wpi_dec25.xlsx", "A2603609J")    # WPI: Total hourly rates excl. bonuses
ppi_raw = load_abs("abs_PPI_dec25.xlsx", "A2333649T")    # PPI: Construction output
rba_raw = load_rba()                                     # RBA Cash Rate Target


def rename_col(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """Rename a single column to a friendlier name, returning the df unchanged if empty."""
    return df.rename(columns={old: new}) if not df.empty else df

# Replace the cryptic ABS series codes with readable column names
cpi_raw = rename_col(cpi_raw, "A130393720C", "cpi")
wpi_raw = rename_col(wpi_raw, "A2603609J",   "wpi")
ppi_raw = rename_col(ppi_raw, "A2333649T",   "ppi")

# Combine all non-empty frames into a single list for merging
frames = [df for df in [cpi_raw, wpi_raw, ppi_raw, rba_raw] if not df.empty]

# Merge all series together on the Date column.
# Outer join = keep every date that appears in any source, even if others don't have it.
eco = frames[0]
for f in frames[1:]:
    eco = eco.merge(f, on="Date", how="outer")

eco = eco.sort_values("Date").reset_index(drop=True)

# ── Deduplication ─────────────────────────────────────────────────────────────
# Problem: when merging monthly and quarterly series, some dates appear twice
# (e.g. one source uses period-start dates, another uses period-end dates for
# the same quarter). We resolve this by keeping whichever duplicate row has
# more actual data values (fewest NaNs).
eco = (
    eco
    .assign(_nonnull=eco.drop(columns="Date").notna().sum(axis=1))  # count non-null values per row
    .sort_values(["Date", "_nonnull"], ascending=[True, False])      # best row first within each date
    .drop_duplicates(subset="Date", keep="first")                    # keep the richest row
    .drop(columns="_nonnull")                                        # remove the helper column
    .reset_index(drop=True)
)

# ── Forward-fill sparse quarterly series ──────────────────────────────────────
# WPI and PPI are published quarterly but our date index is monthly.
# Forward-fill carries the last known quarterly value forward to fill the gaps,
# so that e.g. the Sep-quarter WPI value is repeated for Oct and Nov.
eco[["wpi", "ppi", "cash_rate"]] = eco[["wpi", "ppi", "cash_rate"]].ffill()

# ── Filter to study period ────────────────────────────────────────────────────
# We only care about 2015 onwards — pre-2015 data is outside the project scope
eco = eco[eco["Date"] >= "2015-01-01"].copy()

# ── Feature: Year-on-Year % change ───────────────────────────────────────────
# pct_change(12) compares each month to the same month 12 months earlier.
# Multiply by 100 to express as a percentage rather than a decimal.
# This is the key inflation signal — we want to know "is inflation accelerating?"
for col in ["cpi", "wpi", "ppi", "cash_rate"]:
    if col in eco.columns:
        eco[f"{col}_yoy"] = eco[col].pct_change(12) * 100

# ── Feature: Rolling averages ─────────────────────────────────────────────────
# 3-month rolling average smooths out month-to-month noise.
# 12-month rolling average captures the longer-term trend direction.
for col in ["cpi", "wpi", "ppi"]:
    if col in eco.columns:
        eco[f"{col}_roll3"]  = eco[col].rolling(3).mean()
        eco[f"{col}_roll12"] = eco[col].rolling(12).mean()

# ── Feature: Inflation volatility ─────────────────────────────────────────────
# Standard deviation of CPI YoY over the past 12 months.
# A rising volatility score signals an unstable inflation environment,
# which is relevant for reserving uncertainty.
if "cpi_yoy" in eco.columns:
    eco["cpi_volatility"] = eco["cpi_yoy"].rolling(12).std()

# ── Feature: Binary alert flags ───────────────────────────────────────────────
# Simple 1/0 flags that fire when a threshold is breached.
# These can be used in models as event indicators or in dashboard alerts.
eco["flag_cpi_spike"]  = (eco.get("cpi_yoy",  pd.Series(dtype=float)) > 4).astype(int)  # CPI > 4% YoY
eco["flag_rate_hike"]  = (eco.get("cash_rate", pd.Series(dtype=float)) > 4).astype(int)  # Cash rate > 4%

# Save to disk
eco.to_csv(os.path.join(PROC, "economic_master.csv"), index=False)
print(f"  ✅ economic_master.csv  — {len(eco)} rows, {eco.columns.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BUILD insurance_master.csv
# Load APRA quarterly GI statistics, extract the metrics we care about,
# pivot from long format to wide, then calculate loss ratio and YoY growth.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Building insurance_master.csv ──")

# The two insurance classes we're analysing
TARGET_CLASSES = ["Householders", "Domestic motor"]

# Map APRA's verbose 'Data item' labels to short column prefixes we'll use in the output
# Each item will produce two columns: one for Householders, one for Domestic motor
ITEM_MAP = {
    "Gross claims incurred, by class of business":  "gross_claims",    # Total claims paid out
    "Net claims incurred, by class of business":    "net_claims",      # Claims net of reinsurance recoveries
    "Gross written premium, by class of business":  "gwp",             # Total premium collected
    "Insurance revenue, by class of business":      "insurance_revenue",
}

try:
    # Load the APRA database sheet — it's in long format (one row per data item per period)
    apra_raw = pd.read_excel(
        os.path.join(RAW, "apra_industry_dec25.xlsx"), sheet_name="Database"
    )
    apra_raw["Reporting Period"] = pd.to_datetime(apra_raw["Reporting Period"])

    ins_frames = []

    # Loop through each metric we want to extract
    for item_label, col_name in ITEM_MAP.items():

        # Filter to just this metric and our two classes of business
        sub = apra_raw[
            (apra_raw["Data item"] == item_label)
            & (apra_raw["Class of business"].isin(TARGET_CLASSES))
        ].copy()

        if sub.empty:
            continue   # This metric might not exist in older APRA publications

        # Pivot from long to wide:
        # Before: one row per (period, class)
        # After:  one row per period, one column per class
        piv = sub.pivot_table(
            index="Reporting Period",
            columns="Class of business",
            values="Value",
            aggfunc="sum",   # In case of duplicates, sum them (rare but safe)
        ).reset_index()
        piv.columns.name = None   # Remove the multi-index name left by pivot_table

        # Rename columns to our short format, e.g. "Householders" → "gross_claims_householders"
        rename = {c: f"{col_name}_{c.lower().replace(' ', '_')}" for c in TARGET_CLASSES}
        piv = piv.rename(columns={"Reporting Period": "Date", **rename})

        ins_frames.append(piv)

    # Merge all metrics together on Date (each metric was extracted separately above)
    ins = ins_frames[0]
    for f in ins_frames[1:]:
        ins = ins.merge(f, on="Date", how="outer")

    ins = ins.sort_values("Date").reset_index(drop=True)

    # ── Derived metric: Loss Ratio ─────────────────────────────────────────────
    # Loss Ratio = Gross Claims / Gross Written Premium
    # This is the most important KPI in general insurance.
    # A ratio above 1.0 means claims exceeded premium — the insurer is losing money on underwriting.
    # A ratio above 0.85 is typically a warning sign requiring management attention.
    for cls in ["householders", "domestic_motor"]:
        gc  = f"gross_claims_{cls}"
        gwp = f"gwp_{cls}"
        if gc in ins.columns and gwp in ins.columns:
            ins[f"loss_ratio_{cls}"] = ins[gc] / ins[gwp]

    # ── Derived metric: YoY claims growth ─────────────────────────────────────
    # pct_change(4) compares each quarter to the same quarter one year earlier
    # (4 quarters = 1 year). This removes seasonality from the growth rate.
    for col in [c for c in ins.columns if c.startswith("gross_claims")]:
        ins[f"{col}_yoy"] = ins[col].pct_change(4) * 100

    # Save to disk
    ins.to_csv(os.path.join(PROC, "insurance_master.csv"), index=False)
    print(f"  ✅ insurance_master.csv — {len(ins)} rows, {ins.columns.tolist()}")

except Exception as e:
    print(f"  ❌ APRA Error: {e}")
    ins = pd.DataFrame()   # Carry forward an empty frame so the next section fails gracefully


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BUILD model_ready.csv
# Merge the economic and insurance datasets on their shared quarter-end dates,
# then add lag features so the model can test whether macro conditions from
# 1–6 quarters ago predict insurance outcomes today.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Building model_ready.csv ──")

if not ins.empty:

    # ── Resample economics to quarter-end dates ────────────────────────────────
    # The economic_master is monthly; APRA data is quarterly (March, June, Sep, Dec).
    # We resample the macro data to quarterly frequency by taking the LAST value in
    # each quarter — this represents the most recent reading heading into the next quarter.
    # "QE" = Quarter End frequency in pandas.
    eco_q = eco.copy()
    eco_q = eco_q.set_index("Date").resample("QE").last().reset_index()

    # Inner join: only keep dates that exist in BOTH datasets.
    # This gives us the overlapping window where we have both macro and insurance data.
    model = eco_q.merge(ins, on="Date", how="inner")

    # ── Lag features for core macro indicators ─────────────────────────────────
    # The central hypothesis is that macro conditions TODAY predict insurance claims
    # in FUTURE quarters. We test this by creating lagged versions of each macro series.
    #
    # For example:  wpi_lag2q  =  the WPI value from 2 quarters ago
    #               cpi_lag4q  =  the CPI value from 4 quarters ago
    #
    # When we run a regression, the model can then find which lag (if any) gives the
    # strongest predictive relationship. We test lags 1 through 6 quarters.
    macro_base = ["cpi", "wpi", "ppi", "cash_rate"]
    for col in macro_base:
        if col in model.columns:
            for lag in range(1, 7):
                # shift(n) moves values DOWN by n rows, i.e. the value at row t
                # comes from row t-n — exactly what "macro n quarters ago" means
                model[f"{col}_lag{lag}q"] = model[col].shift(lag)

    # Also lag the YoY rates — testing whether the RATE of change leads claims
    for col in ["cpi_yoy", "wpi_yoy", "ppi_yoy"]:
        if col in model.columns:
            for lag in [1, 2, 3]:
                model[f"{col}_lag{lag}q"] = model[col].shift(lag)

    # ── Seasonality feature ────────────────────────────────────────────────────
    # Quarter number (1=Jan–Mar, 2=Apr–Jun, 3=Jul–Sep, 4=Oct–Dec).
    # Australian summer falls in Q1/Q4, which is peak weather-event season for
    # home insurance — capturing this helps models separate seasonal spikes
    # from macro-driven trends.
    model["quarter"] = model["Date"].dt.quarter

    # Save to disk
    model.to_csv(os.path.join(PROC, "model_ready.csv"), index=False)
    print(f"  ✅ model_ready.csv      — {len(model)} rows, {len(model.columns)} columns")
    print(f"     Date range: {model['Date'].min().date()} → {model['Date'].max().date()}")

else:
    # If APRA loading failed above, we can't build the merged file
    print("  ⚠️  Skipping model_ready.csv — insurance data unavailable")

print("\n✅ Data engineering complete.")
