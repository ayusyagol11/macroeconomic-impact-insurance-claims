"""
Data Engineering Pipeline
Macroeconomic Impact on Insurance Claims

Produces:
  data/processed/economic_master.csv  - Macro indicators (monthly, 2015-2026)
  data/processed/insurance_master.csv - APRA insurance metrics (quarterly, 2023-2025)
  data/processed/model_ready.csv      - Merged + lag features aligned on quarter-end dates
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

RAW = "data/raw/"
PROC = "data/processed/"
os.makedirs(PROC, exist_ok=True)

# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────

def _find_header_row(path: str, sheet: str) -> int | None:
    """Return the 0-based file row index that contains 'Series ID'."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    for i, row in raw.iterrows():
        if "Series ID" in row.values:
            return int(i)
    return None


def load_abs(filename: str, series_id: str) -> pd.DataFrame:
    """Load a single ABS series from an Excel Data1 sheet."""
    path = os.path.join(RAW, filename)
    if not os.path.exists(path):
        print(f"  ⚠️  Missing: {filename}")
        return pd.DataFrame()

    header_row = _find_header_row(path, "Data1")
    if header_row is None:
        print(f"  ⚠️  No 'Series ID' row found in {filename}")
        return pd.DataFrame()

    # header=header_row: use that file row as column names; rows above are skipped
    df = pd.read_excel(path, sheet_name="Data1", header=header_row)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if series_id not in df.columns:
        print(f"  ⚠️  Series '{series_id}' not found in {filename}")
        return pd.DataFrame()

    return df[["Date", series_id]].copy()


def load_rba(filename: str = "rba_f1_1_historical.xlsx") -> pd.DataFrame:
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
    return df[["Date", "FIRMMCRT"]].rename(columns={"FIRMMCRT": "cash_rate"}).copy()


# ─────────────────────────────────────────────
# 2. ECONOMIC MASTER (monthly, 2015-2026)
# ─────────────────────────────────────────────
print("\n── Building economic_master.csv ──")

cpi_raw  = load_abs("abs_cpi_jan26.xlsx",  "A130393720C")
wpi_raw  = load_abs("abs_wpi_dec25.xlsx",  "A2603609J")
ppi_raw  = load_abs("abs_PPI_dec25.xlsx",  "A2333649T")
rba_raw  = load_rba()

# Rename to human-readable labels
def rename_col(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    return df.rename(columns={old: new}) if not df.empty else df

cpi_raw = rename_col(cpi_raw, "A130393720C", "cpi")
wpi_raw = rename_col(wpi_raw, "A2603609J",   "wpi")
ppi_raw = rename_col(ppi_raw, "A2333649T",   "ppi")

# Merge on Date (outer join keeps all periods)
frames = [df for df in [cpi_raw, wpi_raw, ppi_raw, rba_raw] if not df.empty]
eco = frames[0]
for f in frames[1:]:
    eco = eco.merge(f, on="Date", how="outer")

eco = eco.sort_values("Date").reset_index(drop=True)

# Deduplicate: keep row with most non-null values per date
eco = (
    eco.assign(_nonnull=eco.drop(columns="Date").notna().sum(axis=1))
       .sort_values(["Date", "_nonnull"], ascending=[True, False])
       .drop_duplicates(subset="Date", keep="first")
       .drop(columns="_nonnull")
       .reset_index(drop=True)
)

# Forward-fill sparse series (WPI/PPI are quarterly, fill to monthly)
eco[["wpi", "ppi", "cash_rate"]] = eco[["wpi", "ppi", "cash_rate"]].ffill()

# Filter to study period
eco = eco[eco["Date"] >= "2015-01-01"].copy()

# YoY % change features
for col in ["cpi", "wpi", "ppi", "cash_rate"]:
    if col in eco.columns:
        eco[f"{col}_yoy"] = eco[col].pct_change(12) * 100

# 3-month and 12-month rolling average
for col in ["cpi", "wpi", "ppi"]:
    if col in eco.columns:
        eco[f"{col}_roll3"]  = eco[col].rolling(3).mean()
        eco[f"{col}_roll12"] = eco[col].rolling(12).mean()

# Inflation volatility (12-month std of YoY CPI)
if "cpi_yoy" in eco.columns:
    eco["cpi_volatility"] = eco["cpi_yoy"].rolling(12).std()

# Indicator flags
eco["flag_cpi_spike"]   = (eco.get("cpi_yoy",   pd.Series(dtype=float)) > 4).astype(int)
eco["flag_rate_hike"]   = (eco.get("cash_rate",  pd.Series(dtype=float)) > 4).astype(int)

eco.to_csv(os.path.join(PROC, "economic_master.csv"), index=False)
print(f"  ✅ economic_master.csv  — {len(eco)} rows, {eco.columns.tolist()}")


# ─────────────────────────────────────────────
# 3. INSURANCE MASTER (quarterly, APRA)
# ─────────────────────────────────────────────
print("\n── Building insurance_master.csv ──")

TARGET_CLASSES = ["Householders", "Domestic motor"]

ITEM_MAP = {
    "Gross claims incurred, by class of business":  "gross_claims",
    "Net claims incurred, by class of business":    "net_claims",
    "Gross written premium, by class of business":  "gwp",
    "Insurance revenue, by class of business":      "insurance_revenue",
}

try:
    apra_raw = pd.read_excel(
        os.path.join(RAW, "apra_industry_dec25.xlsx"), sheet_name="Database"
    )
    apra_raw["Reporting Period"] = pd.to_datetime(apra_raw["Reporting Period"])

    ins_frames = []
    for item_label, col_name in ITEM_MAP.items():
        sub = apra_raw[
            (apra_raw["Data item"] == item_label)
            & (apra_raw["Class of business"].isin(TARGET_CLASSES))
        ].copy()
        if sub.empty:
            continue
        piv = sub.pivot_table(
            index="Reporting Period",
            columns="Class of business",
            values="Value",
            aggfunc="sum",
        ).reset_index()
        piv.columns.name = None
        rename = {c: f"{col_name}_{c.lower().replace(' ', '_')}" for c in TARGET_CLASSES}
        piv = piv.rename(columns={"Reporting Period": "Date", **rename})
        ins_frames.append(piv)

    ins = ins_frames[0]
    for f in ins_frames[1:]:
        ins = ins.merge(f, on="Date", how="outer")

    ins = ins.sort_values("Date").reset_index(drop=True)

    # Loss Ratio = gross_claims / gwp  (per class)
    for cls in ["householders", "domestic_motor"]:
        gc  = f"gross_claims_{cls}"
        gwp = f"gwp_{cls}"
        if gc in ins.columns and gwp in ins.columns:
            ins[f"loss_ratio_{cls}"] = ins[gc] / ins[gwp]

    # YoY growth in claims
    for col in [c for c in ins.columns if c.startswith("gross_claims")]:
        ins[f"{col}_yoy"] = ins[col].pct_change(4) * 100  # 4 quarters = 1 year

    ins.to_csv(os.path.join(PROC, "insurance_master.csv"), index=False)
    print(f"  ✅ insurance_master.csv — {len(ins)} rows, {ins.columns.tolist()}")

except Exception as e:
    print(f"  ❌ APRA Error: {e}")
    ins = pd.DataFrame()


# ─────────────────────────────────────────────
# 4. MODEL-READY DATASET (merged, lag features)
# ─────────────────────────────────────────────
print("\n── Building model_ready.csv ──")

if not ins.empty:
    # Resample eco to quarter-end to align with APRA
    eco_q = eco.copy()
    eco_q = eco_q.set_index("Date").resample("QE").last().reset_index()

    model = eco_q.merge(ins, on="Date", how="inner")

    # Lag features: macro vars at t-k months (using quarter steps k=1..6)
    macro_base = ["cpi", "wpi", "ppi", "cash_rate"]
    for col in macro_base:
        if col in model.columns:
            for lag in range(1, 7):
                model[f"{col}_lag{lag}q"] = model[col].shift(lag)

    # Lag YoY changes too
    for col in ["cpi_yoy", "wpi_yoy", "ppi_yoy"]:
        if col in model.columns:
            for lag in [1, 2, 3]:
                model[f"{col}_lag{lag}q"] = model[col].shift(lag)

    # Season (quarter number)
    model["quarter"] = model["Date"].dt.quarter

    model.to_csv(os.path.join(PROC, "model_ready.csv"), index=False)
    print(f"  ✅ model_ready.csv      — {len(model)} rows, {len(model.columns)} columns")
    print(f"     Date range: {model['Date'].min().date()} → {model['Date'].max().date()}")
else:
    print("  ⚠️  Skipping model_ready.csv — insurance data unavailable")

print("\n✅ Data engineering complete.")
