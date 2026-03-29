"""
Modeling Pipeline – Macroeconomic Impact on Insurance Claims
============================================================
Steps:
  1. Feature engineering (YoY, rolling, volatility, indicator flags)
  2. Model comparison: OLS, Ridge, Lasso, XGBoost
  3. Time-series backtesting (train 2023 Q3–Q4, test 2024+)
  4. Stress-test scenarios (A: persistent inflation, B: stagflation)
  5. Save results to modeling/results/

Usage:
  cd <project_root>
  python modeling/train_models.py
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROC    = "data/processed/"
RESULTS = "modeling/results/"
os.makedirs(RESULTS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams["figure.dpi"] = 130

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n── Loading processed data ──")
eco = pd.read_csv(PROC + "economic_master.csv", parse_dates=["Date"])
ins = pd.read_csv(PROC + "insurance_master.csv", parse_dates=["Date"])
model_df = pd.read_csv(PROC + "model_ready.csv", parse_dates=["Date"])

print(f"  economic_master : {len(eco)} rows")
print(f"  insurance_master: {len(ins)} rows")
print(f"  model_ready     : {len(model_df)} rows | cols: {model_df.shape[1]}")

if len(model_df) < 6:
    print("\n⚠️  Insufficient rows in model_ready for meaningful modelling.")
    print("   Run data_engineering.py first, or check APRA data coverage.")
    sys.exit(0)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n── Feature engineering ──")

df = model_df.copy()

# Acceleration: difference of YoY rates (is inflation speeding up?)
for col in ["cpi_yoy", "wpi_yoy", "ppi_yoy"]:
    if col in df.columns:
        df[f"{col}_accel"] = df[col].diff()

# 3-quarter accelerating inflation flag
if "cpi_yoy" in df.columns:
    df["cpi_accel3q"] = (df["cpi_yoy"].rolling(3).mean() > df["cpi_yoy"].shift(3)).astype(int)

# Composite macro stress score (normalised sum of YoY measures)
yoy_cols = [c for c in ["wpi_yoy", "ppi_yoy", "cpi_yoy"] if c in df.columns]
if yoy_cols:
    df["macro_stress_score"] = df[yoy_cols].apply(
        lambda row: row.fillna(0).sum() / len(yoy_cols), axis=1
    )

feature_cols = [
    c for c in df.columns
    if c not in ["Date", "quarter"]
    and not any(c.endswith(f"_lag{i}q") for i in range(1, 7))
    and df[c].dtype in [np.float64, np.int64, float, int]
]

print(f"  Derived features: {[c for c in df.columns if c not in model_df.columns]}")

# ─────────────────────────────────────────────
# 3. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n── Model comparison ──")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("  ⚠️  XGBoost unavailable (install libomp via brew) — skipping")

TARGETS = [
    ("loss_ratio_householders",   "Householders Loss Ratio"),
    ("loss_ratio_domestic_motor", "Domestic Motor Loss Ratio"),
    ("gross_claims_householders", "Householders Gross Claims"),
]

model_results = []

for target_col, target_name in TARGETS:
    if target_col not in df.columns:
        print(f"  ⚠️  {target_col} not found — skipping")
        continue

    # Use only core macro predictors (no derived features that create extra NaNs)
    core_macro = ["wpi", "ppi", "cash_rate", "cpi", "quarter"]
    predictor_candidates = [c for c in core_macro if c in df.columns and c != target_col]

    valid_preds = [c for c in predictor_candidates if c in df.columns]
    data = df[valid_preds + [target_col]].dropna()

    if len(data) < 5:
        print(f"  ⚠️  {target_name}: only {len(data)} rows after dropna — skipping")
        continue

    X = data[valid_preds]
    y = data[target_col]

    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    models = [
        ("OLS",   LinearRegression()),
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.01, max_iter=5000)),
    ]
    if HAS_XGB:
        models.append(("XGBoost", XGBRegressor(n_estimators=50, max_depth=2,
                                                learning_rate=0.1, random_state=42,
                                                verbosity=0)))

    print(f"\n  ── {target_name} (n={len(data)}) ──")
    for model_name, model in models:
        if model_name == "XGBoost":
            model.fit(X, y)
            y_hat = model.predict(X)
        else:
            model.fit(Xs, y)
            y_hat = model.predict(Xs)

        mae  = mean_absolute_error(y, y_hat)
        rmse = mean_squared_error(y, y_hat) ** 0.5
        r2   = r2_score(y, y_hat)
        print(f"    {model_name:<8} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.3f}")
        model_results.append({
            "target":    target_name,
            "model":     model_name,
            "n":         len(data),
            "MAE":       round(mae, 4),
            "RMSE":      round(rmse, 4),
            "R2":        round(r2, 3),
        })

if model_results:
    results_df = pd.DataFrame(model_results)
    results_df.to_csv(RESULTS + "model_comparison.csv", index=False)
    print(f"\n  Saved → {RESULTS}model_comparison.csv")

# ─────────────────────────────────────────────
# 4. BACKTEST (time-series split)
# ─────────────────────────────────────────────
print("\n── Backtesting ──")

target_col = "loss_ratio_householders"
if target_col in df.columns:
    core_macro = ["wpi", "ppi", "cash_rate", "cpi", "quarter"]
    valid_preds = [c for c in core_macro if c in df.columns and c != target_col]
    data = df[valid_preds + [target_col, "Date"]].dropna()

    cutoff = data["Date"].quantile(0.6)  # ~first 60% for train
    train  = data[data["Date"] <= cutoff]
    test   = data[data["Date"] >  cutoff]

    print(f"  Train: {len(train)} obs up to {cutoff.date()}")
    print(f"  Test : {len(test)} obs after {cutoff.date()}")

    if len(train) >= 4 and len(test) >= 2:
        scaler = StandardScaler()
        Xtrain = pd.DataFrame(scaler.fit_transform(train[valid_preds]),
                              columns=valid_preds, index=train.index)
        Xtest  = pd.DataFrame(scaler.transform(test[valid_preds]),
                              columns=valid_preds, index=test.index)
        ytrain = train[target_col]
        ytest  = test[target_col]

        ols = LinearRegression().fit(Xtrain, ytrain)
        ridge = Ridge(alpha=1.0).fit(Xtrain, ytrain)

        for name, model in [("OLS", ols), ("Ridge", ridge)]:
            y_hat = model.predict(Xtest)
            mae   = mean_absolute_error(ytest, y_hat)
            rmse  = mean_squared_error(ytest, y_hat) ** 0.5
            print(f"  [{name}] Test MAE={mae:.4f}  RMSE={rmse:.4f}")

        # Plot forecast vs actual
        y_hat_full_ols   = ols.predict(pd.DataFrame(
            scaler.transform(data[valid_preds]), columns=valid_preds))
        y_hat_full_ridge = ridge.predict(pd.DataFrame(
            scaler.transform(data[valid_preds]), columns=valid_preds))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data["Date"], data[target_col], "o-", color="black", lw=2, label="Actual")
        ax.plot(data["Date"], y_hat_full_ols,   "--", color="#2196F3", lw=1.8, label="OLS Fit")
        ax.plot(data["Date"], y_hat_full_ridge,  ":", color="#F57F17", lw=1.8, label="Ridge Fit")
        ax.axvline(cutoff, color="red", lw=1.2, linestyle="--", alpha=0.6, label=f"Train/Test split ({cutoff.date()})")
        ax.set_ylabel("Householders Loss Ratio")
        ax.set_title("Backtest: OLS & Ridge vs Actual – Householders Loss Ratio", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS + "08_backtest_loss_ratio.png", bbox_inches="tight")
        plt.show()
        print(f"  Saved → {RESULTS}08_backtest_loss_ratio.png")
    else:
        print("  ⚠️  Not enough data for meaningful train/test split")

# ─────────────────────────────────────────────
# 5. STRESS-TEST SCENARIOS
# ─────────────────────────────────────────────
print("\n── Stress-test scenarios ──")

# Use latest available values as base
latest_eco = eco.sort_values("Date").tail(4).mean(numeric_only=True)
print("\n  Base values (4Q average of latest eco data):")
for c in ["cpi", "wpi", "ppi", "cash_rate"]:
    if c in latest_eco.index:
        print(f"    {c:<12}: {latest_eco[c]:.3f}")

scenarios = {
    "Baseline":     {"cpi_delta": 0.0,  "wpi_delta": 0.0,  "ppi_delta": 0.0,  "rate_delta": 0.0},
    "Scenario A\n(CPI +4% pa, Rates +1%)":
                    {"cpi_delta": 4.0,  "wpi_delta": 2.5,  "ppi_delta": 3.0,  "rate_delta": 1.0},
    "Scenario B\n(Stagflation: CPI +5%, Weak GDP)":
                    {"cpi_delta": 5.0,  "wpi_delta": 1.5,  "ppi_delta": 4.5,  "rate_delta": 0.5},
    "Scenario C\n(Rate Cut: -1%, Low Inflation)":
                    {"cpi_delta": -1.0, "wpi_delta": 0.5,  "ppi_delta": -0.5, "rate_delta": -1.0},
}

# Rough linear approximation: each 1% YoY increase in WPI → +X% loss ratio
# Based on regression coefficient (placeholder — replace with fitted model coef)
# Here we use a rule-of-thumb estimate from industry benchmarks
# Claim severity scales ~proportionally to repair/rebuild cost (WPI + PPI weighted)
CLAIM_WPI_SENS  = 0.006   # +1% WPI → +0.6% loss ratio (Householders)
CLAIM_PPI_SENS  = 0.004   # +1% PPI → +0.4% loss ratio
CLAIM_CPI_SENS  = 0.003   # +1% CPI → +0.3% loss ratio
BASE_LOSS_RATIO = 0.72    # approximate current Householders loss ratio

print("\n  Stress-test results:")
scenario_rows = []
for scenario, deltas in scenarios.items():
    delta_lr = (
        deltas["wpi_delta"]  * CLAIM_WPI_SENS
        + deltas["ppi_delta"]  * CLAIM_PPI_SENS
        + deltas["cpi_delta"]  * CLAIM_CPI_SENS
    )
    projected_lr = BASE_LOSS_RATIO + delta_lr
    reserve_delta_pct = delta_lr * 100
    row = {
        "Scenario":        scenario.replace("\n", " "),
        "CPI Δ (% pa)":    deltas["cpi_delta"],
        "WPI Δ (% pa)":    deltas["wpi_delta"],
        "PPI Δ (% pa)":    deltas["ppi_delta"],
        "Rate Δ (pp)":     deltas["rate_delta"],
        "ΔLoss Ratio":     round(delta_lr, 3),
        "Projected LR":    round(projected_lr, 3),
        "Reserve Buffer Δ (pp)": round(reserve_delta_pct, 2),
    }
    scenario_rows.append(row)
    print(f"    {row['Scenario'][:45]:<45} Proj LR={row['Projected LR']:.3f}  Reserve Δ={row['Reserve Buffer Δ (pp)']:+.2f}pp")

scenario_df = pd.DataFrame(scenario_rows)
scenario_df.to_csv(RESULTS + "stress_scenarios.csv", index=False)
print(f"\n  Saved → {RESULTS}stress_scenarios.csv")

# Scenario bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ["#607D8B", "#FF5722", "#9C27B0", "#2196F3"]
labels = [r["Scenario"] for r in scenario_rows]
proj_lrs = [r["Projected LR"] for r in scenario_rows]
reserve_deltas = [r["Reserve Buffer Δ (pp)"] for r in scenario_rows]

bars1 = ax1.bar(range(len(labels)), proj_lrs, color=colors, alpha=0.85, width=0.5)
ax1.axhline(BASE_LOSS_RATIO, color="black", lw=1.2, linestyle="--", alpha=0.6, label=f"Base ({BASE_LOSS_RATIO})")
ax1.axhline(1.0, color="red", lw=1, linestyle=":", alpha=0.5, label="Breakeven (1.0)")
for bar, val in zip(bars1, proj_lrs):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}",
             ha="center", va="bottom", fontsize=9)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Projected Householders Loss Ratio")
ax1.set_title("Projected Loss Ratio by Scenario")
ax1.legend(fontsize=9)
ax1.set_ylim(0.6, 1.0)

bars2 = ax2.bar(range(len(labels)), reserve_deltas, color=colors, alpha=0.85, width=0.5)
ax2.axhline(0, color="black", lw=0.8)
for bar, val in zip(bars2, reserve_deltas):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + 0.05 * np.sign(val) if val != 0 else 0.05,
             f"{val:+.2f}pp", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Required Reserve Buffer Change (percentage points)")
ax2.set_title("Reserve Buffer Δ by Stress Scenario")

plt.suptitle("Stress-Test Scenario Analysis – Householders Insurance (Illustrative)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS + "09_stress_scenarios.png", bbox_inches="tight")
plt.show()
print(f"  Saved → {RESULTS}09_stress_scenarios.png")

# ─────────────────────────────────────────────
# 6. OUTPUT SUMMARY
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("✅ Modeling pipeline complete.")
print(f"   Results saved to: {RESULTS}")
for f in sorted(os.listdir(RESULTS)):
    size = os.path.getsize(os.path.join(RESULTS, f))
    print(f"   {f:<45} {size:>8,} bytes")
