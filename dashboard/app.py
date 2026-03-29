"""
Streamlit Dashboard – Macroeconomic Impact on Insurance Claims
==============================================================
Run:
  cd <project_root>
  streamlit run dashboard/app.py
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Path setup ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Macro-Insurance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    eco = pd.read_csv(os.path.join(PROC, "economic_master.csv"), parse_dates=["Date"])
    ins = pd.read_csv(os.path.join(PROC, "insurance_master.csv"), parse_dates=["Date"])
    model_df = pd.read_csv(os.path.join(PROC, "model_ready.csv"), parse_dates=["Date"])
    return eco, ins, model_df

try:
    eco, ins, model_df = load_data()
    data_ok = True
except FileNotFoundError:
    st.error(
        "Processed data not found. Run `python data_engineering.py` from the project root first."
    )
    data_ok = False
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("Macro-Insurance Analytics")
st.sidebar.markdown("**Macroeconomic Impact on Insurance Claims**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Macro Trends", "Insurance Performance", "Lag Correlation", "Scenario Stress Test"],
    index=0,
)

date_min = eco["Date"].min().date()
date_max = eco["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Filter", value=(date_min, date_max), min_value=date_min, max_value=date_max
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    eco_f = eco[(eco["Date"] >= str(date_range[0])) & (eco["Date"] <= str(date_range[1]))]
else:
    eco_f = eco.copy()

st.sidebar.divider()
st.sidebar.caption(
    "Data: ABS (CPI/WPI/PPI), RBA F1.1, APRA GI Stats Dec 2025\n\n"
    "Note: APRA coverage is Sep 2023–Dec 2025 (10 quarters)."
)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 – MACRO TRENDS
# ═══════════════════════════════════════════════════════════════
if page == "Macro Trends":
    st.title("Macroeconomic Index Trends")
    st.markdown(
        "Australian macro indicators driving insurance claims. "
        "WPI and PPI track repair/rebuild cost inflation; "
        "cash rate signals monetary policy tightening."
    )

    col1, col2, col3, col4 = st.columns(4)
    latest = eco.dropna(subset=["wpi"]).tail(1).iloc[0]
    col1.metric("WPI (latest)", f"{latest.get('wpi', 'N/A'):.1f}",
                delta=f"{latest.get('wpi_yoy', 0):.1f}% YoY" if pd.notna(latest.get("wpi_yoy")) else None)
    latest_ppi = eco.dropna(subset=["ppi"]).tail(1).iloc[0]
    col2.metric("PPI – Construction (latest)", f"{latest_ppi.get('ppi', 'N/A'):.1f}",
                delta=f"{latest_ppi.get('ppi_yoy', 0):.1f}% YoY" if pd.notna(latest_ppi.get("ppi_yoy")) else None)
    latest_rate = eco.dropna(subset=["cash_rate"]).tail(1).iloc[0]
    col3.metric("RBA Cash Rate", f"{latest_rate.get('cash_rate', 'N/A'):.2f}%")
    latest_cpi = eco.dropna(subset=["cpi"]).tail(1).iloc[0]
    col4.metric("CPI – Ins. Subgroup (latest)", f"{latest_cpi.get('cpi', 'N/A'):.2f}",
                delta=f"{latest_cpi.get('cpi_yoy', 0):.1f}% YoY" if pd.notna(latest_cpi.get("cpi_yoy")) else None)

    st.divider()

    # Multi-series line chart
    selected_series = st.multiselect(
        "Select indicators",
        options=["wpi", "ppi", "cash_rate", "cpi"],
        default=["wpi", "ppi", "cash_rate"],
        format_func=lambda x: {
            "wpi": "Wage Price Index (WPI)",
            "ppi": "Producer Price Index – Construction",
            "cash_rate": "RBA Cash Rate (%)",
            "cpi": "CPI – Insurance Subgroup",
        }.get(x, x),
    )

    if selected_series:
        fig = make_subplots(
            rows=len(selected_series), cols=1,
            shared_xaxes=True,
            subplot_titles=[
                {"wpi": "WPI", "ppi": "PPI (Construction)", "cash_rate": "Cash Rate (%)", "cpi": "CPI (Ins.)"}[s]
                for s in selected_series
            ],
        )
        colors = {"wpi": "#2196F3", "ppi": "#4CAF50", "cash_rate": "#FF5722", "cpi": "#9C27B0"}
        for i, col in enumerate(selected_series, 1):
            data = eco_f.dropna(subset=[col])
            fig.add_trace(
                go.Scatter(x=data["Date"], y=data[col], mode="lines", name=col.upper(),
                           line=dict(color=colors.get(col, "#607D8B"), width=2)),
                row=i, col=1,
            )
        fig.update_layout(height=250 * len(selected_series), showlegend=False,
                          title_text="Macroeconomic Indicators Over Time")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("YoY Inflation Rates")
    yoy_cols = [c for c in ["wpi_yoy", "ppi_yoy", "cpi_yoy"] if c in eco_f.columns]
    if yoy_cols:
        fig2 = go.Figure()
        yoy_colors = {"wpi_yoy": "#2196F3", "ppi_yoy": "#4CAF50", "cpi_yoy": "#9C27B0"}
        yoy_labels = {"wpi_yoy": "WPI YoY %", "ppi_yoy": "PPI YoY %", "cpi_yoy": "CPI YoY %"}
        for col in yoy_cols:
            data = eco_f.dropna(subset=[col])
            fig2.add_trace(go.Scatter(x=data["Date"], y=data[col], mode="lines",
                                      name=yoy_labels[col], line=dict(color=yoy_colors[col], width=2)))
        fig2.add_hline(y=3.5, line_dash="dash", line_color="orange", annotation_text="Warning: 3.5%")
        fig2.add_hline(y=4.0, line_dash="dash", line_color="red",    annotation_text="Spike: 4.0%")
        fig2.add_hline(y=0,   line_color="black", line_width=0.8)
        fig2.update_layout(yaxis_title="YoY % Change", hovermode="x unified", height=400)
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 – INSURANCE PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "Insurance Performance":
    st.title("Insurance Performance – APRA Industry Stats")
    st.info(
        "**Coverage:** Sep 2023 – Dec 2025 (10 quarterly periods). "
        "APRA Dec 2025 quarterly GI statistics publication."
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    if "gross_claims_householders" in ins.columns and len(ins) > 0:
        latest_ins = ins.tail(1).iloc[0]
        col1.metric("Householders Claims (latest Q)",
                    f"${latest_ins.get('gross_claims_householders', 0) / 1e9:.2f}B")
        col2.metric("Domestic Motor Claims (latest Q)",
                    f"${latest_ins.get('gross_claims_domestic_motor', 0) / 1e9:.2f}B")
        if "loss_ratio_householders" in ins.columns:
            lr = latest_ins.get("loss_ratio_householders", None)
            col3.metric("Householders Loss Ratio", f"{lr:.3f}" if pd.notna(lr) else "N/A",
                        delta="↑ above 0.85 = warning" if (pd.notna(lr) and lr > 0.85) else "Within range")
        if "loss_ratio_domestic_motor" in ins.columns:
            lr = latest_ins.get("loss_ratio_domestic_motor", None)
            col4.metric("Domestic Motor Loss Ratio", f"{lr:.3f}" if pd.notna(lr) else "N/A")

    st.divider()

    # Gross Claims bar chart
    if "gross_claims_householders" in ins.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ins["Date"].astype(str), y=ins["gross_claims_householders"] / 1e9,
                             name="Householders", marker_color="#1565C0"))
        if "gross_claims_domestic_motor" in ins.columns:
            fig.add_trace(go.Bar(x=ins["Date"].astype(str), y=ins["gross_claims_domestic_motor"] / 1e9,
                                 name="Domestic Motor", marker_color="#F57F17"))
        fig.update_layout(barmode="group", title="Gross Claims Incurred by Quarter ($B)",
                          yaxis_title="$AUD Billion", xaxis_title="Quarter End", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Loss Ratio trend
    if "loss_ratio_householders" in ins.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ins["Date"], y=ins["loss_ratio_householders"], mode="lines+markers",
                                  name="Householders", line=dict(color="#1565C0", width=2)))
        if "loss_ratio_domestic_motor" in ins.columns:
            fig2.add_trace(go.Scatter(x=ins["Date"], y=ins["loss_ratio_domestic_motor"], mode="lines+markers",
                                      name="Domestic Motor", line=dict(color="#F57F17", width=2)))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="red",    annotation_text="Breakeven LR = 1.0")
        fig2.add_hline(y=0.85, line_dash="dot",  line_color="orange", annotation_text="Warning: LR = 0.85")
        fig2.update_layout(title="Loss Ratio (Gross Claims / GWP)", yaxis_title="Loss Ratio", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # GWP trend
    if "gwp_householders" in ins.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=ins["Date"].astype(str), y=ins["gwp_householders"] / 1e9,
                              name="Householders GWP", marker_color="#26A69A"))
        if "gwp_domestic_motor" in ins.columns:
            fig3.add_trace(go.Bar(x=ins["Date"].astype(str), y=ins["gwp_domestic_motor"] / 1e9,
                                  name="Domestic Motor GWP", marker_color="#7E57C2"))
        fig3.update_layout(barmode="group", title="Gross Written Premium by Quarter ($B)",
                           yaxis_title="$AUD Billion", height=380)
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 – LAG CORRELATION
# ═══════════════════════════════════════════════════════════════
elif page == "Lag Correlation":
    st.title("Lead-Lag Correlation Analysis")
    st.markdown(
        "Tests whether macro indicators **today** predict insurance claims "
        "**k quarters later** (i.e., macro *leads* claims). "
        "Spearman rank correlation is used to handle non-normal distributions."
    )

    max_lag = st.slider("Maximum lag (quarters)", min_value=1, max_value=6, value=4)

    macro_options = {
        "wpi":       "WPI (Wage Price Index)",
        "ppi":       "PPI (Construction)",
        "cash_rate": "RBA Cash Rate",
        "cpi":       "CPI (Insurance Subgroup)",
    }
    target_options = {
        "gross_claims_householders":   "Householders – Gross Claims",
        "gross_claims_domestic_motor": "Domestic Motor – Gross Claims",
        "loss_ratio_householders":     "Householders – Loss Ratio",
        "loss_ratio_domestic_motor":   "Domestic Motor – Loss Ratio",
    }

    selected_macro  = [k for k in macro_options  if k in model_df.columns]
    selected_target = [k for k in target_options if k in model_df.columns]

    if not selected_macro or not selected_target:
        st.warning("Run `python data_engineering.py` to generate model_ready.csv first.")
        st.stop()

    def lag_corr(macro_s, target_s, max_lag):
        rows = []
        for lag in range(0, max_lag + 1):
            shifted = macro_s.shift(lag)
            combined = pd.concat([shifted, target_s], axis=1).dropna()
            if len(combined) < 4:
                rows.append({"lag": lag, "r": np.nan, "p": np.nan, "n": len(combined)})
                continue
            r, p = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
            rows.append({"lag": lag, "r": r, "p": p, "n": len(combined)})
        return pd.DataFrame(rows)

    # Build heatmap table
    heatmap_data = {}
    for macro in selected_macro:
        for target in selected_target:
            res = lag_corr(model_df[macro], model_df[target], max_lag)
            key = f"{macro_options[macro]}\n→ {target_options[target]}"
            heatmap_data[key] = res.set_index("lag")["r"]

    hm_df = pd.DataFrame(heatmap_data).T
    hm_df.columns.name = "Lag (quarters)"

    fig = px.imshow(
        hm_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Lead-Lag Spearman Correlation (macro at t-k vs claims at t)",
    )
    fig.update_layout(height=max(300, 80 * len(hm_df)), coloraxis_colorbar_title="r")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Drill-down: Single Macro vs Target")

    col1, col2 = st.columns(2)
    drill_macro  = col1.selectbox("Macro indicator", options=selected_macro,
                                   format_func=lambda x: macro_options[x])
    drill_target = col2.selectbox("Insurance target", options=selected_target,
                                   format_func=lambda x: target_options[x])

    res = lag_corr(model_df[drill_macro], model_df[drill_target], max_lag)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=res["lag"], y=res["r"],
        marker_color=["#388E3C" if p < 0.1 else "#90A4AE" for p in res["p"]],
        text=[f"p={p:.2f}" if pd.notna(p) else "" for p in res["p"]],
        textposition="outside",
        name="Spearman r",
    ))
    fig2.add_hline(y=0, line_color="black", line_width=0.8)
    fig2.update_layout(
        title=f"{macro_options[drill_macro]} → {target_options[drill_target]}",
        xaxis_title="Lag (quarters)",
        yaxis_title="Spearman r",
        yaxis_range=[-1.1, 1.1],
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Raw scatter plot at chosen lag
    chosen_lag = st.slider("Scatter plot at lag", 0, max_lag, 0)
    shifted = model_df[drill_macro].shift(chosen_lag)
    scatter_df = pd.concat([shifted, model_df[drill_target], model_df["Date"]], axis=1).dropna()
    scatter_df.columns = [drill_macro, drill_target, "Date"]
    if len(scatter_df) >= 3:
        fig3 = px.scatter(
            scatter_df, x=drill_macro, y=drill_target,
            hover_data={"Date": True},
            trendline="ols",
            title=f"Scatter: {macro_options[drill_macro]} (lag={chosen_lag}Q) vs {target_options[drill_target]}",
            labels={drill_macro: macro_options[drill_macro], drill_target: target_options[drill_target]},
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 – STRESS TEST SCENARIOS
# ═══════════════════════════════════════════════════════════════
elif page == "Scenario Stress Test":
    st.title("Scenario Stress Test – Loss Ratio Projection")
    st.markdown(
        "Estimate the forward impact of macroeconomic scenarios on "
        "**Householders Loss Ratio** using a linear sensitivity model.\n\n"
        "_Sensitivities are illustrative benchmarks; replace with fitted model coefficients once more data is available._"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Sensitivity parameters")
    wpi_sens  = st.sidebar.number_input("WPI sensitivity  (+1% WPI → Δ loss ratio)", value=0.006, step=0.001, format="%.3f")
    ppi_sens  = st.sidebar.number_input("PPI sensitivity  (+1% PPI → Δ loss ratio)", value=0.004, step=0.001, format="%.3f")
    cpi_sens  = st.sidebar.number_input("CPI sensitivity  (+1% CPI → Δ loss ratio)", value=0.003, step=0.001, format="%.3f")
    base_lr   = st.sidebar.number_input("Base loss ratio",                            value=0.720, step=0.01,  format="%.3f")

    st.divider()
    st.subheader("Define Scenario")

    col1, col2, col3, col4 = st.columns(4)
    cpi_d  = col1.number_input("CPI YoY Δ (%)",   min_value=-5.0, max_value=10.0, value=0.0, step=0.5)
    wpi_d  = col2.number_input("WPI YoY Δ (%)",   min_value=-5.0, max_value=10.0, value=0.0, step=0.5)
    ppi_d  = col3.number_input("PPI YoY Δ (%)",   min_value=-5.0, max_value=10.0, value=0.0, step=0.5)
    rate_d = col4.number_input("Cash Rate Δ (pp)", min_value=-3.0, max_value= 5.0, value=0.0, step=0.25)

    delta_lr     = wpi_d * wpi_sens + ppi_d * ppi_sens + cpi_d * cpi_sens
    projected_lr = base_lr + delta_lr

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("ΔLoss Ratio",     f"{delta_lr:+.3f}")
    col_b.metric("Projected Loss Ratio", f"{projected_lr:.3f}",
                 delta=f"{delta_lr:+.3f} vs base",
                 delta_color="inverse")
    reserve_delta_pp = delta_lr * 100
    col_c.metric("Reserve Buffer Δ", f"{reserve_delta_pp:+.2f} pp",
                 delta_color="inverse")

    # Status indicator
    if projected_lr < 0.75:
        st.success(f"Loss Ratio {projected_lr:.3f} — Healthy range")
    elif projected_lr < 0.90:
        st.warning(f"Loss Ratio {projected_lr:.3f} — Elevated. Monitor closely.")
    else:
        st.error(f"Loss Ratio {projected_lr:.3f} — High risk. Review pricing / reserves.")

    st.divider()
    st.subheader("Compare Preset Scenarios")

    preset_scenarios = {
        "Baseline":           (0.0, 0.0, 0.0, 0.0),
        "Scenario A (CPI+4, Rates+1)": (4.0, 2.5, 3.0, 1.0),
        "Scenario B (Stagflation CPI+5)": (5.0, 1.5, 4.5, 0.5),
        "Scenario C (Rate cut -1%)": (-1.0, 0.5, -0.5, -1.0),
        "Custom": (cpi_d, wpi_d, ppi_d, rate_d),
    }

    rows = []
    for name, (c, w, p, r) in preset_scenarios.items():
        dlr = w * wpi_sens + p * ppi_sens + c * cpi_sens
        rows.append({
            "Scenario": name,
            "CPI Δ": c, "WPI Δ": w, "PPI Δ": p, "Rate Δ": r,
            "Δ Loss Ratio": round(dlr, 3),
            "Projected LR": round(base_lr + dlr, 3),
            "Reserve Δ (pp)": round(dlr * 100, 2),
        })
    sc_df = pd.DataFrame(rows)
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    # Waterfall-style bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sc_df["Scenario"],
        y=sc_df["Projected LR"],
        marker_color=["#607D8B", "#FF5722", "#9C27B0", "#2196F3", "#26A69A"],
        text=sc_df["Projected LR"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
    ))
    fig.add_hline(y=base_lr, line_dash="dash", line_color="black",
                  annotation_text=f"Base LR = {base_lr:.3f}")
    fig.add_hline(y=1.0, line_dash="dot",  line_color="red",
                  annotation_text="Breakeven = 1.0")
    fig.update_layout(
        title="Projected Householders Loss Ratio by Scenario",
        yaxis_title="Loss Ratio", yaxis_range=[0.6, 1.05],
        xaxis_title="", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Action triggers
    st.subheader("Action Triggers (Working Hypothesis)")
    triggers = pd.DataFrame([
        {"Condition": "WPI > 4% YoY for 2+ consecutive quarters",  "Action": "Review Householders repair cost assumptions; flag to pricing"},
        {"Condition": "CPI Insurance subgroup > 3.5% YoY",          "Action": "Consider IBNR reserve buffer top-up of +1.0–1.5%"},
        {"Condition": "PPI Construction > 5% YoY",                  "Action": "Increase rebuilding cost estimates in sum-insured calculations"},
        {"Condition": "Cash Rate spike > 4% + PPI acceleration",     "Action": "Stagflation stress watch: run Scenario B quarterly"},
        {"Condition": "Householders Loss Ratio > 0.85 for 2+ Q",    "Action": "Underwriting strategy review trigger"},
        {"Condition": "Domestic Motor Loss Ratio > 0.80 for 2+ Q",  "Action": "Motor repair network cost audit; review parts inflation"},
    ])
    st.dataframe(triggers, use_container_width=True, hide_index=True)
