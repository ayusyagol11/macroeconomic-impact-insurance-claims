
## Key Findings

### Data Availability
- **WPI / PPI / RBA**: Full history 2015–2026 (11 years, monthly/quarterly)
- **CPI (Insurance subgroup A130393720C)**: Apr 2024 – Feb 2026 only — covers 22 months
- **APRA Industry Stats**: Dec 2025 publication covers Sep 2023 – Dec 2025 (10 quarters)
- **Overlap for correlation**: Sep 2023 – Dec 2025 (10 quarter-end observations)

### Correlation Insights (Spearman, n ≈ 10)
- **WPI** shows [see heatmap] directional relationship with Householders claims — the construction labour cost component is embedded in repair costs
- **PPI (Construction)** tracks closely with WPI; high multicollinearity expected
- **CPI Insurance subgroup** overlaps claims data by only 6 quarters — interpret cautiously
- **Loss Ratio** variation driven primarily by one-off weather events (note Q1 2025 Householders spike = ~$4.7B in claims)

### Regression (n ≈ 10, treat as directional only)
- OLS baseline fits are statistically underpowered with 10 observations
- Ridge regularisation helps stability but overfits are likely
- **Recommendation**: re-run once 2016–2023 historical APRA class-level data is available from legacy GI publications

### Action Triggers (working hypothesis)
| Condition | Suggested Action |
|-----------|------------------|
| WPI > 4% YoY for 2+ consecutive quarters | Flag construction/motor repair cost pressure; review Householders pricing |
| CPI Insurance subgroup > 3.5% YoY | Consider IBNR reserve buffer top-up of +1–1.5% |
| Cash Rate spike >4% in same quarter as PPI acceleration | Stress scenario: stagflation watch |
| Householders Loss Ratio > 0.85 for 2 quarters | Underwriting review trigger |

### Limitations & Caveats
1. **Small-N problem**: 10 quarterly observations is below statistical significance thresholds
2. **Causation vs Correlation**: Rating actions, exposure growth, reinsurance structure, catastrophe events all affect claims independently
3. **CPI subgroup continuity**: Index rebasing post-2020 may affect chain-volume comparability
4. **No policy-level data**: This is industry aggregate only — internal claim frequency and average cost would improve precision
