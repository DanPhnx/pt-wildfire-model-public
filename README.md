# Parametric Catastrophe Loss Model — Portugal Wildfires

Frequency/severity model for mainland Portugal wildfire aggregate losses, built to demonstrate the methods used in reinsurance pricing: distribution fitting, extreme value theory, Monte Carlo simulation, and tail-risk metrics (VaR, Expected Shortfall).

---

## Results

| Metric | Value |
|--------|------:|
| VaR(95%) — 1-in-20 aggregate loss | **€764m** |
| ES(95%) — expected shortfall | €1,380m |
| VaR(99%) — 1-in-100 aggregate loss | €1,777m |
| Mean annual loss | €368m |
| 2017 return period | ~1-in-51 |

All figures in 2025 euros, central EUR/ha scenario (€2,296/ha). Production run: 200,000 scenarios, fixed seed.

---

## Methodology

| Phase | What | Key decision |
|-------|------|-------------|
| 1 — EDA | ICNF PRDF (4,686 fires ≥30 ha, 2009–2025) cross-validated against EFFIS | Fire-day event aggregation as the modelling unit |
| 2 — Fitting | Negative Binomial frequency; Lognormal/GPD severity | Threshold sweep: 90th-pct threshold *failed* QQ criterion (p=0.015); 87th-pct passes (p=0.078) |
| 3 — Simulation | NB × LN/GPD, severity capped at 5× historical max fire-day | Frailty grid search confirmed σ_z=0: independent model appropriate |
| 4 — Validation | Backtest on holdout 2021–2025; sensitivity (±10% shocks); climate scenario | Warming elasticity 26.7%/°C — median of 15 published Mediterranean/Iberian sources |

**Frequency:** NB chosen over Poisson on a formal dispersion test (stat=74.0, p=2.1×10⁻¹¹, var/mean=6.7); AIC gap 39.4 points. Parameters: r=16.78, mean=91.75 fires/yr.

**Severity:** Lognormal body (mu=5.46 log-ha, sigma=1.52) + GPD tail above 1,490 ha (shape ξ=0.786, scale=1,802 ha, 143 exceedances). ξ>0.5 implies infinite theoretical variance — expected for a single-peril catastrophe model.

---

## Backtest

Holdout years placed against the simulated distribution (model fitted on 2009–2020 only):

| Year | Actual loss | Simulated percentile |
|------|------------:|---------------------:|
| 2021 | €48m | 0.0th |
| 2022 | €232m | 29.0th |
| 2023 | €64m | 0.1th |
| 2024 | €304m | 51.4th |
| 2025 | €606m | 91.1th |

No holdout year falls outside the simulated distribution.

---

## Climate scenario

NB mean scaled by a warming→frequency elasticity of 26.7%/°C (median of 15 sources; literature SD ≈21%/°C):

| Scenario | NB mean | VaR(95%) | Change |
|----------|--------:|---------:|-------:|
| Baseline | 91.7/yr | €760m | — |
| +0.5°C | 104.0/yr | €856m | +12.6% |
| +1.0°C | 116.2/yr | €941m | +23.8% |

---

## Code structure

```
main.py                   Entry point — runs phases in order via nbconvert
wildfire_model.py         Data loading, fire-day aggregation, train/holdout split
distribution_fitting.py   Phase 2: NB/LN/GPD fitting, goodness-of-fit, bootstrap CIs
monte_carlo.py            Phase 3: simulation engine, tail cap, risk metrics
validation.py             Phase 4: backtest, sensitivity analysis, climate scenario
```

Logic lives in importable `.py` modules; notebooks are EDA only.

---

## Data

Primary: **ICNF Portuguese Rural Fire Database** (PRDF), Zenodo [DOI 10.5281/zenodo.21427772](https://doi.org/10.5281/zenodo.21427772) — 943 MB GeoPackage, not committed. Download and place at `data/raw/fogos.gpkg`.

EUR/ha calibration from three published anchors (2017 EU Solidarity Fund, ICNF long-run average, 2024 forest-sector loss), all restated in 2025 euros via Eurostat HICP.

---

## Setup

```bash
pip install -r requirements.txt   # Python 3.12
python main.py                    # run all phases
python main.py --phase 3          # single phase
```

See [`docs/technical-note.md`](docs/technical-note.md) for the full methodology writeup.
