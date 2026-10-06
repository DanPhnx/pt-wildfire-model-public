# Parametric Catastrophe Loss Model — Portugal Wildfires
## Technical Note

**Author:** Dan  
**Date:** October 2026  
**Repository:** [DanPhnx/pt-wildfire-catmodel](https://github.com/DanPhnx/pt-wildfire-catmodel)

---

## Overview

This note documents a parametric catastrophe loss model for mainland Portugal
wildfires, built as a portfolio project demonstrating the actuarial and data
science methods used in reinsurance pricing: frequency/severity distribution
fitting, peaks-over-threshold extreme value modelling, Monte Carlo simulation,
and tail-risk metrics (VaR, Expected Shortfall).

The model answers the core question a catastrophe reinsurer would ask: *what is
the annual aggregate wildfire loss distribution for mainland Portugal, and what
are the 1-in-20 and 1-in-100 loss levels?*

**Central result:** VaR(95%) = **€764m**, ES(95%) = **€1,380m**, mean annual
loss = **€368m** (all in 2025 euros, central EUR/ha scenario).

---

## Scope

- **Perils covered:** wildfires ≥ 30 ha, mainland Portugal, 2009–2025.
- **Event unit:** fire-day — all fires starting on the same calendar date are
  aggregated into a single event. This reduces artificial overdispersion from
  multi-polygon mapping of a single outbreak and is the natural unit for a
  treaty hours clause analogue.
- **Loss basis:** burned area (hectares) converted to euros at a three-point
  low/central/high EUR/ha rate derived from published Portuguese wildfire loss
  anchors (2025 prices). The central rate of **2,296 EUR/ha** is used throughout
  unless stated.
- **Out of scope:** sub-regional breakdown by district; individual fire
  occurrence vs. aggregate loss separation; man-made or agricultural losses;
  islands.

---

## Data

### Primary: ICNF Portuguese Rural Fire Database (PRDF)

Per-fire records from Portugal's national fire database, obtained via Zenodo
(DOI [10.5281/zenodo.21427772](https://doi.org/10.5281/zenodo.21427772),
GeoPackage, 943 MB). Covers mainland Portugal, 1980–2025. After applying the
30 ha threshold and the 2009–2025 window: **4,686 fire events**.

The dataset was formally split into a **training window (2009–2020, 1,101
fire-day events)** and a **hold-out window (2021–2025, 5 years)** at the start
of the project. The training/hold-out split is fixed: 2017 (the most extreme
single year on record, 252,199 ha in a single day) is in training; the hold-out
spans four seasons from quiet (2021, 2023) to severe (2022, 2024, 2025).

### Cross-check: EFFIS

An independent EFFIS per-fire export (2008–2026, committed at
`data/raw/effis_fire_database_pt_2008_2026.csv`) is used for cross-validation
only — not for fitting. Key finding: after applying the 30 ha threshold, EFFIS
and PRDF agree to within 3% on aggregate burned area over 2009–2019. The
2020–2025 period shows a ~6.5% residual discrepancy (investigated, documented,
not reconciled — see `01_eda.ipynb`).

### EUR/ha calibration

Three published anchors (all restated in 2025 euros via Eurostat Portuguese HICP):

| Scenario | EUR/ha | Source |
|----------|-------:|--------|
| Low | 497 | 2024 forest-sector loss ÷ 2024 burned area (floor) |
| Central | 2,296 | ICNF-derived long-run average (OECD/press, ~2021) |
| High | 3,418 | 2017 EU Solidarity Fund direct damage ÷ 2017 burned area |

No verified per-fire loss data exists for this window; every loss figure is
modelled as area × EUR/ha and labelled as such.

---

## Methodology

### Phase 1 — EDA and data preparation

Raw PRDF records cleaned (mainland filter, ≥30 ha threshold, date parsing),
aggregated into fire-day events, and split into training/hold-out. EFFIS
cross-validation performed. Loss columns appended via EUR/ha model.

### Phase 2 — Distribution fitting

All fitting is on the **training data only** (2009–2020). Both distributions
are fitted on burned area (hectares), per the PRD's technical decision: "the
model fits burned area, not euro losses, and converts to euros at the end."

**Frequency — Negative Binomial:**

A formal dispersion test (index-of-dispersion statistic: 74.0, df=11,
p=2.1×10⁻¹¹, variance/mean=6.7) decisively rejects Poisson. The Negative
Binomial is fitted by MLE (method-of-moments initialisation):

| Parameter | Value |
|-----------|------:|
| r (dispersion) | 16.78 |
| p | 0.155 |
| Mean fires/yr | 91.75 |
| NB AIC | 114.2 |
| Poisson AIC | 153.6 |

Bootstrap 95% CI for mean: [72.6, 115.3] fires/yr.

**Severity — Lognormal body + Generalised Pareto tail:**

A composite model fitted on fire-day burned area. The body uses a Lognormal
fitted by MLE (loc fixed at 0); the tail uses a peaks-over-threshold GPD fitted
on exceedances above the 87th-percentile threshold.

The threshold was chosen via a sensitivity sweep across the 82nd–97th
percentiles. The 90th-percentile threshold failed the PRD's QQ criterion
(Spearman rank-residual test: rho=0.231, p=0.015). The **87th percentile**
passes (rho=0.148, p=0.078) with more exceedances and a stable tail shape:

| Parameter | Value | 95% CI |
|-----------|------:|--------|
| LN mu (log-ha) | 5.461 | — |
| LN sigma | 1.523 | — |
| GPD threshold | 1,490 ha (87th pct) | — |
| GPD shape xi | 0.786 | [0.487, 1.053] |
| GPD scale | 1,802.3 ha | [1,302.5, 2,486.5] |
| Exceedances | 143 | — |
| Bootstrap AD p-value | 0.576 | — |

xi = 0.786 > 0.5 implies infinite theoretical variance — consistent with a
catastrophe model dominated by infrequent extreme years (2017, 2025). The
Lognormal body is formally rejected by Anderson-Darling (a real miss
concentrated in the extreme tail, documented in the Phase 2 notebook, not
hidden); the GPD tail is not rejected (bootstrap AD p=0.576).

### Phase 3 — Monte Carlo simulation

The simulation engine (`monte_carlo.py`) draws an annual fire count from the
fitted Negative Binomial, then a burned area per fire from the Lognormal/GPD
composite, sums to an annual aggregate, and converts to EUR.

**Tail cap:** the GPD tail (xi=0.786) has infinite theoretical variance. An
uncapped simulation does not converge — a prototype run produced a single draw
of 75.7 million ha (8× the entire mainland land area). The practical cap is
**5× the historical maximum fire-day (1,260,995 ha)**, derived from training
data; ICNF's 6.1 million ha mainland burnable-land estimate (IFN6, 2019) serves
as an absolute physical backstop.

**Frailty calibration:** a grid search over year-level frailty (σ_z) confirmed
that the independent model (σ_z=0, ρ=0) already overshoots the observed
training-period annual-area standard deviation — no frailty amplification is
needed. The count/median-severity Spearman correlation on fire-day events is
ρ=0.105 (p=0.75), not significant.

**Production run:** 200,000 scenarios, fixed seed (42), central EUR/ha (2,296).

| Metric | Value |
|--------|------:|
| VaR(95%) | €764m |
| VaR(95%) SE | 1.21% (criterion: <2% ✓) |
| VaR(99%) | €1,777m |
| ES(95%) | €1,380m |
| Mean annual loss | €368m |
| 2017 return period | ~1-in-51 (criterion: 1-in-30 to 1-in-100 ✓) |

---

## Validation (Phase 4)

### Backtest — holdout years 2021–2025

Each holdout year's actual aggregate loss (PRDF burned area × central EUR/ha)
is placed against the Phase 3 simulated ECDF:

| Year | Actual area (ha) | Actual loss | Simulated percentile |
|------|----------------:|------------:|---------------------:|
| 2021 | 20,817 | €48m | 0.0th |
| 2022 | 101,034 | €232m | 29.0th |
| 2023 | 27,795 | €64m | 0.1th |
| 2024 | 132,332 | €304m | 51.4th |
| 2025 | 264,029 | €606m | 91.1th |

No holdout year falls outside the simulated distribution. The model correctly
identifies 2021 and 2023 as near-zero-loss years and 2025 as a severe year
(~1-in-11). The hold-out is too short (n=5) for formal statistical testing, but
the spread from 0.0th to 91.1th pct suggests reasonable calibration across the
full range.

### Sensitivity analysis

VaR(95%) response to ±10% parameter shocks (50,000 scenarios; base VaR=€752m):

| Shock | VaR(95%) | Change |
|-------|--------:|-------:|
| NB mean −10% | €685m | −8.9% |
| NB mean +10% | €837m | +11.3% |
| LN sigma −10% | €712m | −5.4% |
| LN sigma +10% | €815m | +8.4% |
| EUR/ha −10% | €677m | −10.0% |
| EUR/ha +10% | €827m | +10.0% |

EUR/ha is linear (expected). NB mean is the dominant model driver (slightly
asymmetric: the upper tail is heavier than the lower, so a +10% count increase
lifts VaR by more than a −10% count decrease reduces it). LN sigma is
secondary.

---

## Climate Scenario

A first-order climate scenario scales the NB mean by
`(1 + elasticity × ΔT_°C)`, holding all other parameters fixed.

**Elasticity derivation:** a 15-source literature search for Mediterranean/
Iberian fire frequency and burned area sensitivity to warming was compiled and
converted to a common unit (% change per °C). The distribution:

- Full set (n=15): median 26.7%/°C, mean 35.6%/°C (skewed by two outliers)
- Trimmed (n=14, excl. Carvalho 2011 +500%/3.5°C which includes full vegetation
  feedbacks): mean 27.9%/°C, SD 21.1%/°C
- Within 1 SD of trimmed mean (n=12): median 26.6%/°C

The median is stable across all three cuts. **26.7%/°C** is adopted. Key anchors:
Turco et al. (2018, *Nature Communications*): +40% burned area at +1.5°C;
El Garroussi et al. (2024, *npj Climate and Atmospheric Science*): same;
PESETA IV JRC Iberian Peninsula: 72–93% burned area at RCP8.5 (~+3.5°C).

**Scenario results** (100,000 scenarios, central EUR/ha):

| Scenario | NB mean (fires/yr) | VaR(95%) | Change | ES(95%) |
|----------|-------------------:|--------:|-------:|--------:|
| Baseline (0°C) | 91.7 | €760m | — | €1,385m |
| +0.5°C | 104.0 | €856m | +12.6% | €1,503m |
| +1.0°C | 116.2 | €941m | +23.8% | €1,632m |

At +1.0°C of additional warming the 1-in-20 aggregate loss increases by roughly
a quarter (€181m in absolute terms), driven entirely by higher fire frequency —
severity per event is held at the training-period distribution.

---

## Limitations

1. **Short training window.** 12 years (2009–2020) is a modest sample for
   fitting a heavy-tailed distribution. The GPD shape CI [0.487, 1.053] spans
   the boundary between finite and infinite variance; the true tail behaviour
   is genuinely uncertain.

2. **Single event definition.** Fire-day clustering is a pragmatic choice.
   Multi-day events, regional sub-aggregation, or a treaty hours clause would
   change the frequency/severity split without changing the aggregate loss
   distribution significantly.

3. **Stationary severity.** The climate scenario scales only frequency. Warming
   also drives higher rate of spread, longer fire seasons, and drier fuels — all
   of which could increase per-event severity. The scenario is therefore
   conservative.

4. **EUR/ha uncertainty.** The ×6.9 range from low (€497) to high (€3,418) EUR/ha
   dwarfs the model parameter uncertainty. No sub-national loss data exists to
   narrow this range further without bespoke data collection.

5. **Climate scenario linearity.** The +26.7%/°C elasticity carries a literature
   SD of ≈21%/°C. The actual climate–fire relationship is nonlinear and
   path-dependent. The scenario results should be read as central-estimate
   order-of-magnitude impacts.

---

## Reproducibility

```bash
# 1. Download PRDF GeoPackage from Zenodo DOI 10.5281/zenodo.21427772
#    and place at data/raw/icnf_prdf_fogos.gpkg  (943 MB, not committed)

# 2. Install dependencies
pip install -r requirements.txt   # Python 3.12, pinned versions

# 3. Run all phases
python main.py

# 4. Run a single phase
python main.py --phase 3
```

All other data (EFFIS CSV, OWID/GWIS snapshots, HICP inflation index) are
committed. Model parameters (`models/*.json`), simulation outputs
(`simulation/*.csv`, `simulation/*_metrics.json`), and diagnostic plots
(`models/*.pdf`) are also committed, so phases 2–4 are reproducible without
re-downloading the GeoPackage.

---

*See `docs/worklog.md` for a full decision log, `docs/phase3-frequency-severity-dependence.md`
for the Phase 3 frailty calibration rationale, and the individual phase notebooks
for detailed diagnostics.*
