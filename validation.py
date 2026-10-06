"""Phase 4 validation, sensitivity analysis, and climate scenario logic.

Per the PRD's 'notebooks for EDA only' pattern: 04_validation.ipynb imports
and calls these functions rather than defining the logic itself, matching
the pattern set by distribution_fitting.py (Phase 2) and monte_carlo.py
(Phase 3).

Import from a notebook with:

    import sys
    sys.path.insert(0, "..")
    from validation import compute_holdout_percentiles, run_sensitivity_analysis, ...
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = Path("../models")
SIMULATION_DIR = Path("../simulation")

# Warming→fire-frequency elasticity: median of 15 peer-reviewed sources
# for Mediterranean/Iberian fire frequency and burned area vs. warming.
# Distribution (14 sources, outlier Carvalho 2011 +500%/3.5°C excluded):
#   mean=27.9%/°C, SD=21.1%/°C, mean±1SD = [6.8, 49.1]%/°C
#   12 sources fall within 1 SD; median of that subset = 26.6%/°C.
# Full source table: notebooks/04_validation.ipynb "Climate scenario" section.
# Key anchors: Turco 2018 NatCommun (+40% at +1.5°C), El Garroussi 2024 npj
#   (same), PESETA IV JRC Iberia (72-93%/3.5°C). Median is stable across
#   full set, trimmed set, and within-1SD subset — all return 26.6-26.7%/°C.
WARMING_ELASTICITY_PER_C = 0.267


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _scale_nb_mean(nb_params: dict, factor: float) -> dict:
    """Return NB params with mean scaled by `factor`; r (dispersion) held fixed.

    The NB mean = r(1-p)/p; to scale the mean by `factor` without changing
    the dispersion structure (r fixed), solve for the new p:
        new_p = r / (r + factor * mean)
    """
    r = nb_params["r"]
    new_mean = nb_params["mean"] * factor
    return {"r": r, "p": r / (r + new_mean), "mean": new_mean}


def training_simulation_params(prdf_train_df: pd.DataFrame, gpd_params: dict) -> tuple:
    """Derive tail_probability and severity cap_ha from PRDF training data.

    Parameters
    ----------
    prdf_train_df : pd.DataFrame
        PRDF training records with Date and Burned_Area_ha columns
        (from load_icnf_prdf_database + split_train_holdout).
    gpd_params : dict
        Fitted GPD parameters (models/pareto_tail.json); needs n_exceedances.

    Returns
    -------
    (float, float)
        (tail_probability, cap_ha).
        tail_probability = n_exceedances / n_training_fire_day_events.
        cap_ha = 5× the largest observed training fire-day (same formula as
        historical_severity_cap in monte_carlo.py).
    """
    from monte_carlo import historical_severity_cap, DEFAULT_CAP_MULTIPLIER

    day = prdf_train_df["Date"].dt.normalize()
    daily_area = prdf_train_df.groupby(day)["Burned_Area_ha"].sum().values

    tail_probability = gpd_params["n_exceedances"] / len(daily_area)
    cap_ha = historical_severity_cap(daily_area, DEFAULT_CAP_MULTIPLIER)

    return float(tail_probability), float(cap_ha)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def compute_holdout_percentiles(simulated_losses: np.ndarray,
                                 holdout_area_by_year: pd.Series,
                                 eur_per_ha: float) -> pd.DataFrame:
    """Translate holdout annual burned areas to losses and find each year's
    percentile in the production simulation distribution.

    Parameters
    ----------
    simulated_losses : np.ndarray
        Annual aggregate losses from the Phase 3 production run (EUR).
    holdout_area_by_year : pd.Series
        Index = year (int), values = total annual burned area (ha).
        Typically: prdf_holdout.groupby(year)["Burned_Area_ha"].sum().
    eur_per_ha : float
        EUR/ha conversion (central scenario: 2296 EUR/ha in 2025 prices).
        Must match the value used in the production simulation.

    Returns
    -------
    pd.DataFrame
        Columns: year, actual_area_ha, actual_loss_eur, simulated_pct,
        simulated_return_period_yr.
    """
    rows = []
    for year, area in holdout_area_by_year.sort_index().items():
        actual = float(area) * eur_per_ha
        pct = float(np.mean(simulated_losses <= actual) * 100)
        rp = 1.0 / max(1.0 - pct / 100.0, 1e-6)
        rows.append({
            "year": int(year),
            "actual_area_ha": round(float(area), 0),
            "actual_loss_eur": round(actual, 0),
            "simulated_pct": round(pct, 1),
            "simulated_return_period_yr": round(rp, 1),
        })
    return pd.DataFrame(rows)


def plot_backtest_comparison(holdout_results: pd.DataFrame,
                              simulated_losses: np.ndarray, ax=None):
    """Simulated loss ECDF with holdout years plotted as labelled points.

    Parameters
    ----------
    holdout_results : pd.DataFrame
        Output of compute_holdout_percentiles.
    simulated_losses : np.ndarray
        Full simulation array for the ECDF curve.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    sorted_losses = np.sort(simulated_losses)
    ecdf_pcts = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses) * 100
    ax.plot(sorted_losses / 1e6, ecdf_pcts, color="steelblue", lw=1.5,
            label=f"Simulated ECDF ({len(sorted_losses):,} scenarios)")

    palette = plt.cm.tab10.colors
    for i, row in holdout_results.iterrows():
        year = int(row["year"])
        loss_m = row["actual_loss_eur"] / 1e6
        pct = row["simulated_pct"]
        ax.scatter(loss_m, pct, s=90, color=palette[i % 10], zorder=5,
                   label=f"{year}: €{loss_m:.0f}m ({pct:.1f}th pct)")
        ax.annotate(str(year), (loss_m, pct), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color=palette[i % 10])

    ax.set_xlabel("Aggregate annual loss (€m)")
    ax.set_ylabel("Simulated percentile")
    ax.set_title("Backtest: holdout years (2021–2025) against the Phase 3 simulated distribution\n"
                 "(model fitted on 2009–2020 PRDF training data only)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(nb_params: dict, lognormal_params: dict, gpd_params: dict,
                              tail_probability: float, cap_ha: float, eur_per_ha: float,
                              n_scenarios: int = 50_000, seed: int = 42) -> tuple:
    """Re-run the simulation under ±10% shocks on NB mean, LN sigma, and EUR/ha.

    Uses 50k scenarios by default (vs. 200k in production) — sufficient for
    measuring relative VaR(95%) changes; absolute accuracy is not the goal here.

    Parameters
    ----------
    nb_params : dict
        Baseline NB frequency parameters (models/negative_binomial_frequency.json).
    lognormal_params : dict
        Baseline Lognormal severity body (models/lognormal_severity.json).
    gpd_params : dict
        Baseline GPD severity tail (models/pareto_tail.json).
    tail_probability : float
        Fraction of training fire-day events above the GPD threshold
        (from training_simulation_params).
    cap_ha : float
        Per-event severity cap (from training_simulation_params).
    eur_per_ha : float
        Central EUR/ha conversion rate.
    n_scenarios : int
        Scenarios per shocked run.
    seed : int
        Fixed seed.

    Returns
    -------
    (pd.DataFrame, float)
        sensitivity_df : columns shock, var95_eur, delta_pct.
        base_var95     : unshocked VaR(95%) at n_scenarios.
    """
    from monte_carlo import run_monte_carlo, compute_risk_metrics

    base_losses, _ = run_monte_carlo(
        nb_params, lognormal_params, gpd_params, tail_probability,
        eur_per_ha, cap_ha, n_scenarios=n_scenarios, seed=seed,
    )
    base_var95 = compute_risk_metrics(base_losses)["VaR_95"]

    shocks = [
        ("NB mean −10%",
         _scale_nb_mean(nb_params, 0.9), lognormal_params,                                          gpd_params, eur_per_ha),
        ("NB mean +10%",
         _scale_nb_mean(nb_params, 1.1), lognormal_params,                                          gpd_params, eur_per_ha),
        ("LN sigma −10%",
         nb_params,                      {**lognormal_params, "shape": lognormal_params["shape"] * 0.9}, gpd_params, eur_per_ha),
        ("LN sigma +10%",
         nb_params,                      {**lognormal_params, "shape": lognormal_params["shape"] * 1.1}, gpd_params, eur_per_ha),
        ("EUR/ha −10%",
         nb_params,                      lognormal_params,                                          gpd_params, eur_per_ha * 0.9),
        ("EUR/ha +10%",
         nb_params,                      lognormal_params,                                          gpd_params, eur_per_ha * 1.1),
    ]

    rows = []
    for label, nb, ln, gpd, eur in shocks:
        losses, _ = run_monte_carlo(nb, ln, gpd, tail_probability, eur, cap_ha,
                                     n_scenarios=n_scenarios, seed=seed)
        var95 = compute_risk_metrics(losses)["VaR_95"]
        rows.append({
            "shock": label,
            "var95_eur": round(var95, 0),
            "delta_pct": round(100.0 * (var95 - base_var95) / base_var95, 1),
        })

    return pd.DataFrame(rows), base_var95


def plot_sensitivity_tornado(sensitivity_df: pd.DataFrame, base_var95: float, ax=None):
    """Horizontal tornado chart of VaR(95%) change under ±10% parameter shocks.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Output of run_sensitivity_analysis.
    base_var95 : float
        Unshocked VaR(95%) in EUR (for the subtitle).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    params_order = ["NB mean", "LN sigma", "EUR/ha"]
    y_pos = np.arange(len(params_order))
    down_vals, up_vals = [], []
    for param in params_order:
        down = sensitivity_df.loc[sensitivity_df["shock"] == f"{param} −10%", "delta_pct"].values[0]
        up   = sensitivity_df.loc[sensitivity_df["shock"] == f"{param} +10%",   "delta_pct"].values[0]
        down_vals.append(down)
        up_vals.append(up)

    ax.barh(y_pos - 0.2, down_vals, height=0.35, color="tomato",    label="−10% shock")
    ax.barh(y_pos + 0.2, up_vals,   height=0.35, color="steelblue", label="+10% shock")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params_order)
    ax.set_xlabel("Change in VaR(95%) (%)")
    ax.set_title(f"Sensitivity tornado — base VaR(95%): €{base_var95/1e6:.0f}m  (50k scenarios)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    return ax


# ---------------------------------------------------------------------------
# Climate scenario
# ---------------------------------------------------------------------------

def apply_climate_scenario(base_nb_params: dict, warming_c: float,
                            elasticity_per_c: float = WARMING_ELASTICITY_PER_C) -> dict:
    """Scale the NB mean by (1 + elasticity_per_c * warming_c); r held fixed.

    The elasticity is the median of 15 published sources for Mediterranean/
    Iberian fire frequency vs. temperature increase (26.7%/°C — see
    WARMING_ELASTICITY_PER_C). Only the NB mean (expected annual fire-day
    event count) is scaled; the dispersion parameter r, the severity
    distribution, and the EUR/ha conversion are all held constant, making
    this a conservative first-order estimate.

    Parameters
    ----------
    base_nb_params : dict
        Baseline NB parameters (r, p, mean).
    warming_c : float
        Degrees of additional warming above the baseline (e.g. 0.5, 1.0).
    elasticity_per_c : float
        % increase in NB mean per °C. Default = WARMING_ELASTICITY_PER_C.

    Returns
    -------
    dict
        NB params with scaled mean and derived p; r unchanged.
    """
    return _scale_nb_mean(base_nb_params, 1.0 + elasticity_per_c * warming_c)


def run_climate_scenarios(nb_params: dict, lognormal_params: dict, gpd_params: dict,
                           tail_probability: float, cap_ha: float, eur_per_ha: float,
                           warming_levels: list = None,
                           n_scenarios: int = 100_000, seed: int = 42) -> dict:
    """Run baseline and warming-adjusted scenarios; return risk metrics per scenario.

    Parameters
    ----------
    warming_levels : list of float
        Degrees Celsius above baseline to model (default [0.5, 1.0] per PRD).
    n_scenarios : int
        Scenarios per run. Default 100k — faster than the production 200k,
        sufficient for scenario comparison (SE ~1.7% at 100k).

    Returns
    -------
    dict
        {label: metrics_dict} where metrics_dict is from compute_risk_metrics,
        augmented with 'nb_mean' for the scenario's expected annual fire count
        and 'warming_c' for the scenario's warming level.
    """
    from monte_carlo import run_monte_carlo, compute_risk_metrics

    if warming_levels is None:
        warming_levels = [0.5, 1.0]

    scenarios = {"Baseline (0°C)": (nb_params, 0.0)}
    for w in warming_levels:
        scenarios[f"+{w}°C"] = (apply_climate_scenario(nb_params, w), w)

    results = {}
    for label, (nb, w) in scenarios.items():
        losses, _ = run_monte_carlo(
            nb, lognormal_params, gpd_params, tail_probability,
            eur_per_ha, cap_ha, n_scenarios=n_scenarios, seed=seed,
        )
        m = compute_risk_metrics(losses)
        m["nb_mean"] = nb["mean"]
        m["warming_c"] = w
        results[label] = m

    return results


def plot_climate_scenarios(scenario_results: dict, ax=None):
    """Grouped bar chart: VaR(95%) and ES(95%) across warming scenarios.

    Parameters
    ----------
    scenario_results : dict
        Output of run_climate_scenarios.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    labels = list(scenario_results.keys())
    var95 = [scenario_results[l]["VaR_95"] / 1e6 for l in labels]
    es95  = [scenario_results[l]["ES_95"]  / 1e6 for l in labels]
    means = [scenario_results[l]["nb_mean"]       for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    bars_v = ax.bar(x - w / 2, var95, w, label="VaR(95%)", color="steelblue")
    bars_e = ax.bar(x + w / 2, es95,  w, label="ES(95%)",  color="tomato")

    for xi, (v, e) in zip(x, zip(var95, es95)):
        ax.text(xi - w / 2, v + 5, f"€{v:.0f}m", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, e + 5, f"€{e:.0f}m", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\n({m:.1f} fires/yr)" for l, m in zip(labels, means)],
        fontsize=9,
    )
    ax.set_ylabel("Annual aggregate loss (€m)")
    ax.set_title(
        f"Climate scenario impact on risk metrics\n"
        f"Elasticity: {WARMING_ELASTICITY_PER_C:.1%}/°C "
        f"(median of 15 sources, Mediterranean/Iberia)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return ax
