"""Phase 3 Monte Carlo engine: frequency-severity simulation, tail-cap and
year-level frailty (dependence) handling, risk metrics, and diagnostic plots.

Extracted as an importable module up front, matching distribution_fitting.py's
pattern for Phase 2, per the PRD's Technical Decisions ("Entry point: one
main.py; notebooks for EDA only") - 03_monte_carlo.ipynb imports and calls
these functions rather than defining the simulation logic itself.

Implements the design set out in docs/phase3-frequency-severity-dependence.md:
- Frequency: Negative Binomial (Phase 2 rejected Poisson decisively - see
  models/frequency_model_choice.json).
- Severity: Lognormal body + Generalised Pareto tail, in **hectares**, not
  euros - the PRD fits burned area and converts to euros as an explicit
  final step (see run_monte_carlo). The GPD tail (shape xi=0.746) has
  infinite theoretical variance (finite only for xi<0.5), so every severity
  draw is capped - see PHYSICAL_CEILING_HA and DEFAULT_CAP_MULTIPLIER.
- Dependence: a year-level lognormal frailty factor (mean 1, sigma_z=0.25,
  rho=0), reproducing the observed annual-variance gap. The copula rho is
  set to 0: the count/median-severity correlation is not significant at the
  fire-day event level used by this engine (see calibration note in
  simulate_dependent_frequency_and_frailty).

Import from a notebook (which runs with the notebook's own directory as its
working directory) with:

    import sys
    sys.path.insert(0, "..")
    from monte_carlo import run_monte_carlo, compute_risk_metrics, ...
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

MODELS_DIR = Path("../models")
SIMULATION_DIR = Path("../simulation")

# PRD Technical Decisions: "100,000 years, fixed random seed - 10,000 runs is
# noisy at VaR(99)". Increased to 200,000 after the 87th-pct threshold recalibration
# reduced xi from 0.849 to 0.786, pushing SE(VaR95%) to 2.01% at 100k (marginal
# failure). 200k brings SE below 1.5%. SEED is fixed for reproducibility (PRD).
N_SCENARIOS = 200_000
SEED = 42

# Physical ceiling on any single simulated fire-day: mainland Portugal's forest,
# shrubland and unproductive land ("espacos florestais") per ICNF's 6th National
# Forest Inventory (IFN6, 2015 survey data, published June 2019) - 6.1 million
# ha, 69.4% of the 8.91 million ha mainland. A single fire-day cannot physically
# exceed this. Hard-coded as an absolute backstop; not expected to bind given
# DEFAULT_CAP_MULTIPLIER below, which is the cap that actually stabilises the
# simulation. See docs/phase3-frequency-severity-dependence.md for why this
# exists: an uncapped simulation from the fitted GPD tail does not converge and
# can generate non-physical draws (one prototype run reached 75.7 million ha,
# over 8x the mainland's total land area).
PHYSICAL_CEILING_HA = 6_100_000.0

# Practical simulation bound, as a multiple of the observed historical maximum
# fire-day (computed from data, not hardcoded here - see historical_severity_cap
# below). Chosen because it stabilises cross-seed simulated annual SD to within
# ~2.6% while remaining far inside PHYSICAL_CEILING_HA and still letting the
# model exceed anything on record, which is the point of fitting an EVT tail.
DEFAULT_CAP_MULTIPLIER = 5.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model_params(filename: str) -> dict:
    """Load fitted distribution parameters saved by 02_distribution_fitting.ipynb.

    Parameters
    ----------
    filename : str
        JSON filename under MODELS_DIR, e.g. "negative_binomial_frequency.json",
        "lognormal_severity.json", "pareto_tail.json", or
        "severity_euro_equivalents.json".

    Returns
    -------
    dict
        Parameter dictionary as saved by distribution_fitting.save_model_params.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/02_distribution_fitting.ipynb "
            "(or `python main.py --phase 2`) first."
        )
    with open(path) as f:
        return json.load(f)


def historical_severity_cap(historical_areas: np.ndarray, multiplier: float = DEFAULT_CAP_MULTIPLIER) -> float:
    """Derive the practical severity cap from the actual historical record, rather
    than hardcoding a number that would go stale if the data changes (e.g. a
    future EFFIS re-export extending the record).

    Parameters
    ----------
    historical_areas : np.ndarray
        Observed per-fire-day burned area (hectares), training years - i.e.
        train["Burned_Area_ha"] from wildfire_model.build_fire_day_events /
        split_train_holdout.
    multiplier : float
        How many times the historical maximum to allow (default 5x, per
        docs/phase3-frequency-severity-dependence.md).

    Returns
    -------
    float
        The cap in hectares, always <= PHYSICAL_CEILING_HA (a multiplier large
        enough to exceed it would defeat the point of a practical, tighter cap,
        so it is clipped there as a safety net).
    """
    cap = multiplier * float(np.max(historical_areas))
    return min(cap, PHYSICAL_CEILING_HA)


# ---------------------------------------------------------------------------
# Frequency and year-level frailty (dependence)
# ---------------------------------------------------------------------------

def simulate_dependent_frequency_and_frailty(nb_params: dict, sigma_z: float, rho: float,
                                              n_scenarios: int, rng: np.random.Generator) -> tuple:
    """Jointly sample annual fire counts and a year-level severity frailty factor,
    linked by a Gaussian copula.

    Draws two correlated standard Normal variates per scenario-year (correlation
    rho); the first is mapped through the fitted Negative Binomial's inverse CDF
    to get the fire count, the second through a Lognormal (mean 1) to get the
    frailty multiplier Z applied to every severity draw that year. rho=0 makes
    the two independent (frailty still inflates variance; no count/severity
    correlation).

    Calibration (confirmed against training data 2009-2020): sigma_z=0.25 at
    rho=0 matches the observed training-year annual-area SD (147,411 ha) almost
    exactly (ratio 0.999). rho is set to 0 because the count/median-severity
    Spearman correlation, when computed on the fire-day events the model
    actually uses, is 0.105 (p=0.75) on training data and 0.25 (p=0.33) on the
    full 2009-2025 record - neither significant. The 0.57 figure cited in the
    planning doc (docs/phase3-frequency-severity-dependence.md) was computed on
    individual fire polygons before the fire-day clustering step, not on the
    event unit the engine uses; it does not carry over.

    Parameters
    ----------
    nb_params : dict
        Fitted Negative Binomial frequency parameters (models/negative_binomial_frequency.json;
        keys "r", "p").
    sigma_z : float
        Spread of the lognormal frailty factor (mean fixed at 1, i.e.
        mu_z = -0.5*sigma_z**2). 0 disables frailty (independent, no variance
        inflation from this mechanism).
    rho : float
        Gaussian copula correlation between the count draw and the frailty draw.
        0 makes them independent (frailty still applies, just not linked to
        whether the year was high- or low-count).
    n_scenarios : int
        Number of scenario-years to simulate.
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    (np.ndarray, np.ndarray)
        (fire_counts, frailty_factors), each shape (n_scenarios,). fire_counts
        are non-negative integers; frailty_factors have mean 1 by construction
        (exactly 1 everywhere if sigma_z=0).
    """
    g_count = rng.standard_normal(n_scenarios)
    g_frailty = rho * g_count + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.standard_normal(n_scenarios)
    fire_counts = stats.nbinom.ppf(stats.norm.cdf(g_count), nb_params["r"], nb_params["p"]).astype(int)
    if sigma_z > 0:
        mu_z = -0.5 * sigma_z ** 2
        frailty_factors = np.exp(mu_z + sigma_z * g_frailty)
    else:
        frailty_factors = np.ones(n_scenarios)
    return fire_counts, frailty_factors


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

def simulate_severities(n_fires: int, lognormal_params: dict, gpd_params: dict, tail_probability: float,
                         rng: np.random.Generator, cap_ha: float = None) -> np.ndarray:
    """Sample per-fire burned area from the fitted Lognormal body + GPD tail.

    Each draw comes from the Lognormal body below the fitted threshold, or the
    GPD tail above it, with probability `tail_probability` of the latter (the
    actual historical proportion of events exceeding the threshold - do not
    hardcode this, since it depends on the threshold percentile chosen in
    02_distribution_fitting.ipynb; compute it as
    gpd_params["n_exceedances"] / n_training_events and pass it in). Output is
    in **hectares**, not euros - the PRD fits burned area and converts to
    euros as a separate final step (see run_monte_carlo).

    Parameters
    ----------
    n_fires : int
        Number of fire events to sample severities for.
    lognormal_params : dict
        Fitted Lognormal body (models/lognormal_severity.json; keys "shape",
        "loc", "scale").
    gpd_params : dict
        Fitted GPD tail (models/pareto_tail.json; keys "shape", "loc", "scale",
        "threshold").
    tail_probability : float
        Probability a given draw comes from the GPD tail rather than the
        Lognormal body (~0.10 for the current 90th-percentile threshold - see
        the note above for how to derive it precisely rather than hardcoding).
    rng : np.random.Generator
        Random generator for reproducibility.
    cap_ha : float, optional
        Maximum allowed single-fire severity (hectares), applied to tail draws
        only (the Lognormal body essentially never approaches this scale). See
        historical_severity_cap. If omitted, only PHYSICAL_CEILING_HA applies -
        docs/phase3-frequency-severity-dependence.md found that alone is too
        loose to converge, so callers should normally pass a tighter value.

    Returns
    -------
    np.ndarray
        Shape (n_fires,) array of simulated per-event burned area (ha) - not
        yet converted to euros; see the docstring note above.
    """
    is_tail = rng.random(n_fires) < tail_probability
    severities = np.empty(n_fires)
    n_body, n_tail = int((~is_tail).sum()), int(is_tail.sum())
    if n_body:
        severities[~is_tail] = stats.lognorm.rvs(
            lognormal_params["shape"], lognormal_params["loc"], lognormal_params["scale"],
            size=n_body, random_state=rng,
        )
    if n_tail:
        tail_draws = gpd_params["threshold"] + stats.genpareto.rvs(
            gpd_params["shape"], gpd_params["loc"], gpd_params["scale"],
            size=n_tail, random_state=rng,
        )
        tail_draws = np.minimum(tail_draws, cap_ha if cap_ha is not None else PHYSICAL_CEILING_HA)
        severities[is_tail] = tail_draws
    return severities


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def run_monte_carlo(nb_params: dict, lognormal_params: dict, gpd_params: dict, tail_probability: float,
                     eur_per_ha: float, cap_ha: float, sigma_z: float = 0.0, rho: float = 0.0,
                     n_scenarios: int = N_SCENARIOS, seed: int = SEED) -> tuple:
    """Run the full frequency-severity-dependence Monte Carlo simulation.

    For each scenario-year: draw a fire count and year-level frailty factor
    jointly (simulate_dependent_frequency_and_frailty), draw a severity per
    fire (simulate_severities, capped at cap_ha and PHYSICAL_CEILING_HA),
    multiply every severity that year by the year's frailty factor, sum to an
    aggregate annual burned area and track the running per-year maximum (the
    largest single fire that year), both capped at PHYSICAL_CEILING_HA, then
    convert to euros via eur_per_ha as the final step - per the PRD: "the
    model fits burned area, not euro losses, and converts to euros at the end".

    The running maximum uses a single comparison per fire (if new > current max,
    replace; otherwise discard) - O(1) extra storage per year, no fire sizes
    retained.

    sigma_z and rho default to 0 (no frailty, independent) so this function
    also serves as the "independent model" baseline; pass the calibrated values
    to include dependence.

    Parameters
    ----------
    nb_params : dict
        Fitted Negative Binomial frequency parameters (models/negative_binomial_frequency.json).
    lognormal_params : dict
        Fitted Lognormal severity body (models/lognormal_severity.json).
    gpd_params : dict
        Fitted GPD severity tail (models/pareto_tail.json).
    tail_probability : float
        See simulate_severities.
    eur_per_ha : float
        EUR/ha conversion scenario (low/central/high - see
        models/severity_euro_equivalents.json or
        data/raw/annual_loss_calibration.csv).
    cap_ha : float
        Practical per-fire severity cap (hectares) - see historical_severity_cap.
    sigma_z : float
        Frailty factor spread; 0 = no frailty (default).
    rho : float
        Copula correlation between count and frailty; 0 = independent (default).
    n_scenarios : int
        Number of scenario-years to simulate.
    seed : int
        Random seed, for reproducibility (PRD requirement).

    Returns
    -------
    (np.ndarray, np.ndarray)
        (aggregate_losses, occurrence_losses), each shape (n_scenarios,) in EUR.
        aggregate_losses: total annual loss (sum of all fires that year).
        occurrence_losses: largest single-fire loss that year (0 for years with
        no fires). Pass each to compute_risk_metrics to get the PRD's aggregate
        and occurrence return-period losses respectively.
    """
    rng = np.random.default_rng(seed)
    fire_counts, frailty = simulate_dependent_frequency_and_frailty(nb_params, sigma_z, rho, n_scenarios, rng)
    annual_area = np.zeros(n_scenarios)
    annual_max_fire = np.zeros(n_scenarios)
    for i, n in enumerate(fire_counts):
        if n == 0:
            continue
        severities = simulate_severities(int(n), lognormal_params, gpd_params, tail_probability, rng, cap_ha)
        annual_area[i] = min(severities.sum() * frailty[i], PHYSICAL_CEILING_HA)
        annual_max_fire[i] = min(severities.max() * frailty[i], PHYSICAL_CEILING_HA)
    return annual_area * eur_per_ha, annual_max_fire * eur_per_ha


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def compute_risk_metrics(annual_losses: np.ndarray) -> dict:
    """Compute portfolio-level risk metrics from simulated annual losses.

    Aggregate return-period losses (the PRD's "1-in-10/25/100 ... aggregate
    losses") are the same as VaR at 1-1/N probability, so they're included
    here directly (RP_10 = VaR_90, RP_25 = the 96th percentile, RP_100 =
    VaR_99). Occurrence return-period losses (per-event, not per-year-
    aggregate) are NOT computed here - they need per-event severity draws
    retained, not just the annual sums this function takes; that's a
    separate, not-yet-designed extension to run_monte_carlo's output, tracked
    as follow-up work rather than assumed here.

    Parameters
    ----------
    annual_losses : np.ndarray
        Simulated aggregate annual losses, one per scenario (output of
        run_monte_carlo).

    Returns
    -------
    dict
        {"VaR_90": float, "VaR_95": float, "VaR_99": float, "ES_95": float,
         "RP_10_aggregate": float, "RP_25_aggregate": float,
         "RP_100_aggregate": float, "skewness": float, "kurtosis": float,
         "mean_loss": float, "max_loss": float}
    """
    var90 = float(np.percentile(annual_losses, 90))
    var95 = float(np.percentile(annual_losses, 95))
    var99 = float(np.percentile(annual_losses, 99))
    rp25 = float(np.percentile(annual_losses, 96))
    return {
        "VaR_90": var90,
        "VaR_95": var95,
        "VaR_99": var99,
        "ES_95": float(annual_losses[annual_losses >= var95].mean()),
        "RP_10_aggregate": var90,
        "RP_25_aggregate": rp25,
        "RP_100_aggregate": var99,
        "skewness": float(stats.skew(annual_losses)),
        "kurtosis": float(stats.kurtosis(annual_losses)),
        "mean_loss": float(annual_losses.mean()),
        "max_loss": float(annual_losses.max()),
    }


def convergence_check(annual_losses: np.ndarray, n_splits: int = 10) -> dict:
    """Estimate the standard error of VaR(95%) and VaR(99%) by splitting the
    simulation into independent sub-samples, per the PRD's success criterion
    ("standard error on VaR(95%) is below 2% at the chosen number of runs").

    Splitting a single large run into sub-samples (rather than rerunning the
    whole simulation n_splits times) is a standard, cheap way to estimate this
    SE, valid because scenarios are drawn i.i.d.

    Parameters
    ----------
    annual_losses : np.ndarray
        Simulated aggregate annual losses (output of run_monte_carlo);
        ideally the full N_SCENARIOS run, not a small sample.
    n_splits : int
        Number of equal-sized sub-samples to split into.

    Returns
    -------
    dict
        {"VaR_95_se_pct": float, "VaR_99_se_pct": float, "n_splits": int,
         "n_per_split": int} - se_pct is the standard error as a percentage
         of the full-sample VaR estimate, directly comparable to the PRD's
         "< 2%" criterion.
    """
    n_per_split = len(annual_losses) // n_splits
    splits = annual_losses[:n_per_split * n_splits].reshape(n_splits, n_per_split)
    var95_full = np.percentile(annual_losses, 95)
    var99_full = np.percentile(annual_losses, 99)
    var95_splits = np.percentile(splits, 95, axis=1)
    var99_splits = np.percentile(splits, 99, axis=1)
    return {
        "VaR_95_se_pct": float(100 * var95_splits.std(ddof=1) / var95_full),
        "VaR_99_se_pct": float(100 * var99_splits.std(ddof=1) / var99_full),
        "n_splits": n_splits,
        "n_per_split": n_per_split,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_distribution(annual_losses: np.ndarray, metrics: dict = None, ax=None):
    """Plot a histogram of simulated annual losses, optionally marking VaR/ES lines.

    Parameters
    ----------
    annual_losses : np.ndarray
        Simulated aggregate annual losses.
    metrics : dict, optional
        Output of compute_risk_metrics; if given, VaR(95%) and ES(95%) are
        drawn as vertical reference lines.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; a new figure/axes is created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(annual_losses, bins=100)
    if metrics is not None:
        ax.axvline(metrics["VaR_95"], color="C1", linestyle="--", label="VaR(95%)")
        ax.axvline(metrics["ES_95"], color="C2", linestyle=":", label="ES(95%)")
        ax.legend()
    ax.set_xlabel("Aggregate annual loss (EUR)")
    ax.set_ylabel("Scenario count")
    ax.set_title("Simulated portfolio loss distribution")
    return ax


def plot_qq(annual_losses: np.ndarray, dist=stats.lognorm, dist_params: tuple = None, ax=None):
    """QQ plot of simulated losses against a reference distribution, against the
    TRUE y=x identity line (see distribution_fitting.plot_qq_fit for why: an
    OLS-refit reference line, scipy.stats.probplot's default, can visually
    mislead by pulling toward a few extreme points).

    Parameters
    ----------
    annual_losses : np.ndarray
        Simulated aggregate annual losses.
    dist : scipy.stats rv_continuous
        Reference distribution for comparison.
    dist_params : tuple, optional
        Parameters for `dist`; if omitted, fitted to annual_losses directly
        via dist.fit.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; a new figure/axes is created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots()
    if dist_params is None:
        dist_params = dist.fit(annual_losses)
    n = len(annual_losses)
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical = dist.ppf(plotting_positions, *dist_params)
    empirical = np.sort(annual_losses)
    lim = max(theoretical.max(), empirical.max())
    ax.plot([0, lim], [0, lim], "r-", lw=1, label="y = x (perfect fit)")
    ax.scatter(theoretical, empirical, s=10, alpha=0.5)
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Simulated quantile")
    ax.legend()
    return ax


def plot_tail_comparison(annual_losses: np.ndarray, historical_losses: np.ndarray, ax=None):
    """Compare simulated vs. historical annual loss exceedance curves in the
    upper tail (log-y axis).

    Parameters
    ----------
    annual_losses : np.ndarray
        Simulated aggregate annual losses.
    historical_losses : np.ndarray
        Observed historical annual losses (from data/processed/, grouped to
        annual totals).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; a new figure/axes is created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots()
    for values, label in [(annual_losses, "Simulated"), (historical_losses, "Historical")]:
        sorted_vals = np.sort(values)
        exceedance_prob = 1 - np.arange(1, len(sorted_vals) + 1) / (len(sorted_vals) + 1)
        ax.plot(sorted_vals, exceedance_prob, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Aggregate annual loss (EUR)")
    ax.set_ylabel("Exceedance probability (log scale)")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_simulation_results(annual_losses: np.ndarray, metrics: dict,
                             filename: str = "monte_carlo_results.csv") -> None:
    """Persist simulated losses and summary metrics under simulation/.

    Parameters
    ----------
    annual_losses : np.ndarray
        Simulated aggregate annual losses, one row per scenario.
    metrics : dict
        Output of compute_risk_metrics (and, if run, convergence_check),
        saved alongside as a sidecar JSON.
    filename : str
        Output CSV filename, written under SIMULATION_DIR.
    """
    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"scenario": np.arange(len(annual_losses)), "annual_loss_eur": annual_losses}).to_csv(
        SIMULATION_DIR / filename, index=False
    )
    with open(SIMULATION_DIR / (Path(filename).stem + "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
