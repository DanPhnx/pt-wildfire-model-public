"""Phase 2 distribution-fitting logic: frequency and severity models, goodness-of-fit
tests, bootstrap utilities, and diagnostic plots.

Extracted from notebooks/02_distribution_fitting.ipynb per the PRD's Technical
Decisions: "Entry point: one main.py; notebooks for EDA only." The notebook no
longer defines this logic itself - it imports these functions, calls them on the
data, and displays/saves the results, matching the pattern wildfire_model.py
already set for logic shared across phases.

Import from a notebook (which runs with the notebook's own directory as its
working directory) with:

    import sys
    sys.path.insert(0, "..")
    from distribution_fitting import fit_poisson_frequency, ...
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_processed_data(path: Path = PROCESSED_DIR / "wildfires_processed.csv") -> pd.DataFrame:
    """Load the cleaned wildfire dataset produced in 01_eda.ipynb.

    Parameters
    ----------
    path : Path
        Location of the processed CSV.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Location, Burned_Area_ha, Estimated_Loss_EUR,
        Estimated_Loss_EUR_2025, Loss_Source (see 01_eda.ipynb). One row
        per raw EFFIS fire record (30 ha+, mainland, 2009-2025) - not yet
        grouped into fire-day events; see wildfire_model.build_fire_day_events.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/01_eda.ipynb (or `python main.py --phase 1`) first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def save_model_params(params: dict, filename: str) -> None:
    """Persist fitted distribution parameters to models/ as JSON.

    Parameters
    ----------
    params : dict
        Parameter dictionary (e.g. output of fit_poisson_frequency).
    filename : str
        Output filename, written under MODELS_DIR (e.g. "poisson_frequency.json").
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / filename, "w") as f:
        json.dump(params, f, indent=2)


def bootstrap_parameter_ci(data: np.ndarray, fit_func, param_names: list,
                            n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """Nonparametric bootstrap confidence intervals for fitted distribution parameters.

    Resamples `data` with replacement `n_boot` times, refits via `fit_func` on each
    resample, and reports percentile confidence intervals for the named parameters.
    Nonparametric (resampling the observed data) rather than parametric (simulating
    from the fitted distribution, as bootstrap_ad_pvalue does for a goodness-of-fit
    p-value) - appropriate here since it doesn't assume the fitted distribution is
    exactly correct, only that the resample-and-refit distribution approximates the
    true sampling distribution of the estimator.

    A resample on which `fit_func` raises (e.g. the Negative Binomial's variance
    dropping below its mean on a particular resample, or its MLE failing to
    converge) is skipped rather than crashing the whole bootstrap; `n_boot_successful`
    reports how many of the `n_boot` attempts actually contributed.

    Parameters
    ----------
    data : np.ndarray
        Observed sample the original fit was made on.
    fit_func : callable
        Takes a resampled array, returns a dict of fitted parameters (e.g.
        fit_poisson_frequency, or a small lambda wrapping a scipy .fit call).
    param_names : list of str
        Which keys of fit_func's output to report confidence intervals for.
    n_boot : int
        Number of bootstrap replicates.
    ci : float
        Confidence level (e.g. 0.95 for a 95% CI).
    seed : int
        Random seed, for reproducibility (PRD requirement).

    Returns
    -------
    dict
        {param_name: {"estimate": float, "ci_low": float, "ci_high": float,
         "boot_std": float}}. "estimate" is from fit_func(data) itself (the
        original, non-bootstrapped fit), not the bootstrap mean.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    original = fit_func(data)
    boot_values = {name: [] for name in param_names}
    n_ok = 0
    for _ in range(n_boot):
        resample = rng.choice(data, size=n, replace=True)
        try:
            fitted = fit_func(resample)
        except (ValueError, RuntimeError):
            continue
        n_ok += 1
        for name in param_names:
            boot_values[name].append(fitted[name])

    if n_ok == 0:
        raise RuntimeError(
            f"All {n_boot} bootstrap resamples failed to fit (fit_func raised on every one) - "
            "no confidence interval can be computed. Check fit_func against a resample by hand."
        )

    alpha = (1 - ci) / 2
    result = {}
    for name in param_names:
        vals = np.array(boot_values[name])
        result[name] = {
            "estimate": float(original[name]),
            "ci_low": float(np.percentile(vals, 100 * alpha)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha))),
            "boot_std": float(vals.std()),
        }
    result["n_boot_successful"] = n_ok
    result["n_boot_requested"] = n_boot
    return result


# ---------------------------------------------------------------------------
# Frequency model: Poisson vs Negative Binomial
# ---------------------------------------------------------------------------

def overdispersion_test(counts: np.ndarray) -> dict:
    """Test H0: counts are Poisson-distributed (variance = mean).

    Uses the index-of-dispersion statistic T = sum((x - mean)^2) / mean,
    which is approximately chi-squared distributed with n-1 degrees of
    freedom under H0 (a standard, simple test for overdispersion in count
    data). A large T / small p-value means the data are more variable than
    a Poisson process allows.

    Parameters
    ----------
    counts : np.ndarray
        One value per period (e.g. per year): number of events.

    Returns
    -------
    dict
        {"statistic": float, "df": int, "p_value": float,
         "dispersion_ratio": float} - dispersion_ratio is the sample
        variance-to-mean ratio (1.0 under exact Poisson dispersion).
    """
    counts = np.asarray(counts, dtype=float)
    n, mean = len(counts), counts.mean()
    statistic = float(np.sum((counts - mean) ** 2) / mean)
    df = n - 1
    p_value = float(1 - stats.chi2.cdf(statistic, df=df))
    dispersion_ratio = float(counts.var(ddof=1) / mean)
    return {"statistic": statistic, "df": df, "p_value": p_value, "dispersion_ratio": dispersion_ratio}


def fit_poisson_frequency(annual_counts: np.ndarray) -> dict:
    """Fit a Poisson distribution to annual fire-day counts via MLE.

    Parameters
    ----------
    annual_counts : np.ndarray
        One value per year: number of fire-day events that year.

    Returns
    -------
    dict
        {"lambda": float, "loglik": float, "aic": float, "bic": float} -
        the MLE rate parameter is the sample mean; aic/bic use 1 fitted
        parameter (bic additionally penalises by ln(n), n = len(annual_counts)).
    """
    annual_counts = np.asarray(annual_counts, dtype=float)
    n = len(annual_counts)
    lam = float(annual_counts.mean())
    loglik = float(stats.poisson.logpmf(annual_counts, lam).sum())
    return {"lambda": lam, "loglik": loglik, "aic": 2 * 1 - 2 * loglik, "bic": 1 * np.log(n) - 2 * loglik}


def fit_negative_binomial_frequency(annual_counts: np.ndarray) -> dict:
    """Fit a Negative Binomial distribution to annual fire-day counts via MLE.

    Parameterised as scipy.stats.nbinom(r, p) (r = number of "successes",
    p = success probability; mean = r(1-p)/p, var = r(1-p)/p^2). The
    method-of-moments solution (matching the sample mean and variance
    exactly) is used as the optimiser's starting point, since it is a
    good, stable estimate for a two-parameter fit on a short (n=12) annual
    series, then refined by MLE.

    Parameters
    ----------
    annual_counts : np.ndarray
        One value per year: number of fire-day events that year.

    Returns
    -------
    dict
        {"r": float, "p": float, "mean": float, "loglik": float,
         "aic": float, "bic": float} - aic/bic use 2 fitted parameters
         (bic additionally penalises by ln(n), n = len(annual_counts)).
    """
    annual_counts = np.asarray(annual_counts, dtype=float)
    n = len(annual_counts)
    mean, var = annual_counts.mean(), annual_counts.var(ddof=1)
    if var <= mean:
        raise ValueError("Sample variance <= mean; data are not overdispersed, use Poisson instead.")
    r0, p0 = mean ** 2 / (var - mean), mean / var

    def negloglik(params):
        r, p = params
        if r <= 0 or not (0 < p < 1):
            return np.inf
        return -stats.nbinom.logpmf(annual_counts, r, p).sum()

    result = minimize(negloglik, x0=[r0, p0], method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(
            f"Negative Binomial MLE did not converge (method-of-moments start r0={r0:.2f}, p0={p0:.3f}): "
            f"{result.message}"
        )
    r, p = result.x
    loglik = float(-result.fun)
    return {
        "r": float(r), "p": float(p), "mean": float(r * (1 - p) / p), "loglik": loglik,
        "aic": 2 * 2 - 2 * loglik, "bic": 2 * np.log(n) - 2 * loglik,
    }


# ---------------------------------------------------------------------------
# Severity model: Lognormal body + Generalised Pareto tail
# ---------------------------------------------------------------------------

def fit_lognormal_severity(losses: np.ndarray) -> dict:
    """Fit a Lognormal distribution to per-event loss magnitudes via MLE.

    Fitted with loc fixed at 0 (a Lognormal is naturally supported on
    (0, inf); losses are always positive), so only the shape (sigma of the
    underlying Normal) and scale (exp(mu)) are estimated.

    Parameters
    ----------
    losses : np.ndarray
        Per-event burned area (hectares - see Burned_Area_ha; the PRD
        fits severity on burned area, not euro losses, and converts to
        euros afterward - see 02_distribution_fitting.ipynb's "Run:
        severity model" section), training years only. Named `losses` for
        historical reasons; the function itself is unit-agnostic.

    Returns
    -------
    dict
        {"shape": float, "loc": float, "scale": float, "mu": float,
         "loglik": float, "aic": float, "bic": float} - shape/loc/scale are
        scipy.stats.lognorm's parameterisation (loc=0); mu = log(scale) is
        the underlying Normal's mean, reported since it is the more
        commonly quoted Lognormal parameter. aic/bic use 2 fitted
        parameters (bic additionally penalises by ln(n), n = len(losses)).
    """
    shape, loc, scale = stats.lognorm.fit(losses, floc=0)
    loglik = float(stats.lognorm.logpdf(losses, shape, loc, scale).sum())
    return {
        "shape": float(shape), "loc": float(loc), "scale": float(scale),
        "mu": float(np.log(scale)), "loglik": loglik,
        "aic": 2 * 2 - 2 * loglik, "bic": 2 * np.log(len(losses)) - 2 * loglik,
    }


def fit_pareto_tail(losses: np.ndarray, threshold: float) -> dict:
    """Fit a Generalized Pareto distribution to losses exceeding a threshold.

    Fitted on exceedances (losses - threshold) with loc fixed at 0, the
    standard peaks-over-threshold parameterisation.

    Parameters
    ----------
    losses : np.ndarray
        Per-event burned area (hectares - see Burned_Area_ha; the PRD
        fits severity on burned area, not euro losses - see
        02_distribution_fitting.ipynb's "Run: severity model" section),
        training years only. Named `losses` for historical reasons; the
        function itself is unit-agnostic.
    threshold : float
        Loss level above which the tail fit is applied (peaks-over-
        threshold); choose with a mean-excess plot, not an arbitrary
        percentile.

    Returns
    -------
    dict
        {"shape": float, "loc": float, "scale": float, "threshold": float,
         "n_exceedances": int, "loglik": float, "aic": float, "bic": float}
         - shape (xi) is the tail index: xi > 0 (found here, ~0.75) means a
         heavy tail with infinite theoretical variance, consistent with
         catastrophe losses dominated by a few extreme years (2017, 2025).
         aic/bic use 2 fitted parameters (bic additionally penalises by
         ln(n_exceedances), the sample this was actually fitted on).
    """
    exceedances = losses[losses > threshold] - threshold
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    loglik = float(stats.genpareto.logpdf(exceedances, shape, loc, scale).sum())
    return {
        "shape": float(shape), "loc": float(loc), "scale": float(scale),
        "threshold": float(threshold), "n_exceedances": int(len(exceedances)),
        "loglik": loglik, "aic": 2 * 2 - 2 * loglik, "bic": 2 * np.log(len(exceedances)) - 2 * loglik,
    }


def fit_gpd_raw(x: np.ndarray) -> dict:
    """Fit a Generalised Pareto directly to an already-shifted array (e.g. exceedances).

    A thin wrapper around scipy.stats.genpareto.fit(x, floc=0) returning a dict,
    for use as the `fit_func` passed to bootstrap_parameter_ci when the data is
    already exceedances (not raw losses needing a threshold applied) -
    fit_pareto_tail takes a threshold and re-derives exceedances each call,
    which would be wrong to do again on data that's already exceedances (e.g. a
    bootstrap resample of them).

    Not for bootstrap_ad_pvalue: that function's `fit_func` must return a tuple
    compatible with `dist.cdf(x, *params)` (e.g.
    `lambda x: stats.genpareto.fit(x, floc=0)`, unwrapped), not this function's
    dict - the two bootstrap helpers deliberately take different fit_func shapes.

    Parameters
    ----------
    x : np.ndarray
        Exceedances (already shifted so the threshold is at 0).

    Returns
    -------
    dict
        {"shape": float, "loc": float, "scale": float}
    """
    shape, loc, scale = stats.genpareto.fit(x, floc=0)
    return {"shape": shape, "loc": loc, "scale": scale}


# ---------------------------------------------------------------------------
# Goodness-of-fit tests
# ---------------------------------------------------------------------------

def run_anderson_darling_test(data: np.ndarray, dist: str = "norm") -> dict:
    """Run an Anderson-Darling goodness-of-fit test.

    Only for distribution families scipy.stats.anderson supports directly
    (norm, expon, logistic, gumbel, weibull_min, ...), since those are the
    ones with known critical values already adjusted for parameters
    estimated from the same sample. For a fit scipy.stats.anderson does not
    support (e.g. the Generalised Pareto tail here), use
    bootstrap_ad_pvalue instead - do not use this function's critical
    values for an unsupported family, they would not apply.

    Parameters
    ----------
    data : np.ndarray
        Observed sample. For the Lognormal body fitted here, pass
        log(losses) with dist="norm" (since Lognormal(losses) is exactly
        Normal(log(losses)), and scipy.stats.anderson does not support
        lognorm directly).
    dist : str
        Distribution family supported by scipy.stats.anderson.

    Returns
    -------
    dict
        {"statistic": float, "critical_values": list, "significance_levels": list,
         "reject_at_5pct": bool}
    """
    # scipy >= 1.17 warns unless a p-value `method` is chosen, but that switches the
    # return shape to a plain p-value and drops critical_values/significance_level -
    # this function is pinned (requirements.txt: scipy==1.18.1) to the classic form.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = stats.anderson(data, dist=dist)
    idx_5pct = list(result.significance_level).index(5.0)
    return {
        "statistic": float(result.statistic),
        "critical_values": [float(v) for v in result.critical_values],
        "significance_levels": [float(v) for v in result.significance_level],
        "reject_at_5pct": bool(result.statistic > result.critical_values[idx_5pct]),
    }


def _anderson_darling_statistic(sorted_data: np.ndarray, cdf: np.ndarray) -> float:
    """The Anderson-Darling statistic A^2 for sorted data against a fitted CDF.

    A^2 = -n - (1/n) * sum_i (2i - 1) * [ln F(x_i) + ln(1 - F(x_(n+1-i)))],
    the general formula for any fully-specified (or fitted-then-plugged-in)
    continuous distribution. Used here because scipy.stats.anderson does
    not support the Generalised Pareto family.

    Parameters
    ----------
    sorted_data : np.ndarray
        Data sorted ascending (not checked - caller's responsibility).
    cdf : np.ndarray
        The fitted distribution's CDF evaluated at sorted_data, same order.

    Returns
    -------
    float
        The A^2 statistic (larger = worse fit).
    """
    n = len(sorted_data)
    i = np.arange(1, n + 1)
    cdf = np.clip(cdf, 1e-12, 1 - 1e-12)  # avoid log(0) at the extremes
    return float(-n - np.mean((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1]))))


def bootstrap_ad_pvalue(data: np.ndarray, dist, fit_func, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap an Anderson-Darling p-value for a distribution whose parameters
    were fitted on this same data (e.g. the Generalised Pareto tail, which
    scipy.stats.anderson does not support directly).

    Repeatedly simulates a same-size sample from the fitted distribution,
    refits it, and computes the AD statistic each time, to build the null
    distribution of the AD statistic under "parameters estimated from the
    data" rather than "parameters known in advance" (the assumption behind
    scipy's built-in critical values, which is why they cannot be reused
    for a family scipy does not fit natively).

    Parameters
    ----------
    data : np.ndarray
        Observed sample the distribution was fitted to (e.g. GPD exceedances).
    dist : scipy.stats rv_continuous
        Distribution object (e.g. scipy.stats.genpareto).
    fit_func : callable
        Function that takes an array and returns fitted params as a tuple
        compatible with dist.cdf(x, *params) (e.g. lambda x: dist.fit(x, floc=0)).
        May raise ValueError/RuntimeError on a degenerate simulated sample (as
        fit_negative_binomial_frequency does); such a replicate is skipped
        rather than crashing the whole bootstrap, matching bootstrap_parameter_ci.
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Random seed, for reproducibility (PRD requirement).

    Returns
    -------
    dict
        {"statistic": float, "p_value": float, "n_boot": int, "n_boot_successful": int}
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    params = fit_func(data)
    sorted_data = np.sort(data)
    observed = _anderson_darling_statistic(sorted_data, dist.cdf(sorted_data, *params))

    boot_stats = []
    for _ in range(n_boot):
        sim = dist.rvs(*params, size=n, random_state=rng)
        try:
            sim_params = fit_func(sim)
        except (ValueError, RuntimeError):
            continue
        sim_sorted = np.sort(sim)
        boot_stats.append(_anderson_darling_statistic(sim_sorted, dist.cdf(sim_sorted, *sim_params)))

    if not boot_stats:
        raise RuntimeError(
            f"All {n_boot} bootstrap replicates failed to fit - no p-value can be computed."
        )
    boot_stats = np.array(boot_stats)
    p_value = float(np.mean(boot_stats >= observed))
    return {"statistic": observed, "p_value": p_value, "n_boot": n_boot, "n_boot_successful": len(boot_stats)}


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_empirical_vs_fitted_cdf(data: np.ndarray, dist, params: tuple, ax=None, label: str = "Fitted"):
    """Overlay the empirical CDF against a fitted distribution's CDF.

    Parameters
    ----------
    data : np.ndarray
        Observed sample.
    dist : scipy.stats rv_continuous
        Distribution object (e.g. scipy.stats.lognorm).
    params : tuple
        Fitted parameters to pass to dist.cdf.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; a new figure/axes is created if omitted.
    label : str
        Legend label for the fitted curve.
    """
    if ax is None:
        _, ax = plt.subplots()
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, empirical_cdf, label="Empirical")
    ax.plot(sorted_data, dist.cdf(sorted_data, *params), label=label, linestyle="--")
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative probability")
    ax.legend()
    return ax


def plot_mean_excess(losses: np.ndarray, ax=None, xlabel: str = "Threshold", ylabel: str = "Mean excess over threshold"):
    """Plot a mean-excess plot, used to choose the Generalised Pareto tail threshold.

    For a range of candidate thresholds u, plots the mean of (loss - u)
    over losses exceeding u. A roughly linear region indicates a range of
    u for which a GPD tail is a reasonable model; the threshold is chosen
    from where that linearity starts.

    Parameters
    ----------
    losses : np.ndarray
        Per-event severity values (unit-agnostic - hectares, euros, whatever
        the caller is fitting on; label the axes accordingly).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; a new figure/axes is created if omitted.
    xlabel, ylabel : str
        Axis labels. No unit is hardcoded here - an earlier version always
        said "(EUR)" even when called on hectares, which was wrong once
        severity was refit on burned area; pass the caller's actual unit.
    """
    if ax is None:
        _, ax = plt.subplots()
    thresholds = np.sort(losses)[:-5]  # leave enough points above the highest threshold to average
    mean_excess = [losses[losses > u].mean() - u for u in thresholds]
    ax.plot(thresholds, mean_excess)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def _qq_theoretical_and_empirical(data: np.ndarray, dist, params: tuple) -> tuple:
    """Shared theoretical/empirical quantile computation for plot_qq_fit and
    check_qq_systematic_deviation, so the two can never silently compute this
    differently (e.g. a different plotting-position convention) and disagree.

    Parameters
    ----------
    data : np.ndarray
        Observed sample the distribution was fitted to.
    dist : scipy.stats rv_continuous
        Distribution object.
    params : tuple
        Fitted parameters, passed to dist.ppf.

    Returns
    -------
    (np.ndarray, np.ndarray)
        (theoretical, empirical) quantiles, both sorted ascending, same length as data.
    """
    n = len(data)
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical = dist.ppf(plotting_positions, *params)
    empirical = np.sort(data)
    return theoretical, empirical


def plot_qq_fit(data: np.ndarray, dist, params: tuple, ax=None, title: str = None):
    """QQ plot: sorted empirical quantiles against a fitted distribution's theoretical
    quantiles, compared to the TRUE y=x identity line - not a separately OLS-refit
    line (that is what scipy.stats.probplot draws by default, and it can visually
    mislead: a few extreme points can pull that regression line toward the data,
    making a genuinely poor fit look deceptively close to "its own" line). Since
    `params` are already the MLE fit, a perfect fit sits exactly on y=x.

    Per the PRD: R^2 is deliberately not used to judge fit quality here (the prior
    PRD's R^2 criterion was dropped for being too easy to pass). Read this plot
    visually, and see check_qq_systematic_deviation for a non-R^2, rank-based
    numeric check of whether points drift away from the line systematically as you
    move along it (rather than scattering randomly around it).

    Parameters
    ----------
    data : np.ndarray
        Observed sample the distribution was fitted to.
    dist : scipy.stats rv_continuous
        Distribution object (e.g. scipy.stats.lognorm, scipy.stats.genpareto).
    params : tuple
        Fitted parameters, passed to dist.ppf.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; a new figure/axes is created if omitted.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    theoretical, empirical = _qq_theoretical_and_empirical(data, dist, params)
    lim = max(theoretical.max(), empirical.max())
    ax.plot([0, lim], [0, lim], "r-", lw=1, label="y = x (perfect fit)")
    ax.scatter(theoretical, empirical, s=14, alpha=0.6)
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Empirical quantile")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def check_qq_systematic_deviation(data: np.ndarray, dist, params: tuple) -> dict:
    """Test whether QQ-plot residuals (empirical - theoretical) drift systematically
    with rank, rather than scattering randomly around zero.

    A Spearman rank correlation between position (1..n) and residual: a strong,
    significant correlation means the fit gets steadily better or worse as you move
    through the sample (a systematic deviation - e.g. the fit consistently
    under-predicts the largest values); a correlation near zero with a large p-value
    means residuals are consistent with random scatter (no systematic deviation).
    Deliberately not an R^2-based check - the PRD explicitly excludes R^2 as a fit
    criterion.

    Parameters
    ----------
    data : np.ndarray
        Observed sample the distribution was fitted to.
    dist : scipy.stats rv_continuous
        Distribution object.
    params : tuple
        Fitted parameters, passed to dist.ppf.

    Returns
    -------
    dict
        {"rho": float, "p_value": float, "systematic": bool} - systematic is True
        when the rank-residual correlation is significant at 5%.
    """
    theoretical, empirical = _qq_theoretical_and_empirical(data, dist, params)
    residual = empirical - theoretical
    rho, p_value = stats.spearmanr(np.arange(1, len(data) + 1), residual)
    return {"rho": float(rho), "p_value": float(p_value), "systematic": bool(p_value < 0.05)}
