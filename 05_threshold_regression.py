"""
Phase 2c — Threshold Regression (Hansen 2000)
===============================================
Tests for non-linearity in the price response to demand contraction.

Model:
  DeltaP_t = alpha + beta_1 * DeltaQ_t * I(DeltaQ_t <= gamma)
                    + beta_2 * DeltaQ_t * I(DeltaQ_t > gamma) + eps_t

The threshold gamma identifies the tipping point: the demand contraction
magnitude beyond which the price response becomes discontinuously larger.

Hansen (2000) procedure:
  1. Grid search over candidate thresholds
  2. Compute sup-F statistic
  3. Bootstrap p-value

Output:
  - data/clean/threshold_results.csv
  - figures/threshold_regression.png

Usage:
  python 05_threshold_regression.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "clean"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load SVAR data and compute demand-price changes."""
    path = DATA_DIR / "kilian_svar_data.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)

    # Compute monthly changes
    df["delta_price"] = df["real_oil_price"].pct_change() * 100
    df["delta_production"] = df["prod_growth"]

    # Demand proxy: negative of production growth (contraction = positive)
    # In practice, use consumption data if available
    if "rea_index" in df.columns:
        df["delta_demand"] = -df["rea_index"].diff()
    else:
        df["delta_demand"] = -df["prod_growth"]

    df = df.dropna()
    return df


def hansen_threshold_test(y, x, q, n_grid: int = 300, n_boot: int = 1000):
    """
    Hansen (2000) threshold regression test.

    Parameters:
      y: dependent variable (price change)
      x: regressor (demand change)
      q: threshold variable (same as x here)
      n_grid: number of grid points for threshold search
      n_boot: bootstrap replications for p-value

    Returns:
      results dict with threshold estimate, coefficients, F-stat, p-value
    """
    n = len(y)

    # Trim: search over [15th, 85th] percentile of q
    q_lo, q_hi = np.percentile(q, [15, 85])
    gammas = np.linspace(q_lo, q_hi, n_grid)

    # Step 1: Linear model (no threshold) — null hypothesis
    X_linear = sm.add_constant(x)
    res_linear = sm.OLS(y, X_linear).fit()
    ssr_linear = np.sum(res_linear.resid ** 2)

    # Step 2: Grid search for threshold
    ssr_grid = np.full(n_grid, np.inf)
    for i, gamma in enumerate(gammas):
        d_lo = (q <= gamma).astype(float)
        d_hi = (q > gamma).astype(float)

        X_thresh = np.column_stack([
            np.ones(n),
            x * d_lo,
            x * d_hi,
        ])

        # Check sufficient observations in each regime
        n_lo = d_lo.sum()
        n_hi = d_hi.sum()
        if n_lo < 0.1 * n or n_hi < 0.1 * n:
            continue

        try:
            res = sm.OLS(y, X_thresh).fit()
            ssr_grid[i] = np.sum(res.resid ** 2)
        except Exception:
            continue

    # Optimal threshold
    best_idx = np.argmin(ssr_grid)
    gamma_hat = gammas[best_idx]
    ssr_thresh = ssr_grid[best_idx]

    # F-statistic: F = n * (SSR_linear - SSR_thresh) / SSR_thresh
    F_stat = n * (ssr_linear - ssr_thresh) / ssr_thresh

    # Estimate threshold model at optimal gamma
    d_lo = (q <= gamma_hat).astype(float)
    d_hi = (q > gamma_hat).astype(float)
    X_best = np.column_stack([np.ones(n), x * d_lo, x * d_hi])
    res_best = sm.OLS(y, X_best).fit()

    # Step 3: Bootstrap p-value
    print(f"    Bootstrap ({n_boot} reps) for p-value...")
    boot_F = np.zeros(n_boot)
    resid_centered = res_linear.resid - res_linear.resid.mean()
    y_fitted = res_linear.fittedvalues

    for b in range(n_boot):
        # Wild bootstrap (Rademacher)
        w = np.random.choice([-1, 1], size=n)
        y_boot = y_fitted + resid_centered * w

        # Null SSR
        res_null_b = sm.OLS(y_boot, X_linear).fit()
        ssr_null_b = np.sum(res_null_b.resid ** 2)

        # Search
        ssr_min_b = ssr_null_b
        for gamma_b in gammas[::5]:  # Coarser grid for speed
            d_lo_b = (q <= gamma_b).astype(float)
            d_hi_b = (q > gamma_b).astype(float)
            n_lo_b = d_lo_b.sum()
            n_hi_b = d_hi_b.sum()
            if n_lo_b < 0.1 * n or n_hi_b < 0.1 * n:
                continue
            X_b = np.column_stack([np.ones(n), x * d_lo_b, x * d_hi_b])
            try:
                res_b = sm.OLS(y_boot, X_b).fit()
                ssr_b = np.sum(res_b.resid ** 2)
                if ssr_b < ssr_min_b:
                    ssr_min_b = ssr_b
            except Exception:
                continue

        boot_F[b] = n * (ssr_null_b - ssr_min_b) / ssr_min_b

    p_value = np.mean(boot_F >= F_stat)

    return {
        "gamma_hat": gamma_hat,
        "F_stat": F_stat,
        "p_value": p_value,
        "beta_below": res_best.params[1],
        "beta_above": res_best.params[2],
        "se_below": res_best.bse[1],
        "se_above": res_best.bse[2],
        "n_below": int(d_lo.sum()),
        "n_above": int(d_hi.sum()),
        "ssr_linear": ssr_linear,
        "ssr_threshold": ssr_thresh,
        "r2_linear": res_linear.rsquared,
        "r2_threshold": res_best.rsquared,
        "gammas": gammas,
        "ssr_grid": ssr_grid,
        "boot_F": boot_F,
    }


def plot_threshold_results(df, results):
    """Plot threshold regression diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    gamma = results["gamma_hat"]

    # 1. SSR as function of threshold
    ax = axes[0]
    valid = results["ssr_grid"] < np.inf
    ax.plot(results["gammas"][valid], results["ssr_grid"][valid], 'b-', linewidth=1)
    ax.axvline(gamma, color='red', linestyle='--', linewidth=1.5,
              label=f'$\\hat{{\\gamma}}$ = {gamma:.3f}')
    ax.set_xlabel('Threshold ($\\gamma$)', fontsize=10)
    ax.set_ylabel('Sum of Squared Residuals', fontsize=10)
    ax.set_title('Threshold Search\n(Hansen 2000)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. Scatter with two regimes
    ax = axes[1]
    x = df["delta_demand"].values
    y = df["delta_price"].values
    mask_lo = x <= gamma
    mask_hi = x > gamma

    ax.scatter(x[mask_lo], y[mask_lo], s=8, alpha=0.4, color='steelblue', label='Below threshold')
    ax.scatter(x[mask_hi], y[mask_hi], s=8, alpha=0.4, color='tomato', label='Above threshold')

    # Regression lines
    x_range_lo = np.linspace(x[mask_lo].min(), gamma, 100)
    x_range_hi = np.linspace(gamma, x[mask_hi].max(), 100)
    ax.plot(x_range_lo, results["beta_below"] * x_range_lo,
           'b-', linewidth=2, label=f'$\\beta_1$ = {results["beta_below"]:.3f}')
    ax.plot(x_range_hi, results["beta_above"] * x_range_hi,
           'r-', linewidth=2, label=f'$\\beta_2$ = {results["beta_above"]:.3f}')
    ax.axvline(gamma, color='black', linestyle=':', linewidth=1)

    ax.set_xlabel('Demand Contraction', fontsize=10)
    ax.set_ylabel('Price Change (%)', fontsize=10)
    ax.set_title('Two-Regime Price Response', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Bootstrap distribution of F-statistic
    ax = axes[2]
    ax.hist(results["boot_F"], bins=50, color='lightgray', edgecolor='gray', density=True)
    ax.axvline(results["F_stat"], color='red', linewidth=2,
              label=f'F = {results["F_stat"]:.2f} (p = {results["p_value"]:.3f})')
    ax.set_xlabel('F-statistic', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Bootstrap Distribution\nunder $H_0$: No Threshold', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "threshold_regression.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/threshold_regression.png")


def main():
    print("="*60)
    print("Phase 2c: Threshold Regression (Hansen 2000)")
    print("="*60)

    df = load_data()
    print(f"  Sample: {len(df)} observations")

    y = df["delta_price"].values
    x = df["delta_demand"].values
    q = x  # Threshold variable same as regressor

    results = hansen_threshold_test(y, x, q, n_grid=300, n_boot=1000)

    print(f"\n  Results:")
    print(f"    Estimated threshold (gamma*): {results['gamma_hat']:.4f}")
    print(f"    F-statistic:                  {results['F_stat']:.4f}")
    print(f"    Bootstrap p-value:            {results['p_value']:.4f}")
    print(f"    beta (below threshold):       {results['beta_below']:.4f} (SE={results['se_below']:.4f})")
    print(f"    beta (above threshold):       {results['beta_above']:.4f} (SE={results['se_above']:.4f})")
    print(f"    N below / above:              {results['n_below']} / {results['n_above']}")
    print(f"    R2 linear:                    {results['r2_linear']:.4f}")
    print(f"    R2 threshold:                 {results['r2_threshold']:.4f}")

    # Significance
    if results['p_value'] < 0.05:
        print(f"\n    *** Threshold is statistically significant at 5% level ***")
        print(f"    The price response to demand contraction is non-linear.")
        print(f"    This supports the tipping-point mechanism in the MFG model.")
    else:
        print(f"\n    Threshold is NOT significant at 5%.")
        print(f"    Linear price response cannot be rejected.")

    plot_threshold_results(df, results)

    # Save for MFG calibration
    save_results = {k: v for k, v in results.items()
                    if k not in ["gammas", "ssr_grid", "boot_F"]}
    pd.Series(save_results).to_csv(DATA_DIR / "threshold_results.csv")
    print(f"\n  Saved threshold_results.csv")

    print("\n" + "="*60)
    print("Threshold regression complete.")
    print("Run 06_mfg_simulation.py next.")


if __name__ == "__main__":
    main()
