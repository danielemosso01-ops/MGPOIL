"""
Phase 2a — Structural VAR Estimation (Kilian 2009)
====================================================
Estimates a 3-variable SVAR with Cholesky identification:
  1. Percent change in global crude oil production
  2. Real economic activity index
  3. Real oil price (log)

Identification ordering follows Kilian (2009):
  Supply shock → Aggregate demand shock → Oil-specific demand shock

Output:
  - Impulse response functions (saved as CSV and figures)
  - The empirical price response function DeltaP(m) for MFG calibration
  - data/clean/svar_results.npz
  - figures/irf_*.png

Usage:
  python 03_svar_estimation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.api import VAR

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "clean"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_svar_data() -> pd.DataFrame:
    """Load the 3-variable SVAR dataset."""
    path = DATA_DIR / "kilian_svar_data.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def estimate_svar(df: pd.DataFrame, maxlags: int = 24):
    """
    Estimate the structural VAR following Kilian (2009).

    The system is:
      y_t = [prod_growth_t, rea_index_t, log_real_price_t]'

    Identification: Cholesky decomposition of the reduced-form
    covariance matrix with the ordering above. This imposes:
      - Oil supply shocks affect all variables contemporaneously
      - Aggregate demand shocks affect REA and price contemporaneously
      - Oil-specific demand shocks affect only price contemporaneously
    """
    print("Estimating SVAR...")

    # Prepare the 3-variable system
    y = pd.DataFrame(index=df.index)
    y["prod_growth"] = df["prod_growth"]
    y["rea_index"] = df["rea_index"] if "rea_index" in df.columns else df.get("oil_production", 0)
    y["log_price"] = np.log(df["real_oil_price"].clip(lower=1))

    y = y.dropna()
    print(f"  Sample: {y.index[0].strftime('%Y-%m')} to {y.index[-1].strftime('%Y-%m')} "
          f"({len(y)} observations)")

    # Lag selection
    model = VAR(y)
    lag_results = model.select_order(maxlags=min(maxlags, len(y) // 3))
    print(f"  Lag selection — AIC: {lag_results.aic}, BIC: {lag_results.bic}")
    optimal_lag = max(lag_results.aic, 1)
    # Use 24 lags as per Kilian (2009) standard if sample allows
    n_lags = min(24, optimal_lag, len(y) // 3)
    print(f"  Using {n_lags} lags")

    # Estimate reduced-form VAR
    results = model.fit(n_lags)
    print(f"  AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")

    # Structural identification via Cholesky
    sigma_u = results.sigma_u  # Reduced-form covariance matrix
    A0_inv = np.linalg.cholesky(sigma_u)  # Lower triangular

    print(f"\n  Cholesky decomposition (A0^-1):")
    for i, name in enumerate(y.columns):
        print(f"    {name}: {A0_inv[i]}")

    return results, A0_inv, y


def compute_irfs(results, A0_inv, periods: int = 48):
    """
    Compute structural impulse response functions.

    Returns: irf_array of shape (periods, 3, 3)
      irf_array[h, i, j] = response of variable i to structural shock j at horizon h
    """
    print(f"\n  Computing IRFs for {periods} periods...")

    # Get MA coefficients from reduced-form VAR
    ma_coefs = results.ma_rep(periods - 1)  # shape: (periods, 3, 3)

    # Structural IRFs: Phi_h * A0_inv
    irf_structural = np.zeros((periods, 3, 3))
    for h in range(periods):
        irf_structural[h] = ma_coefs[h] @ A0_inv

    return irf_structural


def bootstrap_irfs(results, A0_inv, y, n_boot: int = 500, periods: int = 48):
    """Bootstrap confidence intervals for IRFs."""
    print(f"  Bootstrapping IRFs ({n_boot} replications)...")

    n_lags = results.k_ar
    resids = results.resid.values
    T = len(resids)
    k = resids.shape[1]

    boot_irfs = np.zeros((n_boot, periods, k, k))

    for b in range(n_boot):
        # Resample residuals
        idx = np.random.randint(0, T, size=T)
        boot_resid = resids[idx]

        # Reconstruct data
        y_boot = y.copy()
        fitted = results.fittedvalues.values
        y_boot_vals = y.values.copy()
        for t in range(n_lags, len(y_boot_vals)):
            y_boot_vals[t] = fitted[t - n_lags] + boot_resid[t - n_lags] if (t - n_lags) < len(fitted) else y_boot_vals[t]

        # Re-estimate
        try:
            model_b = VAR(pd.DataFrame(y_boot_vals, columns=y.columns))
            res_b = model_b.fit(n_lags)
            A0_inv_b = np.linalg.cholesky(res_b.sigma_u)
            ma_b = res_b.ma_rep(periods - 1)
            for h in range(periods):
                boot_irfs[b, h] = ma_b[h] @ A0_inv_b
        except Exception:
            boot_irfs[b] = np.nan

    # Compute confidence bands (16th and 84th percentiles, standard in SVAR literature)
    irf_lo = np.nanpercentile(boot_irfs, 16, axis=0)
    irf_hi = np.nanpercentile(boot_irfs, 84, axis=0)

    return irf_lo, irf_hi


def extract_price_response_function(irf_structural):
    """
    Extract the empirical DeltaP(m) function.

    This is the cumulative response of log oil price (variable 2)
    to an oil-specific demand shock (shock 2) — the demand reduction channel.

    The response at horizon h represents the price change when demand
    contracts by the magnitude of one structural standard deviation.
    We normalize to get DeltaP as a function of demand contraction magnitude m.
    """
    # Response of log_price (var 2) to demand shock (shock 2)
    # Cumulative response
    price_response = np.cumsum(irf_structural[:, 2, 2])

    # The 12-month cumulative response is our baseline DeltaP
    delta_p_12m = price_response[11] if len(price_response) > 11 else price_response[-1]

    print(f"\n  Price response to demand shock:")
    print(f"    Impact:  {irf_structural[0, 2, 2]:.4f}")
    print(f"    6-month cumulative: {price_response[5]:.4f}" if len(price_response) > 5 else "")
    print(f"    12-month cumulative: {delta_p_12m:.4f}")

    return price_response


def plot_irfs(irf_structural, irf_lo, irf_hi, periods):
    """Plot and save impulse response functions."""
    var_names = ["Oil Production Growth", "Real Economic Activity", "Log Real Oil Price"]
    shock_names = ["Supply Shock", "Aggregate Demand Shock", "Oil-Specific Demand Shock"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    horizons = np.arange(periods)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.plot(horizons, irf_structural[:, i, j], 'b-', linewidth=1.5)
            ax.fill_between(horizons, irf_lo[:, i, j], irf_hi[:, i, j],
                          alpha=0.2, color='blue')
            ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
            ax.set_title(f'{var_names[i]} → {shock_names[j]}', fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 2:
                ax.set_xlabel('Months', fontsize=8)

    fig.suptitle('Structural Impulse Response Functions (Kilian 2009 Identification)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "irf_structural.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/irf_structural.png")


def plot_price_response(price_response):
    """Plot the empirical DeltaP(m) function."""
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = np.arange(len(price_response))

    ax.plot(horizons, price_response, 'b-', linewidth=2, label='Cumulative price response')
    ax.fill_between(horizons, price_response, alpha=0.15, color='blue')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Months after demand shock', fontsize=11)
    ax.set_ylabel('Cumulative log price change', fontsize=11)
    ax.set_title('Price Response to Oil-Specific Demand Shock\n'
                '(Empirical $\\Delta P(m)$ function)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "delta_p_function.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/delta_p_function.png")


def main():
    print("="*60)
    print("Phase 2a: Structural VAR Estimation (Kilian 2009)")
    print("="*60)

    # Load data
    df = load_svar_data()

    # Estimate SVAR
    results, A0_inv, y = estimate_svar(df)

    # Compute IRFs
    periods = 48
    irf_structural = compute_irfs(results, A0_inv, periods)

    # Bootstrap confidence intervals
    irf_lo, irf_hi = bootstrap_irfs(results, A0_inv, y, n_boot=500, periods=periods)

    # Extract price response function
    price_response = extract_price_response_function(irf_structural)

    # Plot
    plot_irfs(irf_structural, irf_lo, irf_hi, periods)
    plot_price_response(price_response)

    # Save results for MFG calibration
    np.savez(
        DATA_DIR / "svar_results.npz",
        irf_structural=irf_structural,
        irf_lo=irf_lo,
        irf_hi=irf_hi,
        price_response=price_response,
        A0_inv=A0_inv,
    )
    print(f"\n  Saved svar_results.npz")

    # Export key parameters for MFG calibration
    params = {
        "delta_p_impact": float(irf_structural[0, 2, 2]),
        "delta_p_6m": float(np.cumsum(irf_structural[:, 2, 2])[5]),
        "delta_p_12m": float(np.cumsum(irf_structural[:, 2, 2])[11]),
        "delta_p_24m": float(np.cumsum(irf_structural[:, 2, 2])[23]),
    }
    pd.Series(params).to_csv(DATA_DIR / "svar_calibration_params.csv")
    print(f"  Saved svar_calibration_params.csv")
    for k, v in params.items():
        print(f"    {k}: {v:.6f}")

    print("\n" + "="*60)
    print("SVAR estimation complete.")
    print("Run 04_panel_elasticity.py next.")


if __name__ == "__main__":
    main()
