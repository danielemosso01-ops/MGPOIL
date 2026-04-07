"""
Phase 2b — Panel Elasticity Estimation
========================================
Estimates short-run and long-run price elasticity of oil demand
using a panel of OECD countries with fixed effects.

Specification:
  Short-run: ln(Q_it) = alpha_i + beta * ln(P_t) + gamma * X_it + eps_it
  Long-run:  includes lagged dependent variable → LR elasticity = beta / (1 - rho)

Output:
  - data/clean/elasticity_results.csv
  - figures/elasticity_panel.png

Usage:
  python 04_panel_elasticity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "clean"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_panel():
    """Load the country-month panel."""
    path = DATA_DIR / "panel_monthly.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def estimate_short_run_elasticity(df: pd.DataFrame) -> dict:
    """
    Fixed-effects panel estimation of short-run price elasticity.

    ln(Q_it) = alpha_i + beta * ln(P_it) + delta_t + eps_it

    Uses within transformation (demeaned by country).
    """
    print("\n  Short-run elasticity estimation...")

    df = df.copy()
    df["ln_q"] = np.log(df["consumption"].clip(lower=1))
    df["ln_p"] = np.log(df["oil_price"].clip(lower=1))

    # Within transformation (demean by country)
    for col in ["ln_q", "ln_p"]:
        group_mean = df.groupby("country")[col].transform("mean")
        df[f"{col}_dm"] = df[col] - group_mean

    # Time dummies (year-quarter)
    df["yq"] = df["date"].dt.to_period("Q").astype(str)

    # OLS on demeaned data
    X = sm.add_constant(df["ln_p_dm"])
    y = df["ln_q_dm"]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["country"]})

    beta_sr = model.params.iloc[1]
    se_sr = model.bse.iloc[1]
    t_stat = model.tvalues.iloc[1]
    p_val = model.pvalues.iloc[1]

    print(f"    beta (short-run) = {beta_sr:.4f} (SE={se_sr:.4f}, t={t_stat:.2f}, p={p_val:.4f})")
    print(f"    R-squared (within): {model.rsquared:.4f}")
    print(f"    N = {model.nobs:.0f}")

    return {
        "beta_sr": beta_sr,
        "se_sr": se_sr,
        "t_stat_sr": t_stat,
        "p_value_sr": p_val,
        "r2_within_sr": model.rsquared,
        "n_obs": int(model.nobs),
        "model_sr": model,
    }


def estimate_long_run_elasticity(df: pd.DataFrame) -> dict:
    """
    Dynamic panel estimation with lagged dependent variable.

    ln(Q_it) = alpha_i + rho * ln(Q_i,t-1) + beta * ln(P_it) + eps_it

    Long-run elasticity: beta_LR = beta / (1 - rho)

    Note: Nickell bias is a concern with fixed effects + lagged DV.
    With T~250 months, the bias is small.
    """
    print("\n  Long-run elasticity estimation (dynamic panel)...")

    df = df.copy().sort_values(["country", "date"])
    df["ln_q"] = np.log(df["consumption"].clip(lower=1))
    df["ln_p"] = np.log(df["oil_price"].clip(lower=1))

    # Lagged dependent variable
    df["ln_q_lag"] = df.groupby("country")["ln_q"].shift(1)
    df = df.dropna(subset=["ln_q_lag"])

    # Within transformation
    for col in ["ln_q", "ln_p", "ln_q_lag"]:
        group_mean = df.groupby("country")[col].transform("mean")
        df[f"{col}_dm"] = df[col] - group_mean

    X = df[["ln_q_lag_dm", "ln_p_dm"]]
    X = sm.add_constant(X)
    y = df["ln_q_dm"]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["country"]})

    rho = model.params["ln_q_lag_dm"]
    beta = model.params["ln_p_dm"]
    beta_lr = beta / (1 - rho)

    # Delta method for SE of long-run elasticity
    se_beta = model.bse["ln_p_dm"]
    se_rho = model.bse["ln_q_lag_dm"]
    # Approximate SE via delta method
    se_lr = np.sqrt(
        (se_beta / (1 - rho))**2 +
        (beta * se_rho / (1 - rho)**2)**2
    )

    print(f"    rho (persistence)   = {rho:.4f}")
    print(f"    beta (short-run)    = {beta:.4f}")
    print(f"    beta_LR (long-run)  = {beta_lr:.4f} (SE={se_lr:.4f})")
    print(f"    R-squared (within): {model.rsquared:.4f}")

    return {
        "rho": rho,
        "beta_dynamic_sr": beta,
        "beta_lr": beta_lr,
        "se_lr": se_lr,
        "r2_within_lr": model.rsquared,
        "model_lr": model,
    }


def plot_elasticity_results(df, sr_results, lr_results):
    """Diagnostic plots for elasticity estimation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Scatter: ln(P) vs ln(Q), by country
    ax = axes[0]
    df_plot = df.copy()
    df_plot["ln_q"] = np.log(df_plot["consumption"].clip(lower=1))
    df_plot["ln_p"] = np.log(df_plot["oil_price"].clip(lower=1))
    countries = df_plot["country"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
    for c, color in zip(countries, colors):
        mask = df_plot["country"] == c
        ax.scatter(df_plot.loc[mask, "ln_p"], df_plot.loc[mask, "ln_q"],
                  s=3, alpha=0.3, color=color, label=c)
    ax.set_xlabel("ln(Oil Price)", fontsize=10)
    ax.set_ylabel("ln(Consumption)", fontsize=10)
    ax.set_title("Log Price vs Log Consumption\nby Country", fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, markerscale=3)

    # 2. Elasticity comparison
    ax = axes[1]
    labels = ["Short-run\n(static FE)", "Short-run\n(dynamic)", "Long-run\n(dynamic)"]
    values = [sr_results["beta_sr"], lr_results["beta_dynamic_sr"], lr_results["beta_lr"]]
    errors = [sr_results["se_sr"] * 1.96, 0, lr_results["se_lr"] * 1.96]
    colors_bar = ["steelblue", "darkorange", "seagreen"]
    bars = ax.bar(labels, values, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.errorbar(range(len(values)), values, yerr=errors, fmt='none', color='black', capsize=4)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Elasticity", fontsize=10)
    ax.set_title("Price Elasticity of Oil Demand\n(95% CI where available)", fontsize=11, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.01,
               f'{val:.3f}', ha='center', va='top', fontsize=9, fontweight='bold', color='white')

    # 3. Rolling elasticity over time
    ax = axes[2]
    df_roll = df.copy().sort_values("date")
    df_roll["ln_q"] = np.log(df_roll["consumption"].clip(lower=1))
    df_roll["ln_p"] = np.log(df_roll["oil_price"].clip(lower=1))
    # For each year, estimate cross-sectional elasticity
    df_roll["year"] = df_roll["date"].dt.year
    years = sorted(df_roll["year"].unique())
    rolling_beta = []
    for yr in years:
        mask = df_roll["year"] == yr
        sub = df_roll[mask]
        if len(sub) > 20:
            for col in ["ln_q", "ln_p"]:
                gm = sub.groupby("country")[col].transform("mean")
                sub.loc[:, f"{col}_dm"] = sub[col] - gm
            try:
                X = sm.add_constant(sub["ln_p_dm"])
                model = sm.OLS(sub["ln_q_dm"], X).fit()
                rolling_beta.append({"year": yr, "beta": model.params.iloc[1]})
            except Exception:
                pass

    if rolling_beta:
        rb = pd.DataFrame(rolling_beta)
        ax.plot(rb["year"], rb["beta"], 'b-o', markersize=4, linewidth=1.5)
        ax.axhline(sr_results["beta_sr"], color='red', linestyle='--',
                  label=f'Full sample: {sr_results["beta_sr"]:.3f}')
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Annual Elasticity", fontsize=10)
        ax.set_title("Rolling Annual Price Elasticity", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "elasticity_panel.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved figures/elasticity_panel.png")


def main():
    print("="*60)
    print("Phase 2b: Panel Elasticity Estimation")
    print("="*60)

    df = load_panel()
    print(f"  Panel: {df['country'].nunique()} countries, "
          f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")

    sr = estimate_short_run_elasticity(df)
    lr = estimate_long_run_elasticity(df)

    plot_elasticity_results(df, sr, lr)

    # Save results
    results = {
        "beta_sr": sr["beta_sr"],
        "se_sr": sr["se_sr"],
        "beta_lr": lr["beta_lr"],
        "se_lr": lr["se_lr"],
        "rho": lr["rho"],
        "n_obs": sr["n_obs"],
    }
    pd.Series(results).to_csv(DATA_DIR / "elasticity_results.csv")
    print(f"\n  Saved elasticity_results.csv")
    print(f"  Short-run elasticity (c parameter for MFG): {sr['beta_sr']:.4f}")
    print(f"  Long-run elasticity: {lr['beta_lr']:.4f}")

    print("\n" + "="*60)
    print("Elasticity estimation complete.")
    print("Run 05_threshold_regression.py next.")


if __name__ == "__main__":
    main()
