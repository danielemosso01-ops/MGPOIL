"""
Phase 1a — Data Collection
===========================
Downloads open-source datasets for the oil consumer coordination paper.

Sources:
  - EIA Open Data API (requires API key in env var EIA_API_KEY)
  - World Bank Commodity Prices (public CSV)
  - Our World in Data — EV adoption (GitHub CSV)
  - Google Trends via pytrends

Usage:
  set EIA_API_KEY=your_key_here
  python 01_collect_data.py
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. EIA Open Data — Oil consumption, prices, OPEC production
# ============================================================

def fetch_eia_series(series_id: str, api_key: str, description: str) -> pd.DataFrame:
    """Fetch a single series from EIA API v2."""
    url = "https://api.eia.gov/v2/seriesid/" + series_id
    params = {"api_key": api_key}
    print(f"  Fetching EIA: {description} ({series_id})...")
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if "response" in data and "data" in data["response"]:
            records = data["response"]["data"]
            df = pd.DataFrame(records)
            return df
        elif "data" in data:
            records = data["data"]
            df = pd.DataFrame(records)
            return df
        else:
            print(f"    Warning: unexpected response structure for {series_id}")
            return pd.DataFrame()
    except Exception as e:
        print(f"    Error fetching {series_id}: {e}")
        return pd.DataFrame()


def collect_eia_data(api_key: str):
    """Collect oil market data from EIA API."""
    print("\n[1/5] Collecting EIA data...")

    # --- Monthly WTI and Brent prices ---
    price_series = {
        "PET.RWTC.M": "WTI Spot Price Monthly",
        "PET.RBRTE.M": "Brent Spot Price Monthly",
    }

    all_prices = []
    for sid, desc in price_series.items():
        df = fetch_eia_series(sid, api_key, desc)
        if not df.empty:
            df["series"] = sid
            all_prices.append(df)

    if all_prices:
        prices_df = pd.concat(all_prices, ignore_index=True)
        prices_df.to_csv(DATA_DIR / "eia_oil_prices.csv", index=False)
        print(f"  Saved eia_oil_prices.csv ({len(prices_df)} rows)")
    else:
        print("  Warning: no EIA price data retrieved. Using fallback method...")
        fetch_eia_prices_fallback()

    # --- OECD petroleum consumption ---
    consumption_series = {
        "PET.MTTUPUS1.M": "US Total Petroleum Consumption Monthly",
    }
    all_cons = []
    for sid, desc in consumption_series.items():
        df = fetch_eia_series(sid, api_key, desc)
        if not df.empty:
            df["series"] = sid
            all_cons.append(df)

    if all_cons:
        cons_df = pd.concat(all_cons, ignore_index=True)
        cons_df.to_csv(DATA_DIR / "eia_consumption.csv", index=False)
        print(f"  Saved eia_consumption.csv ({len(cons_df)} rows)")

    # --- World production ---
    prod_series = {
        "PET.MCRFPUS2.M": "US Crude Oil Production Monthly",
        "PET.MCRFPW02.M": "World Crude Oil Production Monthly",
    }
    all_prod = []
    for sid, desc in prod_series.items():
        df = fetch_eia_series(sid, api_key, desc)
        if not df.empty:
            df["series"] = sid
            all_prod.append(df)

    if all_prod:
        prod_df = pd.concat(all_prod, ignore_index=True)
        prod_df.to_csv(DATA_DIR / "eia_production.csv", index=False)
        print(f"  Saved eia_production.csv ({len(prod_df)} rows)")


def fetch_eia_prices_fallback():
    """Fallback: download EIA petroleum prices from direct download page."""
    url = "https://www.eia.gov/dnav/pet/hist_xls/RWTCm.xls"
    print("  Attempting direct EIA XLS download for WTI...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path = DATA_DIR / "eia_wti_monthly.xls"
        path.write_bytes(r.content)
        print(f"  Saved {path.name}")
    except Exception as e:
        print(f"  Fallback download failed: {e}")


# ============================================================
# 2. World Bank Commodity Prices
# ============================================================

def collect_worldbank_data():
    """Download World Bank commodity price data (public CSV)."""
    print("\n[2/5] Collecting World Bank commodity prices...")
    url = ("https://thedocs.worldbank.org/en/doc/"
           "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
           "CMO-Historical-Data-Monthly.xlsx")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path = DATA_DIR / "worldbank_commodity_prices.xlsx"
        path.write_bytes(r.content)
        print(f"  Saved worldbank_commodity_prices.xlsx ({len(r.content)//1024} KB)")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Trying alternative World Bank endpoint...")
        alt_url = ("https://www.worldbank.org/en/research/commodity-markets")
        print(f"  Please download manually from: {alt_url}")


# ============================================================
# 3. Our World in Data — EV adoption
# ============================================================

def collect_owid_ev_data():
    """Download EV adoption data from Our World in Data GitHub."""
    print("\n[3/5] Collecting OWID EV adoption data...")
    url = ("https://raw.githubusercontent.com/owid/etl/master/etl/steps/data/"
           "garden/iea/2024-11-20/global_ev_outlook.csv")
    alt_urls = [
        "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/"
        "Electric%20car%20share%20of%20new%20car%20sales%20-%20IEA%20(2024)/"
        "Electric%20car%20share%20of%20new%20car%20sales%20-%20IEA%20(2024).csv",
        "https://raw.githubusercontent.com/owid/etl/master/etl/steps/data/"
        "garden/iea/2024-11-20/global_ev_data_explorer.csv",
        "https://catalog.ourworldindata.org/garden/iea/2024-11-20/"
        "global_ev_outlook/global_ev_outlook.csv",
    ]

    for u in [url] + alt_urls:
        try:
            r = requests.get(u, timeout=60)
            r.raise_for_status()
            path = DATA_DIR / "owid_ev_adoption.csv"
            path.write_bytes(r.content)
            print(f"  Saved owid_ev_adoption.csv ({len(r.content)//1024} KB)")
            return
        except Exception:
            continue

    # Final fallback: use the OWID catalog endpoint
    print("  Direct CSV failed. Trying OWID catalog...")
    try:
        catalog_url = ("https://ourworldindata.org/grapher/electric-car-share-of-new-car-sales"
                       "?tab=table&time=2010..latest&v=1&csvType=full&useColumnShortNames=true")
        r = requests.get(catalog_url, timeout=60, headers={"Accept": "text/csv"})
        r.raise_for_status()
        path = DATA_DIR / "owid_ev_adoption.csv"
        path.write_bytes(r.content)
        print(f"  Saved owid_ev_adoption.csv ({len(r.content)//1024} KB)")
    except Exception as e:
        print(f"  All OWID downloads failed: {e}")
        print("  Please download manually from: https://ourworldindata.org/electric-car")


# ============================================================
# 4. Google Trends via pytrends
# ============================================================

def collect_google_trends():
    """Fetch Google Trends data for coordination proxy terms."""
    print("\n[4/5] Collecting Google Trends data...")
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  pytrends not installed. Run: pip install pytrends")
        return

    pytrends = TrendReq(hl='en-US', tz=360)
    keywords = [
        "reduce fuel consumption",
        "electric vehicle",
        "fuel boycott",
        "oil demand",
    ]
    geos = {"US": "US", "DE": "DE", "FR": "FR", "GB": "GB", "": "global"}

    all_trends = []
    for geo_code, geo_name in geos.items():
        print(f"  Fetching trends for geo={geo_name}...")
        for kw in keywords:
            try:
                pytrends.build_payload(
                    [kw],
                    cat=0,
                    timeframe='2004-01-01 2024-12-31',
                    geo=geo_code,
                )
                df = pytrends.interest_over_time()
                if not df.empty:
                    df = df.reset_index()
                    df["keyword"] = kw
                    df["geo"] = geo_name
                    all_trends.append(df[["date", kw, "keyword", "geo"]])
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"    Error for '{kw}' in {geo_name}: {e}")
                time.sleep(5)

    if all_trends:
        trends_df = pd.concat(all_trends, ignore_index=True)
        trends_df.to_csv(DATA_DIR / "google_trends.csv", index=False)
        print(f"  Saved google_trends.csv ({len(trends_df)} rows)")
    else:
        print("  No Google Trends data retrieved.")


# ============================================================
# 5. OPEC Annual Statistical Bulletin
# ============================================================

def collect_opec_data():
    """Download OPEC production data."""
    print("\n[5/5] Collecting OPEC production data...")
    # OPEC ASB data — the direct download URL changes yearly
    urls = [
        "https://asb.opec.org/data/ASB_Data.xlsx",
        "https://www.opec.org/opec_web/static_files_project/media/downloads/"
        "publications/ASB/ASB_Data_2024.xlsx",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=120, headers={
                "User-Agent": "Mozilla/5.0 (academic research)"
            })
            r.raise_for_status()
            path = DATA_DIR / "opec_production.xlsx"
            path.write_bytes(r.content)
            print(f"  Saved opec_production.xlsx ({len(r.content)//1024} KB)")
            return
        except Exception:
            continue

    print("  OPEC direct download not available.")
    print("  Please download manually from: https://asb.opec.org/")
    print("  Save as: data/raw/opec_production.xlsx")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        print("WARNING: EIA_API_KEY not set. EIA data collection will be limited.")
        print("Register at https://www.eia.gov/opendata/register.php")
        print("Then: set EIA_API_KEY=your_key_here\n")

    if api_key:
        collect_eia_data(api_key)
    else:
        print("\n[1/5] Skipping EIA API (no key). Using fallback...")
        fetch_eia_prices_fallback()

    collect_worldbank_data()
    collect_owid_ev_data()
    collect_google_trends()
    collect_opec_data()

    print("\n" + "="*60)
    print("Data collection complete.")
    print(f"Raw files saved in: {DATA_DIR}")
    print("Run 02_clean_merge_data.py next.")
