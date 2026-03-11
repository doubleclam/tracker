#!/usr/bin/env python3
"""
Market Intelligence: Gold - Full 47-Factor Dynamic Engine
Calculates 5Y Means, Z-Scores, and Percentiles dynamically from historical API data.
"""

import os
import ssl
import re
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
from pytrends.request import TrendReq
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import OrderedDict
import xml.etree.ElementTree as ET

# ─── Security & API Setup ────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
FRED_API_KEY = "412665086b998f7954423844843240b6"
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

st.set_page_config(page_title="Market Intelligence: Gold", layout="wide")

# FT-style page background
st.markdown("""
<style>
    .stApp {
        background-color: #FFF1E5;
    }
</style>
""", unsafe_allow_html=True)

# ─── Factor Configuration (loaded from spreadsheet) ──────────────────────────
# Config lives in gold_scoring_config.xlsx for easy review and editing.
# Each factor has: ind, ticker, source, weight, higher_is_bullish,
# cluster_group, why, definition, unit

def load_factor_config():
    """Load factor configuration from gold_scoring_config.xlsx.
    Returns OrderedDict of {category: [factor_dicts]} matching the old FACTOR_CONFIG format.
    Falls back to an error message if the file is missing."""
    xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_scoring_config.xlsx")
    if not os.path.exists(xlsx_path):
        st.error(f"Scoring config not found: {xlsx_path}")
        return OrderedDict()

    df = pd.read_excel(xlsx_path, sheet_name="Factor Config", engine="openpyxl")

    config = OrderedDict()
    for _, row in df.iterrows():
        cat = str(row.get("Category", ""))
        entry = {
            "ind": str(row.get("Indicator", "")),
            "ticker": str(row.get("Ticker", "")),
            "source": str(row.get("Source", "")),
            "weight": int(row.get("Weight", 1)),
            "higher_is_bullish": bool(row.get("Higher Is Bullish", False)),
            "cluster_group": str(row.get("Cluster Group", "")),
            "why": str(row.get("Why", "")),
            "definition": str(row.get("Definition", "")),
            "unit": str(row.get("Unit", "")),
        }
        if cat not in config:
            config[cat] = []
        config[cat].append(entry)

    return config

FACTOR_CONFIG = load_factor_config()

# ─── Dynamic Historical Data Engine ──────────────────────────────────────────

def fetch_cftc_cot_data(start_date, end_date):
    """
    Fetch CFTC Commitments of Traders (Disaggregated) data for Gold futures.
    Returns dict with COT_MM (Managed Money net position), COT_SD (Swap Dealer net position),
    and COMEX_OI (Open Interest) as pandas Series.

    Uses the CFTC public Socrata API:
    https://publicreporting.cftc.gov/resource/72hh-3qpy.json (Disaggregated Futures-Only)
    Gold contract code = 088691
    """
    results = {}
    errors = []

    # CFTC API: Disaggregated Futures-Only report
    base_url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"

    # Format dates for Socrata API query
    start_str = start_date.strftime("%Y-%m-%dT00:00:00.000")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59.000")

    # Fetch all weekly reports for Gold (088691) in the 5Y window
    # Socrata has a 1000-row default limit; 5 years ≈ 260 weeks, so 1000 is sufficient
    params = {
        "$where": f"cftc_contract_market_code='088691' AND report_date_as_yyyy_mm_dd >= '{start_str}' AND report_date_as_yyyy_mm_dd <= '{end_str}'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 1000
    }

    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            errors.append("CFTC COT: API returned empty data for Gold (088691)")
            return results, errors

        dates = []
        mm_net = []       # Managed Money net (long - short)
        sd_net = []       # Swap Dealer net (long - short)
        oi_vals = []      # Open Interest

        for record in data:
            try:
                # Parse the date
                date_str = record.get("report_date_as_yyyy_mm_dd", "")
                if not date_str:
                    continue
                dt = pd.Timestamp(date_str[:10])

                # Managed Money: long - short = net position
                mm_long = float(record.get("m_money_positions_long_all", 0))
                mm_short = float(record.get("m_money_positions_short_all", 0))

                # Swap Dealers: long - short = net position
                # Note: CFTC API has double underscore in swap short field name
                sd_long = float(record.get("swap_positions_long_all", 0))
                sd_short = float(record.get("swap__positions_short_all", record.get("swap_positions_short_all", 0)))

                # Open Interest
                oi = float(record.get("open_interest_all", 0))

                dates.append(dt)
                mm_net.append(mm_long - mm_short)
                sd_net.append(sd_long - sd_short)
                oi_vals.append(oi)

            except (ValueError, TypeError) as e:
                continue  # skip malformed records

        if dates:
            idx = pd.DatetimeIndex(dates)
            results["COT_MM"] = pd.Series(mm_net, index=idx, name="COT_MM")
            results["COT_SD"] = pd.Series(sd_net, index=idx, name="COT_SD")
            results["COMEX_OI"] = pd.Series(oi_vals, index=idx, name="COMEX_OI")
        else:
            errors.append("CFTC COT: parsed 0 valid records from API response")

    except requests.exceptions.RequestException as e:
        errors.append(f"CFTC COT API request failed: {e}")
    except Exception as e:
        errors.append(f"CFTC COT processing error: {e}")

    return results, errors


@st.cache_data(ttl=3600)
def fetch_historical_data():
    """Fetches 5 years of daily/monthly history to calculate true Z-Scores for all 47 factors"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    hist_data = {}
    fetch_errors = []

    # 0. Fetch CFTC COT Data (Managed Money, Swap Dealers, Open Interest)
    cot_data, cot_errors = fetch_cftc_cot_data(start_date, end_date)
    hist_data.update(cot_data)
    fetch_errors.extend(cot_errors)

    # 1. Fetch FRED Data
    if fred:
        fred_tickers = [
            "DFII10", "DFII5", "T5YIE", "MICH", "M2SL", "GFDEGDQ188S",
            "FYFSGDA188S", "A091RC1Q027SBEA", "T10Y2Y", "BAMLH0A0HYM2",
            "TEDRATE", "RRPONTSYD", "TOTRESNS"
        ]
        for t in fred_tickers:
            try:
                series = fred.get_series(t, observation_start=start_date)
                s = series.dropna()
                if len(s) > 0:
                    hist_data[t] = s
                else:
                    fetch_errors.append(f"FRED {t}: returned empty series")
            except Exception as e:
                fetch_errors.append(f"FRED {t}: {e}")

        # Compute Fed Funds Real Rate = FEDFUNDS - T5YIE (inflation expectations)
        try:
            fedfunds = fred.get_series("FEDFUNDS", observation_start=start_date).dropna()
            t5yie = hist_data.get("T5YIE")
            if t5yie is not None and len(fedfunds) > 0:
                # Align monthly fed funds to daily inflation expectations
                fedfunds_daily = fedfunds.reindex(t5yie.index, method="ffill")
                real_ff = (fedfunds_daily - t5yie).dropna()
                if len(real_ff) > 10:
                    hist_data["FEDFUNDS_REAL"] = real_ff
        except Exception as e:
            fetch_errors.append(f"FEDFUNDS_REAL calc: {e}")

        # Compute Gold/M2 ratio
        try:
            m2 = hist_data.get("M2SL")
            if m2 is not None:
                m2_daily = m2.resample("D").ffill()
                hist_data["_M2_DAILY"] = m2_daily  # store for ratio calc later
        except Exception as e:
            fetch_errors.append(f"M2 resample: {e}")

        # Compute Deficit/GDP: FRED stores as negative for deficit.
        # We want "bigger deficit = more bullish for gold", so we negate it
        # so that a larger deficit (e.g. -6%) becomes +6 and scores as bullish.
        if "FYFSGDA188S" in hist_data:
            hist_data["FYFSGDA188S"] = -hist_data["FYFSGDA188S"]

    # 1B. Fetch SOFR and DFF (Fed Funds Daily) from FRED for SOFR-OIS spread
    if fred:
        try:
            sofr = fred.get_series("SOFR", observation_start=start_date).dropna()
            dff = fred.get_series("DFF", observation_start=start_date).dropna()
            if len(sofr) > 10 and len(dff) > 10:
                aligned_rates = pd.concat([sofr, dff], axis=1).dropna()
                aligned_rates.columns = ["SOFR", "DFF"]
                # Spread in basis points
                sofr_ois = (aligned_rates["SOFR"] - aligned_rates["DFF"]) * 100
                if len(sofr_ois) > 10:
                    hist_data["SOFR_OIS"] = sofr_ois
        except Exception as e:
            fetch_errors.append(f"SOFR-OIS calc: {e}")

    # 1C. Fetch Geopolitical Risk Index (Caldara-Iacoviello)
    try:
        gpr_url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
        gpr_df = pd.read_excel(gpr_url, engine="xlrd")
        gpr_series = gpr_df.set_index("month")["GPR"].dropna()
        gpr_series.index = pd.DatetimeIndex(gpr_series.index)
        # Filter to our 5Y window
        gpr_series = gpr_series[gpr_series.index >= start_date]
        if len(gpr_series) > 10:
            hist_data["GPR_IDX"] = gpr_series
    except Exception as e:
        fetch_errors.append(f"GPR Index: {e}")

    # 1D. Build seasonal calendar indicators (India Wedding + China Lunar New Year)
    cal_dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # India Wedding Season: peak Oct-Feb (100), shoulder Mar/Sep (50), off-season Apr-Aug (25)
    india_scores = {1: 100, 2: 100, 3: 50, 4: 25, 5: 25, 6: 25,
                    7: 25, 8: 25, 9: 50, 10: 100, 11: 100, 12: 100}
    hist_data["IN_WEDDING"] = pd.Series(
        [india_scores[d.month] for d in cal_dates], index=cal_dates)

    # China Lunar New Year: peak Jan-Feb (100), Oct Golden Week (75), shoulder (50), off (25)
    china_scores = {1: 100, 2: 100, 3: 50, 4: 25, 5: 25, 6: 25,
                    7: 25, 8: 25, 9: 50, 10: 75, 11: 50, 12: 75}
    hist_data["CN_LNY"] = pd.Series(
        [china_scores[d.month] for d in cal_dates], index=cal_dates)

    # 1E. Fetch Shanghai Gold Exchange premium (SGE benchmark - COMEX, in USD/oz)
    # SGE publishes daily benchmark prices in CNY/gram via a JSON endpoint.
    # We convert to USD/oz using the CNY=X FX rate and subtract COMEX gold futures.
    try:
        sge_url = "https://en.sge.com.cn/graph/DayilyJzj"
        sge_resp = requests.get(sge_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        sge_json = sge_resp.json()
        if "zp" in sge_json and len(sge_json["zp"]) > 0:
            sge_dates = [pd.Timestamp(p[0], unit="ms") for p in sge_json["zp"]]
            sge_prices = [float(p[1]) for p in sge_json["zp"]]
            sge_cny_g = pd.Series(sge_prices, index=sge_dates, name="SGE_CNY_g")
            # Filter to 5Y window
            sge_cny_g = sge_cny_g[sge_cny_g.index >= start_date]
            if len(sge_cny_g) > 10:
                hist_data["_SGE_CNY_G"] = sge_cny_g  # store for premium calc after YF data
    except Exception as e:
        fetch_errors.append(f"SGE benchmark: {e}")

    # 1F. Fetch Google Trends for "buy gold" (5Y weekly via pytrends)
    # Note: Google rate-limits pytrends aggressively (429 errors are common).
    # This is non-critical — GTRENDS falls back to simulated data silently.
    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(kw_list=["buy gold"], timeframe="today 5-y", geo="")
        gt_df = pytrends.interest_over_time()
        if len(gt_df) > 10:
            hist_data["GTRENDS"] = gt_df["buy gold"].astype(float)
    except Exception:
        pass  # Google Trends 429 rate-limits are expected; factor uses simulated fallback

    # 2. Fetch Yahoo Finance Data (now includes FX pairs for gold-in-other-currencies)
    yf_tickers = [
        "DX-Y.NYB", "^VIX", "GC=F", "SI=F", "PL=F", "CL=F",
        "^GSPC", "BTC-USD", "VNQ", "GDX", "GLD", "GDXJ",
        "PA=F", "HG=F",
        "EUR=X", "GBPUSD=X", "JPY=X", "CNY=X"
    ]
    try:
        yf_df = yf.download(yf_tickers, start=start_date, end=end_date, progress=False)

        # Handle both MultiIndex and flat column formats
        if isinstance(yf_df.columns, pd.MultiIndex):
            close_df = yf_df["Close"]
        else:
            close_df = yf_df

        for t in yf_tickers:
            if t in close_df.columns:
                s = close_df[t].dropna()
                if len(s) > 0:
                    hist_data[t] = s
                else:
                    fetch_errors.append(f"YF {t}: no data after dropna")
            else:
                fetch_errors.append(f"YF {t}: not in downloaded columns")

        # Retry failed crypto tickers individually (batch download often returns NaN for crypto)
        crypto_tickers = [t for t in ["BTC-USD"] if t not in hist_data]
        for t in crypto_tickers:
            try:
                solo_df = yf.download(t, start=start_date, end=end_date, progress=False)
                if isinstance(solo_df.columns, pd.MultiIndex):
                    s = solo_df["Close"][t].dropna()
                else:
                    s = solo_df["Close"].dropna()
                if len(s) > 0:
                    hist_data[t] = s
                    # Remove the earlier error since we recovered
                    fetch_errors[:] = [e for e in fetch_errors if t not in e]
            except Exception:
                pass

        # Compute Ratios dynamically
        ratio_pairs = {
            "GC=F/SI=F": ("GC=F", "SI=F"),
            "GC=F/CL=F": ("GC=F", "CL=F"),
            "GC=F/^GSPC": ("GC=F", "^GSPC"),
            "PL=F/GC=F": ("PL=F", "GC=F"),
            "GC=F/BTC-USD": ("GC=F", "BTC-USD"),
            "GC=F/VNQ": ("GC=F", "VNQ"),
            "GDX/GLD": ("GDX", "GLD"),
            "GDXJ/GDX": ("GDXJ", "GDX"),
        }
        for ratio_name, (num, den) in ratio_pairs.items():
            if num in close_df.columns and den in close_df.columns:
                ratio = (close_df[num] / close_df[den]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(ratio) > 10:
                    hist_data[ratio_name] = ratio
                else:
                    fetch_errors.append(f"Ratio {ratio_name}: insufficient data ({len(ratio)} pts)")

        # Compute Gold/M2 ratio from real data
        if "GC=F" in hist_data and "_M2_DAILY" in hist_data:
            gc = hist_data["GC=F"]
            m2d = hist_data["_M2_DAILY"]
            aligned = pd.concat([gc, m2d], axis=1).dropna()
            if len(aligned) > 10:
                hist_data["AU_M2"] = (aligned.iloc[:, 0] / aligned.iloc[:, 1]).dropna()

        # Compute Silver Beta (60-day rolling beta of SI vs GC)
        if "SI=F" in hist_data and "GC=F" in hist_data:
            si_ret = hist_data["SI=F"].pct_change().dropna()
            gc_ret = hist_data["GC=F"].pct_change().dropna()
            aligned = pd.concat([si_ret, gc_ret], axis=1).dropna()
            aligned.columns = ["SI", "GC"]
            if len(aligned) > 60:
                rolling_cov = aligned["SI"].rolling(60).cov(aligned["GC"])
                rolling_var = aligned["GC"].rolling(60).var()
                beta = (rolling_cov / rolling_var).dropna()
                if len(beta) > 10:
                    hist_data["SI_BETA"] = beta

        # Compute Moving Average signal (price above 50D & 200D = bullish)
        if "GC=F" in hist_data:
            gc = hist_data["GC=F"]
            ma50 = gc.rolling(50).mean()
            ma200 = gc.rolling(200).mean()
            # Score: 2 if above both, 1 if above one, 0 if below both
            ma_signal = ((gc > ma50).astype(float) + (gc > ma200).astype(float)).dropna()
            if len(ma_signal) > 200:
                hist_data["MA_BULL"] = ma_signal

        # Compute RSI (14-day) for gold
        if "GC=F" in hist_data:
            gc = hist_data["GC=F"]
            delta = gc.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.dropna()
            if len(rsi) > 10:
                hist_data["RSI_DIV"] = rsi

        # Compute Bollinger Band width for gold
        if "GC=F" in hist_data:
            gc = hist_data["GC=F"]
            bb_mid = gc.rolling(20).mean()
            bb_std = gc.rolling(20).std()
            bb_width = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid * 100
            bb_width = bb_width.dropna()
            if len(bb_width) > 10:
                hist_data["BB_EXP"] = bb_width

        # Compute Futures Structure: GC=F (front-month futures) minus GLD*conversion as spot proxy
        # GLD represents ~0.0925 oz per share; inverse ≈ 10.81 but drifts with expense ratio
        # We use a dynamic ratio: GC=F / GLD gives the implied conversion factor
        if "GC=F" in hist_data and "GLD" in hist_data:
            gc = hist_data["GC=F"]
            gld = hist_data["GLD"]
            # Compute basis spread: positive = contango, negative = backwardation
            # Use GC=F - GLD * (GC=F.iloc[0] / GLD.iloc[0]) to anchor the conversion factor
            conv_factor = gc.iloc[0] / gld.iloc[0] if gld.iloc[0] != 0 else 10.0
            basis = (gc - gld * conv_factor).replace([np.inf, -np.inf], np.nan).dropna()
            if len(basis) > 10:
                hist_data["FUT_CURVE"] = basis

        # Compute Gold in Other Currencies basket (equal-weighted index)
        # EUR=X gives USD per EUR, others give currency per USD
        if "GC=F" in hist_data:
            gc = hist_data["GC=F"]
            fx_components = []

            # Gold in EUR: GC=F * EUR=X (EUR=X = USD per 1 EUR, so gold_eur = GC=F / (1/EUR=X) = GC=F * EUR=X)
            # Actually EUR=X on YF returns EUR per 1 USD. So gold_eur = GC=F * EUR=X
            if "EUR=X" in hist_data:
                fx_components.append(gc * hist_data["EUR=X"])
            # Gold in GBP: GBPUSD=X = USD per 1 GBP, so gold_gbp = GC=F / GBPUSD=X
            if "GBPUSD=X" in hist_data:
                fx_components.append(gc / hist_data["GBPUSD=X"])
            # Gold in JPY: JPY=X on YF = USD/JPY? Actually YF JPY=X returns JPY per 1 USD
            # So gold_jpy = GC=F * JPY=X (but very large numbers, we'll normalize)
            if "JPY=X" in hist_data:
                fx_components.append(gc * hist_data["JPY=X"])
            # Gold in CNY: CNY=X = CNY per 1 USD, so gold_cny = GC=F * CNY=X
            if "CNY=X" in hist_data:
                fx_components.append(gc * hist_data["CNY=X"])

            if len(fx_components) >= 2:
                # Normalize each to base=100 at start, then average
                normalized = []
                for comp in fx_components:
                    comp = comp.dropna()
                    if len(comp) > 10:
                        normalized.append(comp / comp.iloc[0] * 100)
                if normalized:
                    basket = pd.concat(normalized, axis=1).dropna().mean(axis=1)
                    if len(basket) > 10:
                        hist_data["XAU_BASKET"] = basket

        # Compute Shanghai Premium: SGE (CNY/g -> USD/oz) minus COMEX
        if "_SGE_CNY_G" in hist_data and "GC=F" in hist_data and "CNY=X" in hist_data:
            try:
                sge_cny = hist_data["_SGE_CNY_G"].resample("D").ffill()
                cny_rate = hist_data["CNY=X"].resample("D").ffill()
                gc_daily = hist_data["GC=F"].resample("D").ffill()
                sge_aligned = pd.concat([sge_cny, cny_rate, gc_daily], axis=1).dropna()
                sge_aligned.columns = ["sge_cny_g", "cny_per_usd", "comex"]
                # Convert SGE CNY/gram to USD/oz: price * 31.1035 / cny_rate
                sge_aligned["sge_usd_oz"] = sge_aligned["sge_cny_g"] * 31.1035 / sge_aligned["cny_per_usd"]
                sge_premium = sge_aligned["sge_usd_oz"] - sge_aligned["comex"]
                sge_premium = sge_premium.dropna()
                if len(sge_premium) > 10:
                    hist_data["SGE_PREM"] = sge_premium
            except Exception as e:
                fetch_errors.append(f"SGE premium calc: {e}")

        # Compute ETF Flows proxy: 20-day average dollar volume of GLD (in billions)
        if "GLD" in hist_data:
            try:
                gld_full = yf_df
                if isinstance(gld_full.columns, pd.MultiIndex):
                    gld_vol = gld_full[("Volume", "GLD")].dropna()
                    gld_close = gld_full[("Close", "GLD")].dropna()
                else:
                    gld_vol = gld_full["Volume"].dropna() if "Volume" in gld_full.columns else None
                    gld_close = gld_full["Close"].dropna() if "Close" in gld_full.columns else None

                if gld_vol is not None and gld_close is not None:
                    aligned_gld = pd.concat([gld_close, gld_vol], axis=1).dropna()
                    aligned_gld.columns = ["close", "vol"]
                    dollar_vol = (aligned_gld["close"] * aligned_gld["vol"]) / 1e9  # in $B
                    etf_flow = dollar_vol.rolling(20).mean().dropna()
                    if len(etf_flow) > 10:
                        hist_data["ETF_FLOWS"] = etf_flow
            except Exception as e:
                fetch_errors.append(f"ETF_FLOWS calc: {e}")

        # Compute Volume Profile: 20-day avg volume / 5Y avg volume ratio for GC=F
        try:
            if isinstance(yf_df.columns, pd.MultiIndex):
                gc_vol = yf_df[("Volume", "GC=F")].dropna()
            else:
                gc_vol = None

            if gc_vol is not None and len(gc_vol) > 200:
                avg_5y = gc_vol.mean()
                if avg_5y > 0:
                    vol_ratio = gc_vol.rolling(20).mean() / avg_5y
                    vol_ratio = vol_ratio.dropna()
                    if len(vol_ratio) > 10:
                        hist_data["VOL_PROF"] = vol_ratio
        except Exception as e:
            fetch_errors.append(f"VOL_PROF calc: {e}")

    except Exception as e:
        fetch_errors.append(f"YF bulk download: {e}")

    # 3. Simulate remaining factors that have no free API source
    # These use random walks as placeholders until real data sources are connected.
    np.random.seed(int(end_date.strftime("%d")))
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    simulated_count = 0
    for category in FACTOR_CONFIG.values():
        for item in category:
            ticker = item['ticker']
            if item['source'] == 'SIMULATED' and ticker not in hist_data:
                steps = np.random.normal(loc=0.01, scale=1.5, size=len(dates))
                walk = np.cumsum(steps)
                walk = (walk - walk.min()) + 5.0
                hist_data[ticker] = pd.Series(walk, index=dates)
                simulated_count += 1

    # Store metadata for display
    hist_data["_fetch_errors"] = fetch_errors
    hist_data["_simulated_count"] = simulated_count

    return hist_data

def format_value_with_unit(val, unit):
    """Format a numeric value with its unit for display."""
    if unit in ("%", "% GDP", "% ann.", "pp"):
        return f"{val:.2f}{unit}"
    elif unit in ("$B", "$B/yr"):
        if abs(val) >= 1000:
            return f"${val/1000:.1f}T"
        return f"${val:.0f}B"
    elif unit == "$T":
        return f"${val:.2f}T"
    elif unit == "$M/pt":
        return f"${val:.1f}M/pt"
    elif unit == "$M/mo":
        return f"${val:.1f}M/mo"
    elif unit in ("$/oz",):
        return f"${val:.2f}/oz"
    elif unit == "pts":
        return f"{val:.2f} pts"
    elif unit in ("vol pts",):
        return f"{val:.1f} vol"
    elif unit == "bps":
        return f"{val:.1f} bps"
    elif unit in ("x",):
        return f"{val:.2f}x"
    elif unit == "ratio":
        return f"{val:.4f}"
    elif unit in ("contracts", "M oz", "oz"):
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.2f}M"
        elif abs(val) >= 1000:
            return f"{val/1000:.1f}K"
        return f"{val:.0f}"
    elif unit in ("tonnes", "tonnes/yr"):
        return f"{val:.0f} t"
    elif unit == "bbl/oz":
        return f"{val:.1f} bbl/oz"
    elif unit == "/2":
        return f"{val:.0f}/2"
    elif unit == "0-100":
        return f"{val:.1f}"
    elif unit == "index":
        return f"{val:.2f}"
    else:
        return f"{val:.2f} {unit}"


def make_sparkline_svg(series, n_days=50, width=120, height=28):
    """
    Generate an inline SVG sparkline from the last n_days of a pandas Series.
    Returns an HTML string containing the SVG element.

    The line is colored:
      - green if the last value > first value (uptrend over the window)
      - red   if the last value < first value (downtrend)
      - gray  if flat / insufficient data

    A small dot marks the current (rightmost) value.
    """
    if series is None or len(series) < 3:
        return f'<svg width="{width}" height="{height}"></svg>'

    # Take the last n_days data points
    tail = series.dropna().tail(n_days)
    if len(tail) < 3:
        return f'<svg width="{width}" height="{height}"></svg>'

    values = tail.values.astype(float)
    n = len(values)

    # Determine color from trend direction
    if values[-1] > values[0] * 1.001:
        stroke = "#22c55e"  # green = up
    elif values[-1] < values[0] * 0.999:
        stroke = "#ef4444"  # red = down
    else:
        stroke = "#888"     # gray = flat

    # Normalize values to SVG coordinate space
    v_min = values.min()
    v_max = values.max()
    v_range = v_max - v_min
    if v_range == 0:
        v_range = 1.0  # avoid div-by-zero for constant series

    pad_y = 3  # vertical padding in px
    usable_h = height - 2 * pad_y

    points = []
    for i, v in enumerate(values):
        x = (i / (n - 1)) * width
        y = pad_y + usable_h - ((v - v_min) / v_range) * usable_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    last_x = width
    last_y = pad_y + usable_h - ((values[-1] - v_min) / v_range) * usable_h

    svg = (
        f'<svg width="{width}" height="{height}" style="display:block;">'
        f'<polyline points="{polyline}" fill="none" stroke="{stroke}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2" fill="{stroke}"/>'
        f'</svg>'
    )
    return svg


def calculate_statistics(series, config):
    """Dynamically calculates 5Y Mean, Std, Z-Score, and Confidence Scores"""
    unit = config.get("unit", "")
    if series is None or len(series) < 10:
        return {"Value": "N/A", "Colour Indicator": "Yellow", "Total Factor Score": 0,
                "Change": "N/A", "Change Color": "gray"}

    current_val = float(series.iloc[-1])
    mean_5y = float(series.mean())
    std_5y = float(series.std())

    # Period-over-period % change (daily or weekly depending on data frequency)
    if len(series) >= 2:
        prev_val = float(series.iloc[-2])
        if abs(prev_val) > 1e-10:
            change_pct = (current_val - prev_val) / abs(prev_val) * 100
        else:
            change_pct = 0.0
        if change_pct > 0.005:
            change_str = f"▲ +{change_pct:.2f}%"
            change_color = "green"
        elif change_pct < -0.005:
            change_str = f"▼ {change_pct:.2f}%"
            change_color = "red"
        else:
            change_str = "— 0.00%"
            change_color = "gray"
    else:
        change_str = "N/A"
        change_color = "gray"

    # Mathematical Z-Score Formula: (Current - Mean) / Std
    z_score = (current_val - mean_5y) / std_5y if std_5y != 0 else 0.0

    # Percentile
    percentile = float((series < current_val).mean() * 100)

    # Directional Scoring
    weight = config['weight']
    is_bullish = config['higher_is_bullish']

    if is_bullish:
        raw_score = z_score * weight
    else:
        raw_score = -z_score * weight

    # Cap maximum scores structurally
    factor_score = max(min(raw_score, weight * 3), -weight * 3)

    if factor_score > (weight * 0.5):
        color = "Green"
    elif factor_score < -(weight * 0.5):
        color = "Red"
    else:
        color = "Yellow"

    return {
        "Value": format_value_with_unit(current_val, unit),
        "Change": change_str,
        "Change Color": change_color,
        "5Y Mean": format_value_with_unit(mean_5y, unit),
        "5Y Std": f"{std_5y:.2f}",
        "Z-Score": f"{z_score:.2f}",
        "Percentile": f"{percentile:.0f}%",
        "Total Factor Score": round(factor_score, 1),
        "Colour Indicator": color
    }

def compute_rolling_scores(historical_data, n_days=50):
    """
    Compute a daily Gold Score time series over the last n_days using cluster-aware scoring.
    For each day t in the window, we treat each factor's value at day t as
    'current' and compute its z-score against the full 5Y history up to day t.
    Factors are averaged within clusters before summing to prevent redundancy.
    Returns a pd.Series of normalized gold scores indexed by date.
    """
    end_date = datetime.now()
    start_window = end_date - timedelta(days=n_days + 10)  # slight buffer

    # Get the common date range from GC=F (most liquid series)
    gc = historical_data.get("GC=F")
    if gc is None or len(gc) < n_days:
        return pd.Series(dtype=float)

    date_range = gc.index[gc.index >= pd.Timestamp(start_window)]
    if len(date_range) < 5:
        return pd.Series(dtype=float)

    # Sample every 2nd trading day to keep compute time reasonable (~45 points for 90 days)
    sampled_dates = date_range[::2]
    if len(sampled_dates) < 3:
        sampled_dates = date_range

    # Pre-compute cluster max scores for normalization
    cluster_max = {}
    for _cat, indicators in FACTOR_CONFIG.items():
        for item in indicators:
            cg = item.get('cluster_group', item['ticker'])
            w3 = item['weight'] * 3
            if cg not in cluster_max or w3 > cluster_max[cg]:
                cluster_max[cg] = w3
    max_possible_score = sum(cluster_max.values()) if cluster_max else 1.0

    daily_scores = {}
    for d in sampled_dates:
        # Collect factor scores grouped by cluster
        cluster_factor_scores = {}
        for _cat, indicators in FACTOR_CONFIG.items():
            for item in indicators:
                ticker = item['ticker']
                series = historical_data.get(ticker)
                if series is None or len(series) < 10:
                    continue
                # Use data up to day d
                s_up_to = series[series.index <= d]
                if len(s_up_to) < 10:
                    continue
                current_val = float(s_up_to.iloc[-1])
                mean_val = float(s_up_to.mean())
                std_val = float(s_up_to.std())
                if std_val == 0:
                    continue
                z = (current_val - mean_val) / std_val
                weight = item['weight']
                raw = z * weight if item['higher_is_bullish'] else -z * weight
                factor_score = max(min(raw, weight * 3), -weight * 3)

                cg = item.get('cluster_group', ticker)
                if cg not in cluster_factor_scores:
                    cluster_factor_scores[cg] = []
                cluster_factor_scores[cg].append(factor_score)

        # Average within clusters, then sum
        day_total = sum(
            np.mean(scores) for scores in cluster_factor_scores.values()
        )
        if max_possible_score > 0:
            daily_scores[d] = max(min(day_total / max_possible_score, 1.0), -1.0)

    if not daily_scores:
        return pd.Series(dtype=float)
    return pd.Series(daily_scores).sort_index()


def fetch_market_headlines():
    """
    Fetch recent financial headlines relevant to gold/precious metals from
    free news sources. Returns a list of dicts with 'date', 'title', 'source', 'url'.
    Uses the EODHD / Finviz / or a simple RSS approach.
    Falls back to identifying significant factor changes as 'headlines'.
    """
    headlines = []

    # Approach 1: Try Google News RSS for gold-related headlines
    try:
        rss_url = "https://news.google.com/rss/search?q=gold+price+OR+federal+reserve+OR+inflation+OR+precious+metals&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            items = root.findall('.//item')
            for item in items[:30]:  # last 30 headlines
                title_el = item.find('title')
                pub_el = item.find('pubDate')
                source_el = item.find('source')
                link_el = item.find('link')
                if title_el is not None and pub_el is not None:
                    try:
                        pub_date = pd.Timestamp(pub_el.text)
                        # Strip timezone to avoid tz-naive vs tz-aware comparison errors
                        if pub_date.tzinfo is not None:
                            pub_date = pub_date.tz_localize(None)
                        headlines.append({
                            'date': pub_date.strftime('%Y-%m-%d'),
                            'datetime': pub_date,
                            'title': title_el.text or '',
                            'source': source_el.text if source_el is not None else 'Google News',
                            'url': link_el.text if link_el is not None else ''
                        })
                    except Exception:
                        continue
    except Exception:
        pass

    # Approach 2: Add FOMC meeting dates as structural headlines
    fomc_dates_2024_2026 = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"
    ]
    now = datetime.now()
    for d_str in fomc_dates_2024_2026:
        d = pd.Timestamp(d_str)
        if (now - timedelta(days=90)) <= d <= (now + timedelta(days=30)):
            label = "FOMC Rate Decision" if d <= pd.Timestamp(now) else "FOMC Meeting (Upcoming)"
            headlines.append({
                'date': d_str,
                'datetime': d,
                'title': label,
                'source': 'Federal Reserve',
                'url': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
                'upcoming': d > pd.Timestamp(now)
            })

    # Approach 3: Fetch upcoming macro indicator releases from FRED API
    macro_releases = {
        10:  ("CPI Release", "Bureau of Labor Statistics"),
        50:  ("Employment Situation (NFP)", "Bureau of Labor Statistics"),
        53:  ("GDP Report", "Bureau of Economic Analysis"),
        54:  ("PCE / Personal Income", "Bureau of Economic Analysis"),
        124: ("PPI Release", "Bureau of Labor Statistics"),
        46:  ("Durable Goods Orders", "U.S. Census Bureau"),
        27:  ("Retail Sales", "U.S. Census Bureau"),
        18:  ("Industrial Production", "Federal Reserve"),
        21:  ("FOMC Minutes", "Federal Reserve"),
        82:  ("Treasury Budget", "U.S. Treasury"),
        19:  ("Consumer Confidence", "Conference Board"),
        29:  ("ISM Manufacturing PMI", "ISM"),
    }
    today_str = datetime.now().strftime('%Y-%m-%d')
    future_str = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
    past_str = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    for release_id, (label, source) in macro_releases.items():
        try:
            url = (
                f"https://api.stlouisfed.org/fred/release/dates"
                f"?release_id={release_id}"
                f"&api_key={FRED_API_KEY}"
                f"&file_type=json"
                f"&include_release_dates_with_no_data=true"
                f"&sort_order=asc"
            )
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                dates_data = resp.json().get("release_dates", [])
                for entry in dates_data:
                    rd = entry.get("date", "")
                    if past_str <= rd <= future_str:
                        is_upcoming = rd >= today_str
                        tag = " (Upcoming)" if is_upcoming else ""
                        headlines.append({
                            'date': rd,
                            'datetime': pd.Timestamp(rd),
                            'title': f"{label}{tag}",
                            'source': source,
                            'url': '',
                            'upcoming': is_upcoming
                        })
        except Exception:
            continue

    # Sort by date descending
    headlines.sort(key=lambda x: x.get('datetime', pd.Timestamp('2000-01-01')), reverse=True)
    return headlines


# ─── Dashboard Execution ─────────────────────────────────────────────────────

st.title("📊 Market Intelligence: Gold")
st.markdown(f"**Report Timestamp:** {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

with st.spinner("Downloading 5 years of API market history & computing live Z-Scores for all 47 factors..."):
    historical_data = fetch_historical_data()

    # Show data quality info
    fetch_errors = historical_data.get("_fetch_errors", [])
    simulated_count = historical_data.get("_simulated_count", 0)

    report_data = []

    for category, indicators in FACTOR_CONFIG.items():
        for item in indicators:
            ticker = item['ticker']
            series = historical_data.get(ticker)

            stats = calculate_statistics(series, item)

            # Determine actual data source status
            is_computed_live = ticker in historical_data and not (
                item['source'] == 'SIMULATED' and ticker not in [
                    "FEDFUNDS_REAL", "AU_M2", "SI_BETA"
                ]
            )
            if item['source'] == 'SIMULATED' and not is_computed_live:
                source_label = f"{ticker} (Simulated)"
            elif series is not None and len(series) >= 10:
                source_label = f"{ticker} ({item['source']})"
            else:
                source_label = f"{ticker} (No Data)"

            # Generate 50-day sparkline SVG
            sparkline_svg = make_sparkline_svg(series, n_days=50)

            entry = {
                "Category": category,
                "Indicator": item['ind'],
                "Cluster Group": item.get('cluster_group', ''),
                "Definition": item.get('definition', ''),
                "Ticker / Source": source_label,
                "Value": stats["Value"],
                "Change": stats.get("Change", "N/A"),
                "Change Color": stats.get("Change Color", "gray"),
                "Sparkline": sparkline_svg,
                "Colour Indicator": stats["Colour Indicator"],
                "Total Factor Score": stats["Total Factor Score"],
                "Z-Score": stats.get("Z-Score", "N/A"),
                "5Y Mean": stats.get("5Y Mean", "N/A"),
                "Percentile": stats.get("Percentile", "N/A"),
                "Why It Matters": item['why'],
                "What to Monitor (Signal)": "Mean Reversion / Z-Score extremes"
            }
            report_data.append(entry)

    final_df = pd.DataFrame(report_data)

    # Compute rolling Gold Scores for sparkline and chart
    rolling_scores = compute_rolling_scores(historical_data, n_days=90)

    # Fetch market headlines
    headlines = fetch_market_headlines()

# ─── Overall Confidence & Scoring Calculations (Cluster-Aware) ───────────────

total_factors = len(final_df)
decisive_signals = len(final_df[final_df['Colour Indicator'].isin(['Green', 'Red'])])
confidence_pct = (decisive_signals / total_factors) * 100 if total_factors > 0 else 0

# Cluster-aware scoring: average factor scores within each cluster, then sum clusters.
# This prevents correlated factors (e.g., 4 real-rate measures) from dominating the total.
if 'Cluster Group' in final_df.columns and final_df['Cluster Group'].notna().any():
    cluster_scores = final_df.groupby('Cluster Group')['Total Factor Score'].mean()
    overall_score = float(cluster_scores.sum())

    # Max possible per cluster = max(weight*3) of factors in that cluster
    # (since we average within cluster, the max avg approaches the heaviest factor's max)
    cluster_max = {}
    for _, row in final_df.iterrows():
        cg = row.get('Cluster Group', '')
        w = 0
        # Look up the weight from FACTOR_CONFIG
        for indicators in FACTOR_CONFIG.values():
            for item in indicators:
                if item['ind'] == row['Indicator']:
                    w = item['weight']
                    break
        if cg not in cluster_max or w * 3 > cluster_max[cg]:
            cluster_max[cg] = w * 3
    max_possible_score = sum(cluster_max.values()) if cluster_max else 1.0
    num_clusters = len(cluster_scores)
else:
    # Fallback: simple sum (no cluster info)
    overall_score = final_df['Total Factor Score'].sum()
    max_possible_score = sum(
        item['weight'] * 3
        for indicators in FACTOR_CONFIG.values()
        for item in indicators
    )
    num_clusters = total_factors

# Normalized Gold Score: actual / max, bounded [-1.0, +1.0]
gold_score = overall_score / max_possible_score if max_possible_score > 0 else 0.0
gold_score = max(min(gold_score, 1.0), -1.0)

# Interpretation scale
if gold_score >= 0.5:
    interpretation = "Strong Bullish"
    interp_color = "#22c55e"
    interp_emoji = "🟢🟢"
elif gold_score >= 0.1:
    interpretation = "Moderately Bullish"
    interp_color = "#86efac"
    interp_emoji = "🟢"
elif gold_score > -0.1:
    interpretation = "Neutral"
    interp_color = "#facc15"
    interp_emoji = "🟡"
elif gold_score > -0.5:
    interpretation = "Moderately Bearish"
    interp_color = "#fca5a5"
    interp_emoji = "🔴"
else:
    interpretation = "Strong Bearish"
    interp_color = "#ef4444"
    interp_emoji = "🔴🔴"

# ─── Summary Score Panel ─────────────────────────────────────────────────────

# Gold Score change (from rolling_scores)
if len(rolling_scores) >= 2:
    score_prev = float(rolling_scores.iloc[-2])
    score_change = gold_score - score_prev
    if score_change > 0.0005:
        score_chg_str = f"▲ +{score_change:.3f}"
        score_chg_color = "#22c55e"
    elif score_change < -0.0005:
        score_chg_str = f"▼ {score_change:.3f}"
        score_chg_color = "#ef4444"
    else:
        score_chg_str = "— 0.000"
        score_chg_color = "#888"
else:
    score_chg_str = "N/A"
    score_chg_color = "#888"

# Gauge bar: map gold_score from [-1, 1] to [0%, 100%] for the fill
gauge_pct = (gold_score + 1.0) / 2.0 * 100  # -1 → 0%, 0 → 50%, +1 → 100%

# Generate score history sparkline (wider than indicator sparklines)
score_sparkline_svg = make_sparkline_svg(rolling_scores, n_days=90, width=180, height=40)

_summary_html = (
    '<div style="background:linear-gradient(135deg,#0f0f23 0%,#1a1a2e 100%);'
    'border:1px solid #333;border-radius:12px;padding:24px 32px;margin-bottom:20px;">'
    '<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:20px;">'
    # Big score + sparkline
    '<div style="flex:0 0 auto;text-align:center;min-width:200px;">'
    '<div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;">Gold Score</div>'
    f'<div style="font-size:3.2rem;font-weight:800;color:{interp_color};line-height:1.1;'
    f'font-family:\'SF Mono\',\'Fira Code\',monospace;">{gold_score:+.2f}</div>'
    f'<div style="font-size:1.05rem;color:{interp_color};font-weight:600;margin-top:4px;">'
    f'{interp_emoji} {interpretation}</div>'
    f'<div style="font-size:0.9rem;color:{score_chg_color};font-weight:600;margin-top:6px;'
    f'font-family:\'SF Mono\',\'Fira Code\',monospace;">{score_chg_str}</div>'
    '<div style="margin-top:8px;">'
    '<div style="font-size:0.65rem;color:#666;margin-bottom:2px;">90-Day Score Trend</div>'
    f'{score_sparkline_svg}'
    '</div>'
    '</div>'
    # Score breakdown
    '<div style="flex:1 1 280px;min-width:250px;">'
    '<div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Score Breakdown</div>'
    '<table style="font-size:0.85rem;color:#d0d0d0;border-collapse:collapse;width:100%;">'
    f'<tr><td style="padding:3px 0;color:#888;">Total Weighted Score</td>'
    f'<td style="padding:3px 0;text-align:right;font-weight:600;font-family:monospace;">{overall_score:+.1f}</td></tr>'
    f'<tr><td style="padding:3px 0;color:#888;">Max Possible Bullish</td>'
    f'<td style="padding:3px 0;text-align:right;font-family:monospace;">{max_possible_score:.0f}</td></tr>'
    f'<tr><td style="padding:3px 0;color:#888;">Max Possible Bearish</td>'
    f'<td style="padding:3px 0;text-align:right;font-family:monospace;">-{max_possible_score:.0f}</td></tr>'
    f'<tr style="border-top:1px solid #333;">'
    f'<td style="padding:6px 0 3px 0;color:#888;">Normalized</td>'
    f'<td style="padding:6px 0 3px 0;text-align:right;font-weight:700;color:{interp_color};font-family:monospace;">{gold_score:+.2f}</td></tr>'
    f'<tr><td style="padding:3px 0;color:#888;">Last Period Change</td>'
    f'<td style="padding:3px 0;text-align:right;font-weight:600;color:{score_chg_color};font-family:monospace;">{score_chg_str}</td></tr>'
    '</table></div>'
    # Interpretation scale
    '<div style="flex:0 0 auto;min-width:220px;">'
    '<div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Interpretation Scale</div>'
    '<table style="font-size:0.78rem;color:#aaa;border-collapse:collapse;">'
    '<tr><td style="padding:2px 10px 2px 0;color:#ef4444;font-family:monospace;">-1.0 to -0.5</td><td>Strong Bearish</td></tr>'
    '<tr><td style="padding:2px 10px 2px 0;color:#fca5a5;font-family:monospace;">-0.5 to -0.1</td><td>Bearish</td></tr>'
    '<tr><td style="padding:2px 10px 2px 0;color:#facc15;font-family:monospace;">-0.1 to +0.1</td><td>Neutral</td></tr>'
    '<tr><td style="padding:2px 10px 2px 0;color:#86efac;font-family:monospace;">+0.1 to +0.5</td><td>Bullish</td></tr>'
    '<tr><td style="padding:2px 10px 2px 0;color:#22c55e;font-family:monospace;">+0.5 to +1.0</td><td>Strong Bullish</td></tr>'
    '</table>'
    f'<div style="margin-top:12px;font-size:0.75rem;color:#666;">'
    f'{total_factors} factors &middot; {num_clusters} clusters &middot; {total_factors - simulated_count} live &middot; {simulated_count} simulated</div>'
    '</div>'
    '</div>'
    # Gauge bar
    '<div style="margin-top:18px;">'
    '<div style="position:relative;height:18px;background:linear-gradient(to right,'
    '#ef4444 0%,#fca5a5 25%,#facc15 45%,#facc15 55%,#86efac 75%,#22c55e 100%);'
    'border-radius:9px;overflow:visible;">'
    f'<div style="position:absolute;left:{gauge_pct:.1f}%;top:-3px;'
    'transform:translateX(-50%);width:4px;height:24px;'
    'background:white;border-radius:2px;box-shadow:0 0 6px rgba(255,255,255,0.6);"></div>'
    '<div style="position:absolute;left:50%;top:-1px;transform:translateX(-50%);'
    'width:1px;height:20px;background:rgba(0,0,0,0.4);"></div>'
    '</div>'
    '<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#666;margin-top:3px;">'
    '<span>-1.0 Strong Bearish</span><span>0.0 Neutral</span><span>+1.0 Strong Bullish</span>'
    '</div></div>'
    '</div>'
)
st.markdown(_summary_html, unsafe_allow_html=True)

# Confidence & error details
col1, col2 = st.columns(2)
col1.metric("Signal Confidence", f"{confidence_pct:.1f}%", help="Percentage of factors producing a decisive (non-neutral) signal")
col2.metric("Decisive Signals", f"{decisive_signals} / {total_factors}", help="Factors with Green or Red signals vs total")

# Clamp progress to [0.0, 1.0]
st.progress(min(max(confidence_pct / 100.0, 0.0), 1.0))
st.divider()

# Show fetch errors in an expander if any
if fetch_errors:
    with st.expander(f"⚠️ Data Fetch Issues ({len(fetch_errors)})", expanded=False):
        for err in fetch_errors:
            st.text(f"  • {err}")

# ─── Spot Prices Panel ──────────────────────────────────────────────────────
st.subheader("Spot Precious Metals Prices")

spot_metals = [
    {"name": "Gold", "ticker": "GC=F", "icon": "🥇"},
    {"name": "Silver", "ticker": "SI=F", "icon": "🥈"},
    {"name": "Platinum", "ticker": "PL=F", "icon": "🏆"},
]

_spot_rows = ""
for metal in spot_metals:
    series = historical_data.get(metal["ticker"])
    if series is not None and len(series) > 1:
        current = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) > 1 else current
        change = current - prev
        change_pct = (change / prev * 100) if prev != 0 else 0
        sparkline = make_sparkline_svg(series, n_days=50, width=160, height=32)

        chg_color = "#22c55e" if change >= 0 else "#ef4444"
        chg_arrow = "▲" if change >= 0 else "▼"
    else:
        current = 0
        change = 0
        change_pct = 0
        sparkline = f'<svg width="160" height="32"></svg>'
        chg_color = "#888"
        chg_arrow = "—"

    _spot_rows += (
        '<div style="flex:1;min-width:240px;background:#1a1a2e;border:1px solid #333;'
        'border-radius:10px;padding:16px 20px;text-align:center;">'
        f'<div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;">'
        f'{metal["icon"]} {metal["name"]}</div>'
        f'<div style="font-size:2rem;font-weight:700;color:#e0e0e0;'
        f'font-family:\'SF Mono\',\'Fira Code\',monospace;margin:4px 0;">'
        f'${current:,.2f}</div>'
        f'<div style="font-size:0.9rem;color:{chg_color};font-weight:600;">'
        f'{chg_arrow} ${abs(change):,.2f} ({change_pct:+.2f}%)</div>'
        f'<div style="margin-top:8px;display:flex;justify-content:center;">'
        f'<div style="display:inline-block;">'
        f'<div style="font-size:0.6rem;color:#666;margin-bottom:2px;">50-Day Trend</div>'
        f'{sparkline}</div></div>'
        '</div>'
    )

_spot_html = (
    '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;">'
    f'{_spot_rows}'
    '</div>'
)
st.markdown(_spot_html, unsafe_allow_html=True)

# ─── Interactive Chart + Headlines (fully client-side JS) ────────────────────
st.subheader("Gold Price vs Gold Score — Click timeline to filter headlines")

gc_series = historical_data.get("GC=F")
if gc_series is not None and len(rolling_scores) > 3:
    import plotly.io as pio
    import html as html_mod
    import streamlit.components.v1 as components

    # Align gold price to the rolling score date range
    score_start = rolling_scores.index.min()
    gc_chart = gc_series[gc_series.index >= score_start]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=gc_chart.index, y=gc_chart.values,
            name="Gold Price ($/oz)", mode="lines",
            line=dict(color="#FFD700", width=2),
            hovertemplate="<b>Gold Price</b><br>%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_scores.index, y=rolling_scores.values,
            name="Gold Score", mode="lines",
            line=dict(color="#60a5fa", width=2, dash="dot"),
            hovertemplate="<b>Gold Score</b><br>%{x|%b %d, %Y}<br>%{y:+.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0, secondary_y=True, line_dash="dash",
                  line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f0f23",
        plot_bgcolor="#0e0e1a",
        font=dict(color="#d0d0d0"),
        height=480,
        margin=dict(l=55, r=55, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(15,15,35,0.9)", font_size=12, font_color="#d0d0d0"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True,
                     tickfont=dict(color="#aaa"),
                     showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikecolor="rgba(255,255,255,0.3)",
                     spikedash="dot")
    fig.update_yaxes(title_text="Gold $/oz", gridcolor="rgba(255,255,255,0.05)",
                     showgrid=True, secondary_y=False, tickformat="$,.0f",
                     title_font=dict(color="#FFD700", size=11),
                     tickfont=dict(color="#ccc"))
    fig.update_yaxes(title_text="Score", secondary_y=True, range=[-1.05, 1.05],
                     tickformat="+.2f", title_font=dict(color="#60a5fa", size=11),
                     gridcolor="rgba(255,255,255,0.03)",
                     tickfont=dict(color="#ccc"))

    chart_div_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                                  div_id="goldChart", config={"displayModeBar": False})

    headlines_json = json.dumps([
        {"date": h["date"],
         "title": html_mod.escape(h.get("title", ""), quote=True),
         "source": html_mod.escape(h.get("source", ""), quote=True),
         "url": h.get("url", ""),
         "upcoming": h.get("upcoming", False)}
        for h in headlines
    ])

    interactive_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #0f0f23; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

  .layout {
    display: flex;
    gap: 14px;
    width: 100%;
  }
  /* Left: headlines panel */
  .news-panel {
    flex: 0 0 300px;
    min-width: 280px;
    background: #12121f;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px 16px;
    max-height: 540px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .news-panel::-webkit-scrollbar { width: 5px; }
  .news-panel::-webkit-scrollbar-track { background: #1a1a2e; border-radius: 5px; }
  .news-panel::-webkit-scrollbar-thumb { background: #444; border-radius: 5px; }

  .news-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    flex-shrink: 0;
  }
  .news-title { font-size: 0.88rem; font-weight: 700; color: #e0e0e0; }
  .clear-btn {
    background: #2a2a3a;
    color: #facc15;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.68rem;
    cursor: pointer;
    display: none;
    transition: background 0.15s;
  }
  .clear-btn:hover { background: #3a3a4a; }
  .date-label {
    font-size: 0.76rem;
    color: #facc15;
    margin-bottom: 8px;
    display: none;
    flex-shrink: 0;
  }
  .hint {
    text-align: center;
    font-size: 0.7rem;
    color: #666;
    padding: 6px 0;
    flex-shrink: 0;
  }
  .news-list { flex: 1; overflow-y: auto; }

  .day-header {
    font-size: 0.76rem;
    font-weight: 700;
    color: #60a5fa;
    margin-top: 10px;
    margin-bottom: 4px;
    border-bottom: 1px solid #2a2a3a;
    padding-bottom: 3px;
  }
  .headline-item {
    font-size: 0.72rem;
    color: #d0d0d0;
    margin: 3px 0;
    line-height: 1.4;
  }
  .headline-item a {
    color: #d0d0d0;
    text-decoration: none;
    transition: color 0.15s;
  }
  .headline-item a:hover { color: #60a5fa; }
  .headline-src { color: #666; }
  .headline-upcoming {
    color: #fbbf24;
    font-style: italic;
  }
  .headline-upcoming::before {
    content: "📅 ";
  }
  .upcoming-tag {
    font-size: 0.55rem;
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 4px;
    vertical-align: middle;
  }
  .news-footer {
    font-size: 0.58rem;
    color: #444;
    margin-top: 10px;
    border-top: 1px solid #2a2a3a;
    padding-top: 5px;
    flex-shrink: 0;
  }

  /* Right: chart */
  .chart-panel {
    flex: 1;
    min-width: 0;
    position: relative;
  }
  .chart-panel .js-plotly-plot { cursor: crosshair !important; }

  /* Responsive: stack on narrow screens */
  @media (max-width: 800px) {
    .layout { flex-direction: column-reverse; }
    .news-panel { flex: none; max-height: 320px; }
  }
</style>
</head>
<body>
<div class="layout">
  <!-- LEFT: Headlines -->
  <div class="news-panel">
    <div class="news-header">
      <div class="news-title">&#x1F4F0; Headlines & Events</div>
      <button class="clear-btn" id="clearBtn" onclick="clearFilter()">&#x2715; Reset</button>
    </div>
    <div class="date-label" id="dateLabel"></div>
    <div class="hint" id="hintText">Click on the chart to filter by date</div>
    <div class="news-list" id="newsList"></div>
    <div class="news-footer">Sources: Google News RSS &middot; FRED Release Calendar &middot; Federal Reserve</div>
  </div>
  <!-- RIGHT: Chart -->
  <div class="chart-panel">
    """ + chart_div_html + """
  </div>
</div>

<script>
var ALL_HEADLINES = """ + headlines_json + """;

function renderHeadlines(clickedDate) {
    var list = document.getElementById("newsList");
    var lbl  = document.getElementById("dateLabel");
    var btn  = document.getElementById("clearBtn");
    var hint = document.getElementById("hintText");
    if (!list) return;

    var filtered;
    if (clickedDate) {
        var fd = clickedDate.split("T")[0];
        // Show headlines on the clicked day and PRIOR
        filtered = ALL_HEADLINES.filter(function(h){ return h.date <= fd; });
        lbl.innerHTML = "&#x1F4CC; Headlines up to <b>" + fd + "</b>";
        lbl.style.display = "block";
        btn.style.display = "inline-block";
        hint.style.display = "none";
    } else {
        filtered = ALL_HEADLINES;
        lbl.style.display = "none";
        btn.style.display = "none";
        hint.style.display = "block";
    }

    // Separate upcoming from past, then group by date
    var upcoming = filtered.filter(function(h){ return h.upcoming; });
    var past = filtered.filter(function(h){ return !h.upcoming; });

    // Group by date
    var groups = {};
    var dateOrder = [];
    // Upcoming dates first (ascending — soonest first)
    upcoming.sort(function(a,b){ return a.date.localeCompare(b.date); });
    upcoming.forEach(function(h){
        if (!groups[h.date]) { groups[h.date] = []; dateOrder.push(h.date); }
        groups[h.date].push(h);
    });
    // Then past dates (descending — most recent first)
    past.forEach(function(h){
        if (!groups[h.date]) { groups[h.date] = []; dateOrder.push(h.date); }
        groups[h.date].push(h);
    });
    // dateOrder already has upcoming (asc) then past (desc) — no re-sort needed

    var out = "";
    var shown = 0;
    for (var i = 0; i < dateOrder.length && shown < 20; i++) {
        var d = dateOrder[i];
        var items = groups[d];
        var dt = new Date(d + "T12:00:00");
        var nice = dt.toLocaleDateString("en-US", {year:"numeric",month:"short",day:"numeric",weekday:"short"});
        var isUpcomingDay = items.some(function(h){ return h.upcoming; });

        if (isUpcomingDay) {
            out += '<div class="day-header" style="color:#fbbf24;">' + nice + '</div>';
        } else {
            out += '<div class="day-header">' + nice + '</div>';
        }
        var mx = Math.min(items.length, 5);
        for (var j = 0; j < mx; j++) {
            var h = items[j];
            var src = h.source ? ' <span class="headline-src">&mdash; ' + h.source + '</span>' : "";
            var tag = h.upcoming ? '<span class="upcoming-tag">UPCOMING</span>' : "";
            var cls = h.upcoming ? "headline-item headline-upcoming" : "headline-item";
            if (h.url) {
                out += '<div class="' + cls + '">&bull; <a href="' + h.url + '" target="_blank">' + h.title + '</a>' + tag + src + '</div>';
            } else {
                out += '<div class="' + cls + '">&bull; ' + h.title + tag + src + '</div>';
            }
        }
        shown++;
    }
    if (!out) {
        out = '<div style="color:#666;padding:12px 0;font-size:0.76rem;">No headlines for this period.</div>';
    }
    list.innerHTML = out;
    // Scroll to top of list
    list.scrollTop = 0;
}

function clearFilter() {
    renderHeadlines(null);
}

// Initial render
renderHeadlines(null);

// Attach plotly_click — poll until the chart is ready, then listen continuously
(function() {
    var ready = false;
    function tryAttach() {
        var gd = document.getElementById("goldChart");
        if (!gd || !gd._fullLayout) {
            setTimeout(tryAttach, 150);
            return;
        }
        if (ready) return;
        ready = true;

        // Use Plotly's own event system which fires on every click
        gd.on("plotly_click", function(evtData) {
            if (evtData && evtData.points && evtData.points.length > 0) {
                renderHeadlines(String(evtData.points[0].x));
            }
        });

        // Hover dots — draw SVG circles on each trace at the hovered x position
        var dotColors = ["#FFD700", "#60a5fa"];

        gd.on("plotly_hover", function(evtData) {
            // Remove previous dots
            gd.querySelectorAll(".hover-dot").forEach(function(el){ el.remove(); });
            if (!evtData || !evtData.points) return;
            var svg = gd.querySelector(".main-svg");
            if (!svg) return;

            evtData.points.forEach(function(pt) {
                var xa = gd._fullLayout.xaxis;
                var yaKey = pt.fullData.yaxis === "y2" ? "yaxis2" : "yaxis";
                var ya = gd._fullLayout[yaKey];
                if (!xa || !ya) return;

                var px = xa.l2p(xa.d2c(pt.x)) + xa._offset;
                var py = ya.l2p(pt.y) + ya._offset;

                var dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                dot.setAttribute("cx", px);
                dot.setAttribute("cy", py);
                dot.setAttribute("r", 5);
                dot.setAttribute("fill", dotColors[pt.traceIndex] || "#fff");
                dot.setAttribute("stroke", "#fff");
                dot.setAttribute("stroke-width", "1.5");
                dot.setAttribute("class", "hover-dot");
                dot.setAttribute("pointer-events", "none");
                svg.appendChild(dot);
            });
        });

        gd.on("plotly_unhover", function() {
            gd.querySelectorAll(".hover-dot").forEach(function(el){ el.remove(); });
        });
    }
    tryAttach();
})();
</script>
</body>
</html>
"""

    components.html(interactive_html, height=580, scrolling=False)

else:
    st.info("Insufficient data to render dual-axis chart.")

st.divider()

# ─── CSS for hover tooltips ──────────────────────────────────────────────────
st.markdown("""
<style>
.factor-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}
.factor-table th {
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #333;
    position: sticky;
    top: 0;
}
.factor-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #2a2a3a;
    vertical-align: top;
}
.factor-table tr:hover {
    background: #1e1e30;
}
.factor-table tr:hover td {
    color: #ffffff !important;
}
/* Indicator cell with tooltip */
.ind-cell {
    position: relative;
    cursor: help;
}
.ind-name {
    border-bottom: 1px dotted #888;
    font-weight: 500;
}
.ind-cell .tooltip-box {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    z-index: 1000;
    left: 0;
    top: 100%;
    width: 380px;
    max-width: 90vw;
    background: #1a1a2e;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 12px 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    transition: opacity 0.15s ease-in-out, visibility 0.15s;
    pointer-events: none;
}
.ind-cell:hover .tooltip-box {
    visibility: visible;
    opacity: 1;
}
.tooltip-box .tt-title {
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 6px;
    font-size: 0.85rem;
}
.tooltip-box .tt-def {
    color: #d0d0d0;
    font-size: 0.8rem;
    line-height: 1.45;
    margin-bottom: 8px;
}
.tooltip-box .tt-why {
    color: #facc15;
    font-size: 0.78rem;
    font-style: italic;
}
/* Signal colors */
.sig-green { color: #22c55e; font-weight: 600; }
.sig-red { color: #ef4444; font-weight: 600; }
.sig-yellow { color: #facc15; font-weight: 600; }
.score-pos { color: #22c55e; font-weight: 600; }
.score-neg { color: #ef4444; font-weight: 600; }
.score-zero { color: #888; }
.val-cell { font-weight: 600; font-family: 'SF Mono', 'Fira Code', monospace; }
.chg-cell { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; font-weight: 600; white-space: nowrap; }
.chg-green { color: #22c55e; }
.chg-red { color: #ef4444; }
.chg-gray { color: #888; }
.spark-cell { padding: 4px 6px !important; vertical-align: middle; min-width: 130px; }
.spark-cell svg { display: block; }
</style>
""", unsafe_allow_html=True)


def render_category_table(cat_df):
    """Render a category table as HTML with hover tooltips on indicator names and sparklines."""
    cols = ["Indicator", "Ticker / Source", "Value", "Change", "Sparkline", "5Y Mean", "Z-Score",
            "Percentile", "Colour Indicator", "Total Factor Score"]

    header_labels = {
        "Colour Indicator": "Signal",
        "Total Factor Score": "Score",
        "Sparkline": "50D Trend",
        "Change": "Chg"
    }

    html = '<table class="factor-table">'
    html += '<tr>'
    for c in cols:
        label = header_labels.get(c, c)
        html += f'<th>{label}</th>'
    html += '</tr>'

    for _, row in cat_df.iterrows():
        ind_name = row["Indicator"]
        cluster = row.get("Cluster Group", "")
        definition = row.get("Definition", "")
        why = row.get("Why It Matters", "")

        # Escape HTML entities in text
        def esc(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

        html += '<tr>'
        for c in cols:
            val = row.get(c, "")

            if c == "Indicator":
                # Indicator cell with tooltip and cluster label
                html += '<td class="ind-cell">'
                html += f'<span class="ind-name">{esc(ind_name)}</span>'
                if cluster:
                    html += f'<div style="font-size:0.65rem;color:#666;margin-top:1px;">{esc(cluster)}</div>'
                if definition:
                    html += '<div class="tooltip-box">'
                    html += f'<div class="tt-title">{esc(ind_name)}</div>'
                    html += f'<div class="tt-def">{esc(definition)}</div>'
                    if why:
                        html += f'<div class="tt-why">Why it matters: {esc(why)}</div>'
                    html += '</div>'
                html += '</td>'

            elif c == "Change":
                chg_color = row.get("Change Color", "gray")
                css_class = f"chg-cell chg-{chg_color}"
                html += f'<td class="{css_class}">{esc(str(val))}</td>'

            elif c == "Sparkline":
                # Insert the pre-built SVG sparkline directly (trusted HTML)
                html += f'<td class="spark-cell">{val}</td>'

            elif c == "Colour Indicator":
                css = {"Green": "sig-green", "Red": "sig-red", "Yellow": "sig-yellow"}.get(val, "")
                label = {"Green": "▲ Bullish", "Red": "▼ Bearish", "Yellow": "● Neutral"}.get(val, val)
                html += f'<td class="{css}">{label}</td>'

            elif c == "Total Factor Score":
                fval = float(val) if val != "N/A" else 0
                css = "score-pos" if fval > 0 else ("score-neg" if fval < 0 else "score-zero")
                html += f'<td class="{css}">{fval:+.1f}</td>'

            elif c == "Value":
                html += f'<td class="val-cell">{esc(str(val))}</td>'

            else:
                html += f'<td>{esc(str(val))}</td>'

        html += '</tr>'

    html += '</table>'
    return html


for cat in final_df['Category'].unique():
    cat_df = final_df[final_df['Category'] == cat]

    subset_score = cat_df['Total Factor Score'].sum()
    c_color = "🟢" if subset_score > 0 else ("🔴" if subset_score < 0 else "🟡")

    st.subheader(f"{cat}  {c_color} (Subset Score: {subset_score:.1f})")
    st.markdown(render_category_table(cat_df), unsafe_allow_html=True)
