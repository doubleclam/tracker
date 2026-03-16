#!/usr/bin/env python3
"""
Market Intelligence: Gold - Full 47-Factor Dynamic Engine
Calculates 5Y Means, Z-Scores, and Percentiles dynamically from historical API data.
"""

import os
import io
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

# ─── Global CSS: Mobile-First + Dark Theme ───────────────────────────────────
# Forces dark color-scheme for iOS Safari (fixes color rendering on iPhone),
# adds viewport-safe sizing, and comprehensive mobile breakpoints.
st.markdown("""
<style>
    /* Force dark color scheme for iOS Safari / mobile browsers */
    :root {
        color-scheme: dark;
        -webkit-text-size-adjust: 100%;
    }
    html, body, .stApp {
        background-color: #0f0f23 !important;
        color: #d0d0d0 !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stApp {
        background-color: #0f0f23 !important;
    }

    /* Prevent horizontal overflow on mobile */
    .stMainBlockContainer, .block-container, .stVerticalBlock {
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    /* Mobile: tighten Streamlit's default padding */
    @media (max-width: 768px) {
        .stMainBlockContainer, .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 1rem !important;
        }
        /* Streamlit header / toolbar — reduce on mobile */
        header[data-testid="stHeader"] {
            padding: 0.25rem 0.5rem !important;
        }
        /* Subheaders smaller on mobile */
        .stSubheader, h2, h3 {
            font-size: 1.05rem !important;
            line-height: 1.3 !important;
        }
        /* Metrics compact on mobile */
        [data-testid="stMetric"] {
            padding: 0.3rem 0 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        /* Dividers thinner */
        hr {
            margin: 0.5rem 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─── LBMA Analyst Forecast Data ───────────────────────────────────────────────
# The LBMA Annual Precious Metals Forecast Survey is published each January.
# 28 named analysts from major banks and consultancies provide low/avg/high
# gold price forecasts for the year.  We try a live scrape first, then fall
# back to the hardcoded 2026 survey data (it only changes once per year).

def fetch_analyst_forecasts():
    """Return list of analyst gold price forecasts for the current survey year.
    Each entry: {analyst, institution, low, avg, high, date, source}.
    Combines LBMA survey (28 analysts, Jan 2026) with major Wall Street bank
    forecasts (updated Feb/Mar 2026). Sorted by avg ascending."""

    # ── LBMA 2026 Forecast Survey (published Jan 20, 2026) ───────────────
    # 28 named analysts — annual calendar-year average price forecasts.
    lbma = [
        {"analyst": "Robin Bhar",           "institution": "Robin Bhar Metals Consulting", "low": 3500, "avg": 4000, "high": 5000},
        {"analyst": "Bart Melek",           "institution": "TD Securities",               "low": 3920, "avg": 4213, "high": 4775},
        {"analyst": "Bernard Dahdah",       "institution": "Natixis",                     "low": 3700, "avg": 4250, "high": 4950},
        {"analyst": "Debajit Saha",         "institution": "Refinitiv Metals Research",   "low": 3700, "avg": 4269, "high": 5175},
        {"analyst": "Caroline Bain",        "institution": "Bain Commodities",            "low": 3500, "avg": 4299, "high": 4800},
        {"analyst": "Rhona O'Connell",      "institution": "StoneX Financial",            "low": 3650, "avg": 4380, "high": 4950},
        {"analyst": "Srivatsava Ganapathy", "institution": "Eventell Global Advisory",    "low": 3600, "avg": 4420, "high": 5250},
        {"analyst": "Christopher Louney",   "institution": "RBC Capital Markets",         "low": 3704, "avg": 4427, "high": 5108},
        {"analyst": "Michael Hsueh",        "institution": "Deutsche Bank",               "low": 3950, "avg": 4450, "high": 4950},
        {"analyst": "Rohit Savant",         "institution": "CPM Group",                   "low": 3975, "avg": 4460, "high": 5000},
        {"analyst": "Nicky Shiels",         "institution": "MKS PAMP SA",                 "low": 3750, "avg": 4500, "high": 5400},
        {"analyst": "James Steel",          "institution": "HSBC",                        "low": 3950, "avg": 4586, "high": 5050},
        {"analyst": "Renisha Chainani",     "institution": "Augmont",                     "low": 3900, "avg": 4600, "high": 5800},
        {"analyst": "Alexander Zumpfe",     "institution": "Heraeus Metals Germany",      "low": 3450, "avg": 4620, "high": 5200},
        {"analyst": "Frank Schallenberger", "institution": "LBBW",                        "low": 3809, "avg": 4621, "high": 4872},
        {"analyst": "Kirill Kirilenko",     "institution": "CRU International",           "low": 4200, "avg": 4650, "high": 5100},
        {"analyst": "Joni Teves",           "institution": "UBS",                         "low": 4150, "avg": 4675, "high": 5000},
        {"analyst": "Suki Cooper",          "institution": "Standard Chartered",          "low": 3700, "avg": 4788, "high": 5500},
        {"analyst": "Kieran Tompkins",      "institution": "Capital Economics",           "low": 4100, "avg": 4800, "high": 5100},
        {"analyst": "Grant Sporre",         "institution": "Bloomberg Intelligence",      "low": 4025, "avg": 4820, "high": 5280},
        {"analyst": "Nikos Kavalis",        "institution": "Metals Focus",                "low": 4300, "avg": 4850, "high": 5500},
        {"analyst": "Chantelle Schieven",   "institution": "Capitalight Research",        "low": 4240, "avg": 5072, "high": 5830},
        {"analyst": "Jacob Smith",          "institution": "Mitsubishi Corporation",      "low": 3900, "avg": 5100, "high": 5800},
        {"analyst": "Keisuke Okui",         "institution": "Sumitomo Corporation",        "low": 3500, "avg": 5300, "high": 6000},
        {"analyst": "Ross Norman",          "institution": "Metals Daily",                "low": 4350, "avg": 5375, "high": 6400},
        {"analyst": "Bruce Ikemizu",        "institution": "Japan Bullion Market Assoc",  "low": 4200, "avg": 5450, "high": 6200},
        {"analyst": "Rene Hochreiter",      "institution": "NOAH Capital Markets",        "low": 4352, "avg": 5750, "high": 6300},
        {"analyst": "Julia Du",             "institution": "ICBC Standard Bank",          "low": 4100, "avg": 6050, "high": 7150},
    ]
    for r in lbma:
        r["date"] = "Jan 2026"
        r["source"] = "LBMA"

    # ── Major Wall Street / Bank Forecasts (revised post-LBMA) ───────────
    # Year-end or 12-month targets; low/high from stated ranges or analyst
    # upside/downside scenarios.
    banks = [
        {"analyst": "Kenny Hu",              "institution": "Citi",              "low": 4500, "avg": 5000, "high": 5500,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Lina Thomas",           "institution": "Goldman Sachs",     "low": 4600, "avg": 5400, "high": 6000,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Commodities Desk",      "institution": "Morgan Stanley",    "low": 4800, "avg": 5700, "high": 6200,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Daniel Hynes",          "institution": "ANZ",               "low": 5000, "avg": 5800, "high": 6200,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Michael Widmer",        "institution": "Bank of America",   "low": 5000, "avg": 6000, "high": 6500,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Commodities Desk",      "institution": "BNP Paribas",       "low": 5000, "avg": 6000, "high": 6500,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Michael Hsueh",         "institution": "Deutsche Bank (Rev)", "low": 5000, "avg": 6000, "high": 6500, "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Joni Teves",            "institution": "UBS (Revised)",     "low": 4600, "avg": 6200, "high": 7200,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Investment Institute",  "institution": "Wells Fargo",       "low": 6100, "avg": 6200, "high": 6300,  "date": "Feb 2026", "source": "Bank"},
        {"analyst": "Gregory Shearer",       "institution": "J.P. Morgan",       "low": 5000, "avg": 6300, "high": 8500,  "date": "Feb 2026", "source": "Bank"},
    ]

    data = lbma + banks
    return sorted(data, key=lambda r: r["avg"]), 2026

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
            "TEDRATE", "RRPONTSYD", "TOTRESNS",
            # Additional series for computed indicators
            "FEDFUNDS", "SOFR", "DFF",
            "DGS10", "CPIAUCSL", "EXPINF1YR",
            "IRLTLT01DEM156N", "IRLTLT01JPM156N", "IRLTLT01GBM156N",
            "DEUCPIALLMINMEI", "JPNCPIALLMINMEI", "GBRCPIALLMINMEI",
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
            fedfunds = hist_data.get("FEDFUNDS")
            t5yie = hist_data.get("T5YIE")
            if fedfunds is not None and t5yie is not None and len(fedfunds) > 0:
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

    # 1B. Compute SOFR-OIS spread from already-fetched FRED data
    if fred:
        try:
            sofr = hist_data.get("SOFR")
            dff = hist_data.get("DFF")
            if sofr is not None and dff is not None and len(sofr) > 10 and len(dff) > 10:
                aligned_rates = pd.concat([sofr, dff], axis=1).dropna()
                aligned_rates.columns = ["SOFR", "DFF"]
                sofr_ois = (aligned_rates["SOFR"] - aligned_rates["DFF"]) * 100
                if len(sofr_ois) > 10:
                    hist_data["SOFR_OIS"] = sofr_ois
        except Exception as e:
            fetch_errors.append(f"SOFR-OIS calc: {e}")

    # 1B2. Compute additional indicators from already-fetched FRED data
    if fred:
        # Global Real Rates (GDP-weighted: US 45%, EU 25%, Japan 15%, UK 15%)
        try:
            us_nom = hist_data.get("DGS10")
            de_nom = hist_data.get("IRLTLT01DEM156N")
            jp_nom = hist_data.get("IRLTLT01JPM156N")
            gb_nom = hist_data.get("IRLTLT01GBM156N")
            us_cpi = hist_data.get("CPIAUCSL")

            if all(v is not None for v in [us_nom, de_nom, jp_nom, gb_nom, us_cpi]):
                us_infl = us_cpi.pct_change(12) * 100
                us_infl = us_infl.dropna()

                de_cpi = hist_data.get("DEUCPIALLMINMEI")
                jp_cpi = hist_data.get("JPNCPIALLMINMEI")
                gb_cpi = hist_data.get("GBRCPIALLMINMEI")

                if all(v is not None for v in [de_cpi, jp_cpi, gb_cpi]):
                    de_infl = de_cpi.pct_change(12) * 100
                    jp_infl = jp_cpi.pct_change(12) * 100
                    gb_infl = gb_cpi.pct_change(12) * 100

                    us_real = (us_nom - us_infl.reindex(us_nom.index, method="ffill")).dropna()
                    de_real = (de_nom - de_infl.reindex(de_nom.index, method="ffill")).dropna()
                    jp_real = (jp_nom - jp_infl.reindex(jp_nom.index, method="ffill")).dropna()
                    gb_real = (gb_nom - gb_infl.reindex(gb_nom.index, method="ffill")).dropna()

                    glb = pd.concat([us_real, de_real, jp_real, gb_real], axis=1).dropna()
                    glb.columns = ["US", "DE", "JP", "GB"]
                    glb_weighted = glb["US"] * 0.45 + glb["DE"] * 0.25 + glb["JP"] * 0.15 + glb["GB"] * 0.15
                    if len(glb_weighted) > 10:
                        hist_data["GLB_REAL"] = glb_weighted
        except Exception as e:
            fetch_errors.append(f"GLB_REAL calc: {e}")

        # Cleveland Fed Inflation Nowcast (1-Year Expected Inflation)
        clev = hist_data.get("EXPINF1YR")
        if clev is not None and len(clev) > 10:
            hist_data["CLEV_INFL"] = clev

        # Flag M2 data for Gold Allocation calculation after YF data
        hist_data["_AU_ALLOC_M2"] = hist_data.get("M2SL")

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
        "DX=F", "^VIX", "GC=F", "SI=F", "PL=F", "CL=F",
        "^GSPC", "BTC-USD", "VNQ", "GDX", "GLD", "GDXJ",
        "PA=F", "HG=F", "^SKEW",
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

        # Compute EFP Spread: COMEX futures minus spot proxy (GLD * conversion)
        # GLD tracks spot gold; converting to $/oz gives a spot proxy
        if "GC=F" in hist_data and "GLD" in hist_data:
            try:
                gc = hist_data["GC=F"]
                gld = hist_data["GLD"]
                conv = gc.iloc[0] / gld.iloc[0] if gld.iloc[0] != 0 else 10.0
                spot_proxy = gld * conv
                efp = (gc - spot_proxy).replace([np.inf, -np.inf], np.nan).dropna()
                if len(efp) > 10:
                    hist_data["EFP_SPREAD"] = efp
            except Exception as e:
                fetch_errors.append(f"EFP_SPREAD calc: {e}")

        # Compute Gold Allocation proxy: gold market cap as % of US M2 * 4 (global estimate)
        # Total above-ground gold ≈ 210,000 tonnes = ~6.75 billion troy oz
        if "GC=F" in hist_data and hist_data.get("_AU_ALLOC_M2") is not None:
            try:
                gc = hist_data["GC=F"]
                m2 = hist_data["_AU_ALLOC_M2"]
                m2_daily = m2.resample("D").ffill()
                aligned_alloc = pd.concat([gc, m2_daily], axis=1).dropna()
                aligned_alloc.columns = ["gold_price", "m2"]
                # Gold market cap in $B: price * 6.75B oz / 1e9
                gold_mktcap = aligned_alloc["gold_price"] * 6.75
                # Global financial assets proxy: M2 * 4 (US is ~25% of global)
                global_assets = aligned_alloc["m2"] * 4
                au_alloc = (gold_mktcap / global_assets) * 100  # as percentage
                au_alloc = au_alloc.dropna()
                if len(au_alloc) > 10:
                    hist_data["AU_ALLOCATION"] = au_alloc
            except Exception as e:
                fetch_errors.append(f"AU_ALLOCATION calc: {e}")

        # Options Skew: use CBOE SKEW index as proxy for market tail risk
        if "^SKEW" in hist_data:
            hist_data["OPT_SKEW"] = hist_data["^SKEW"]

        # Compute Implied Lease Rate: SOFR minus gold forward rate from futures curve
        if "GC=F" in hist_data and "GLD" in hist_data and "SOFR_OIS" in hist_data:
            try:
                gc = hist_data["GC=F"]
                gld = hist_data["GLD"]
                sofr_ois = hist_data["SOFR_OIS"]
                # Gold forward rate proxy: (futures / spot proxy - 1) * annualized
                conv = gc.iloc[0] / gld.iloc[0] if gld.iloc[0] != 0 else 10.0
                spot_proxy = gld * conv
                fwd_rate = ((gc / spot_proxy) - 1) * 365 / 60 * 100  # annualized %
                fwd_rate = fwd_rate.replace([np.inf, -np.inf], np.nan).dropna()
                # Lease rate = SOFR - gold forward rate
                sofr_daily = sofr_ois.reindex(fwd_rate.index, method="ffill") / 100  # convert bps to %
                lease = sofr_daily - fwd_rate
                lease = lease.dropna()
                if len(lease) > 10:
                    hist_data["LEASE_RATE"] = lease
            except Exception as e:
                fetch_errors.append(f"LEASE_RATE calc: {e}")

    except Exception as e:
        fetch_errors.append(f"YF bulk download: {e}")

    # 2B. Fetch COMEX warehouse inventory from CME
    try:
        comex_url = "https://www.cmegroup.com/delivery_reports/Gold_Stocks.xls"
        comex_resp = requests.get(comex_url, timeout=10,
                                   headers={"User-Agent": "Mozilla/5.0"})
        if comex_resp.status_code == 200:
            comex_df = pd.read_excel(io.BytesIO(comex_resp.content), engine="xlrd",
                                      header=None, skiprows=4)
            for _, row_data in comex_df.iterrows():
                if str(row_data.iloc[0]).strip().upper() == "TOTAL":
                    try:
                        reg = float(str(row_data.iloc[2]).replace(",", ""))
                        elig = float(str(row_data.iloc[4]).replace(",", ""))
                        total_oz = (reg + elig) / 1_000_000  # convert to M oz
                        dates = pd.date_range(start=start_date, end=end_date, freq='B')
                        hist_data["COMEX_INV"] = pd.Series(total_oz, index=dates)
                    except Exception:
                        pass
                    break
    except Exception as e:
        fetch_errors.append(f"COMEX_INV: {e}")

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
                "Change": "N/A", "Change Color": "gray", "Change Pct": 0.0}

    current_val = float(series.iloc[-1])
    mean_5y = float(series.mean())
    std_5y = float(series.std())

    # Period-over-period % change (daily or weekly depending on data frequency)
    change_pct = 0.0
    if len(series) >= 2:
        prev_val = float(series.iloc[-2])
        if abs(prev_val) > 1e-10:
            change_pct = (current_val - prev_val) / abs(prev_val) * 100
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
        "Change Pct": change_pct,
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

    # Pre-compute cluster max weights for weighted normalization
    cluster_max_weight = {}
    cluster_factor_weights = {}  # {cluster: {ticker: weight}}
    for _cat, indicators in FACTOR_CONFIG.items():
        for item in indicators:
            cg = item.get('cluster_group', item['ticker'])
            w = item['weight']
            if cg not in cluster_max_weight or w > cluster_max_weight[cg]:
                cluster_max_weight[cg] = w
            if cg not in cluster_factor_weights:
                cluster_factor_weights[cg] = {}
            cluster_factor_weights[cg][item['ticker']] = w
    # max contribution per cluster = max_w * max_w * 3 (matching main scoring)
    max_possible_score = sum(mw * mw * 3 for mw in cluster_max_weight.values()) if cluster_max_weight else 1.0

    daily_scores = {}
    for d in sampled_dates:
        # Collect factor scores grouped by cluster with weights
        cluster_factor_data = {}  # {cluster: [(score, weight), ...]}
        for _cat, indicators in FACTOR_CONFIG.items():
            for item in indicators:
                ticker = item['ticker']
                series = historical_data.get(ticker)
                if series is None or len(series) < 10:
                    continue
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
                if cg not in cluster_factor_data:
                    cluster_factor_data[cg] = []
                cluster_factor_data[cg].append((factor_score, weight))

        # Weighted average within clusters, then weight by max_weight
        day_total = 0.0
        for cg, items in cluster_factor_data.items():
            w_sum = sum(w for _, w in items)
            if w_sum > 0:
                w_avg = sum(s * w for s, w in items) / w_sum
            else:
                w_avg = 0.0
            mw = cluster_max_weight.get(cg, 1)
            day_total += w_avg * mw

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
                f"&sort_order=desc"
                f"&limit=5"
            )
            resp = requests.get(url, timeout=4)
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
            if item['source'] == 'SIMULATED' and (series is None or ticker not in historical_data):
                source_label = f"{ticker} (Simulated)"
            elif item['source'] == 'SIMULATED' and ticker in historical_data:
                source_label = f"{ticker} (Simulated)"
            elif series is not None and len(series) >= 10:
                source_label = f"{ticker} ({item['source']})"
            else:
                source_label = f"{ticker} (No Data)"

            # Generate 50-day sparkline SVG
            sparkline_svg = make_sparkline_svg(series, n_days=50, width=80)

            entry = {
                "Category": category,
                "Indicator": item['ind'],
                "Cluster Group": item.get('cluster_group', ''),
                "Definition": item.get('definition', ''),
                "Unit": item.get('unit', ''),
                "Ticker / Source": source_label,
                "Value": stats["Value"],
                "Change": stats.get("Change", "N/A"),
                "Change Color": stats.get("Change Color", "gray"),
                "Change Pct": stats.get("Change Pct", 0.0),
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

# Cluster-aware scoring: weighted average within clusters, then weighted sum across.
# Each cluster's contribution is scaled by its max weight so macro clusters dominate.
# Build weight lookup
_weight_lookup = {}
for indicators in FACTOR_CONFIG.values():
    for item in indicators:
        _weight_lookup[item['ind']] = item['weight']

if 'Cluster Group' in final_df.columns and final_df['Cluster Group'].notna().any():
    # For each cluster: compute weighted average of factor scores, and track max weight
    cluster_data = {}  # {cluster: {'weighted_sum': float, 'weight_sum': float, 'max_weight': float}}
    for _, row in final_df.iterrows():
        cg = row.get('Cluster Group', row['Indicator'])
        w = _weight_lookup.get(row['Indicator'], 1)
        fs = row['Total Factor Score']
        if cg not in cluster_data:
            cluster_data[cg] = {'weighted_sum': 0.0, 'weight_sum': 0.0, 'max_weight': 0}
        cluster_data[cg]['weighted_sum'] += fs * w  # weight the score by its own weight
        cluster_data[cg]['weight_sum'] += w
        cluster_data[cg]['max_weight'] = max(cluster_data[cg]['max_weight'], w)

    # Each cluster gets a normalized score in [-3, +3] (since factor_score is capped at w*3)
    # Then we weight each cluster by its max_weight for the final sum
    cluster_weighted_scores = {}
    cluster_max_contributions = {}
    for cg, cd in cluster_data.items():
        # Weighted average score for cluster (if all factors maxed at +3*w, this = 3*w)
        if cd['weight_sum'] > 0:
            cluster_avg = cd['weighted_sum'] / cd['weight_sum']
        else:
            cluster_avg = 0.0
        # Scale by max weight: high-weight clusters pull the total more
        mw = cd['max_weight']
        cluster_weighted_scores[cg] = cluster_avg * mw
        cluster_max_contributions[cg] = mw * mw * 3  # max contribution = max_w * (max_w * 3)

    overall_score = sum(cluster_weighted_scores.values())
    max_possible_score = sum(cluster_max_contributions.values()) if cluster_max_contributions else 1.0
    num_clusters = len(cluster_data)

    # Compute each factor's contribution to the overall score
    # contribution = (fs × w / cluster_weight_sum) × cluster_max_weight
    _factor_overall_contrib = {}
    for _, row in final_df.iterrows():
        cg = row.get('Cluster Group', row['Indicator'])
        w = _weight_lookup.get(row['Indicator'], 1)
        fs = row['Total Factor Score']
        cd = cluster_data.get(cg, {})
        ws = cd.get('weight_sum', 1)
        mw = cd.get('max_weight', w)
        contrib = (fs * w / ws) * mw if ws > 0 else 0.0
        _factor_overall_contrib[row['Indicator']] = contrib
    final_df['Overall Contribution'] = final_df['Indicator'].map(_factor_overall_contrib).fillna(0.0)

    # Store breakdown for display
    cluster_breakdown = {}
    for cg, cd in cluster_data.items():
        mw = cd['max_weight']
        cluster_breakdown[cg] = {
            'score': cluster_weighted_scores[cg],
            'max': cluster_max_contributions[cg],
            'pct': cluster_weighted_scores[cg] / max_possible_score * 100 if max_possible_score > 0 else 0,
            'max_weight': mw,
        }
else:
    # Fallback: simple weighted sum
    overall_score = final_df['Total Factor Score'].sum()
    max_possible_score = sum(
        item['weight'] * 3
        for indicators in FACTOR_CONFIG.values()
        for item in indicators
    )
    num_clusters = total_factors
    cluster_breakdown = {}
    final_df['Overall Contribution'] = final_df['Total Factor Score']

# Compute previous-day cluster scores for daily change column
_prev_cluster_scores = {}
if cluster_breakdown:
    for _cat, indicators in FACTOR_CONFIG.items():
        for item in indicators:
            ticker = item['ticker']
            series = historical_data.get(ticker)
            if series is None or len(series) < 12:
                continue
            # Use second-to-last value as "previous day"
            prev_series = series.iloc[:-1]
            if len(prev_series) < 10:
                continue
            prev_val = float(prev_series.iloc[-1])
            prev_mean = float(prev_series.mean())
            prev_std = float(prev_series.std())
            if prev_std == 0:
                continue
            z = (prev_val - prev_mean) / prev_std
            w = item['weight']
            raw = z * w if item['higher_is_bullish'] else -z * w
            fs = max(min(raw, w * 3), -w * 3)
            cg = item.get('cluster_group', ticker)
            if cg not in _prev_cluster_scores:
                _prev_cluster_scores[cg] = {'weighted_sum': 0.0, 'weight_sum': 0.0, 'max_weight': 0}
            _prev_cluster_scores[cg]['weighted_sum'] += fs * w
            _prev_cluster_scores[cg]['weight_sum'] += w
            _prev_cluster_scores[cg]['max_weight'] = max(_prev_cluster_scores[cg]['max_weight'], w)
    # Convert to final scores matching current methodology
    for cg, cd in _prev_cluster_scores.items():
        if cd['weight_sum'] > 0:
            avg = cd['weighted_sum'] / cd['weight_sum']
        else:
            avg = 0.0
        mw = cd['max_weight']
        _prev_cluster_scores[cg] = avg * mw  # same as current: cluster_avg * max_weight

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
    '<div class="summary-panel">'
    '<div class="summary-flex">'
    # Big score + sparkline
    '<div class="summary-score-block">'
    '<div class="summary-label">Gold Score</div>'
    f'<div class="summary-score-value" style="color:{interp_color};">{gold_score:+.2f}</div>'
    f'<div class="summary-interp" style="color:{interp_color};">'
    f'{interp_emoji} {interpretation}</div>'
    f'<div class="summary-chg" style="color:{score_chg_color};">{score_chg_str}</div>'
    '<div style="margin-top:8px;">'
    '<div style="font-size:0.65rem;color:#666;margin-bottom:2px;">90-Day Score Trend</div>'
    f'{score_sparkline_svg}'
    '</div>'
    '</div>'
    # Score breakdown — show top cluster contributions
    '<div class="summary-breakdown">'
    '<div class="summary-label" style="margin-bottom:10px;">Score Breakdown</div>'
    '<table class="summary-table">'
    f'<tr><td>Weighted Score</td>'
    f'<td class="summary-td-right" style="font-weight:600;">{overall_score:+.1f}</td></tr>'
    f'<tr><td>Max Possible</td>'
    f'<td class="summary-td-right">±{max_possible_score:.0f}</td></tr>'
    f'<tr style="border-top:1px solid #333;">'
    f'<td style="padding-top:6px;">Normalized</td>'
    f'<td class="summary-td-right" style="padding-top:6px;font-weight:700;color:{interp_color};">{gold_score:+.2f}</td></tr>'
    f'<tr><td>Last Period Δ</td>'
    f'<td class="summary-td-right" style="font-weight:600;color:{score_chg_color};">{score_chg_str}</td></tr>'
    '</table>'
    '</div>'
    '</div>'
)

# ─── Analyst Forecasts & Support/Resistance ──────────────────────────────────
_gc = historical_data.get("GC=F")
_targets_html = ""
_analyst_forecasts = []  # store for the expander below
_forecast_year = 2026
if _gc is not None and len(_gc) > 200:
    _spot = float(_gc.iloc[-1])

    # --- LBMA Analyst Consensus Forecasts ---
    _analyst_forecasts, _forecast_year = fetch_analyst_forecasts()
    _n_analysts = len(_analyst_forecasts)

    # Compute consensus statistics
    _all_lows = [a["low"] for a in _analyst_forecasts]
    _all_avgs = [a["avg"] for a in _analyst_forecasts]
    _all_highs = [a["high"] for a in _analyst_forecasts]

    _consensus_low = min(_all_lows)
    _consensus_avg = sum(_all_avgs) / _n_analysts
    _consensus_high = max(_all_highs)

    # Find who set the extremes
    _low_analyst = min(_analyst_forecasts, key=lambda a: a["low"])
    _high_analyst = max(_analyst_forecasts, key=lambda a: a["high"])

    # How many forecast above/below spot
    _above_spot = sum(1 for a in _analyst_forecasts if a["avg"] > _spot)
    _below_spot = _n_analysts - _above_spot

    # % vs spot
    _low_pct = (_consensus_low - _spot) / _spot * 100
    _avg_pct = (_consensus_avg - _spot) / _spot * 100
    _high_pct = (_consensus_high - _spot) / _spot * 100

    def _pct_color(p):
        return "#22c55e" if p > 0.5 else ("#ef4444" if p < -0.5 else "#facc15")
    def _pct_arrow(p):
        return "▲" if p > 0.5 else ("▼" if p < -0.5 else "—")

    # --- Support & Resistance (Pivot Points) ---
    # Classic pivot: use prior day's High, Low, Close
    try:
        _gc_full = yf.download("GC=F", period="5d", interval="1d", progress=False)
        if isinstance(_gc_full.columns, pd.MultiIndex):
            _h = float(_gc_full[("High", "GC=F")].dropna().iloc[-2])
            _l = float(_gc_full[("Low", "GC=F")].dropna().iloc[-2])
            _c = float(_gc_full[("Close", "GC=F")].dropna().iloc[-2])
        else:
            _h = float(_gc_full["High"].dropna().iloc[-2])
            _l = float(_gc_full["Low"].dropna().iloc[-2])
            _c = float(_gc_full["Close"].dropna().iloc[-2])
    except Exception:
        _h, _l, _c = _spot * 1.005, _spot * 0.995, _spot

    _pivot = (_h + _l + _c) / 3
    _r1 = 2 * _pivot - _l
    _r2 = _pivot + (_h - _l)
    _r3 = _h + 2 * (_pivot - _l)
    _s1 = 2 * _pivot - _h
    _s2 = _pivot - (_h - _l)
    _s3 = _l - 2 * (_h - _pivot)

    # --- Build HTML ---
    _targets_html = (
        '<div class="targets-flex">'

        # Analyst Consensus
        '<div class="targets-col">'
        f'<div class="summary-label" style="margin-bottom:8px;">'
        f'{_forecast_year} Gold Forecasts ({_n_analysts} Analysts)</div>'
        '<table class="summary-table">'
        '<tr style="border-bottom:1px solid #333;">'
        '<th style="padding:3px 0;text-align:left;color:#888;font-weight:600;"></th>'
        '<th style="padding:3px 8px;text-align:right;color:#888;font-weight:600;">Price</th>'
        '<th style="padding:3px 6px;text-align:right;color:#888;font-weight:600;">vs Spot</th>'
        '<th style="padding:3px 0;text-align:left;color:#888;font-weight:600;font-size:0.7rem;">Source</th></tr>'

        f'<tr style="border-bottom:1px solid #1a1a2e;">'
        f'<td style="padding:4px 0;color:#ef4444;font-weight:600;">Low</td>'
        f'<td style="padding:4px 8px;text-align:right;font-family:monospace;font-weight:600;">'
        f'${_consensus_low:,.0f}</td>'
        f'<td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:0.78rem;color:{_pct_color(_low_pct)};">'
        f'{_pct_arrow(_low_pct)} {_low_pct:+.1f}%</td>'
        f'<td style="padding:4px 0;font-size:0.65rem;color:#888;">'
        f'{_low_analyst["analyst"][:15]}<br><span style="color:#555;">{_low_analyst["institution"][:20]}</span></td></tr>'

        f'<tr style="border-bottom:1px solid #333;background:rgba(255,255,255,0.03);">'
        f'<td style="padding:5px 0;color:#60a5fa;font-weight:700;">Avg</td>'
        f'<td style="padding:5px 8px;text-align:right;font-family:monospace;font-weight:700;color:#60a5fa;">'
        f'${_consensus_avg:,.0f}</td>'
        f'<td style="padding:5px 6px;text-align:right;font-family:monospace;font-size:0.78rem;color:{_pct_color(_avg_pct)};">'
        f'{_pct_arrow(_avg_pct)} {_avg_pct:+.1f}%</td>'
        f'<td style="padding:5px 0;font-size:0.65rem;color:#888;">{_n_analysts} analysts</td></tr>'

        f'<tr style="border-bottom:1px solid #1a1a2e;">'
        f'<td style="padding:4px 0;color:#22c55e;font-weight:600;">High</td>'
        f'<td style="padding:4px 8px;text-align:right;font-family:monospace;font-weight:600;">'
        f'${_consensus_high:,.0f}</td>'
        f'<td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:0.78rem;color:{_pct_color(_high_pct)};">'
        f'{_pct_arrow(_high_pct)} {_high_pct:+.1f}%</td>'
        f'<td style="padding:4px 0;font-size:0.65rem;color:#888;">'
        f'{_high_analyst["analyst"][:15]}<br><span style="color:#555;">{_high_analyst["institution"][:20]}</span></td></tr>'

        '</table>'
        f'<div style="font-size:0.58rem;color:#555;margin-top:6px;">'
        f'{_above_spot} of {_n_analysts} forecast above spot (${_spot:,.0f}) '
        f'&middot; LBMA Survey (Jan) + Wall St Banks (Feb/Mar)</div>'
        '</div>'

        # Support & Resistance
        '<div class="targets-col">'
        '<div class="summary-label" style="margin-bottom:8px;">Daily Pivot Levels</div>'
        '<table class="summary-table">'
        f'<tr style="border-bottom:1px solid #1a1a2e;">'
        f'<td style="padding:3px 0;color:#ef4444;">R3</td>'
        f'<td class="summary-td-right">${_r3:,.0f}</td></tr>'
        f'<tr style="border-bottom:1px solid #1a1a2e;">'
        f'<td style="padding:3px 0;color:#ef4444;">R2</td>'
        f'<td class="summary-td-right">${_r2:,.0f}</td></tr>'
        f'<tr style="border-bottom:1px solid #1a1a2e;">'
        f'<td style="padding:3px 0;color:#fca5a5;">R1</td>'
        f'<td class="summary-td-right">${_r1:,.0f}</td></tr>'
        f'<tr style="border-bottom:1px solid #333;background:rgba(255,255,255,0.03);">'
        f'<td style="padding:4px 0;color:#facc15;font-weight:600;">Pivot</td>'
        f'<td class="summary-td-right" style="font-weight:600;color:#facc15;">'
        f'${_pivot:,.0f}</td></tr>'
        f'<tr style="border-bottom:1px solid #1a1a2e;">'
        f'<td style="padding:3px 0;color:#86efac;">S1</td>'
        f'<td class="summary-td-right">${_s1:,.0f}</td></tr>'
        f'<tr style="border-bottom:1px solid #1a1a2e;">'
        f'<td style="padding:3px 0;color:#22c55e;">S2</td>'
        f'<td class="summary-td-right">${_s2:,.0f}</td></tr>'
        f'<tr>'
        f'<td style="padding:3px 0;color:#22c55e;">S3</td>'
        f'<td class="summary-td-right">${_s3:,.0f}</td></tr>'
        '</table>'
        f'<div style="font-size:0.58rem;color:#555;margin-top:6px;">'
        f'Classic pivot points (H=${_h:,.0f} L=${_l:,.0f} C=${_c:,.0f})</div>'
        '</div>'
        '</div>'
    )

_summary_html += _targets_html + (
    f'<div class="summary-footer">'
    f'{total_factors} factors &middot; {num_clusters} clusters &middot; '
    f'{total_factors - simulated_count} live &middot; {simulated_count} simulated</div>'
) + (
    # Gauge bar
    '<div class="gauge-wrap">'
    '<div class="gauge-bar">'
    f'<div class="gauge-needle" style="left:{gauge_pct:.1f}%;"></div>'
    '<div class="gauge-center"></div>'
    '</div>'
    '<div class="gauge-labels">'
    '<span>-1.0 Bearish</span><span>Neutral</span><span>+1.0 Bullish</span>'
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

# ─── Score Calculation Detail Expander ─────────────────────────────────────
# Cluster descriptions for hover tooltips — explain how each cluster impacts gold
_CLUSTER_DESCRIPTIONS = {
    "Real Rates": "THE dominant driver of gold. When real interest rates (nominal yield minus inflation) are low or negative, the opportunity cost of holding non-yielding gold drops to zero — making it attractive vs bonds. Deeply negative real rates (like 2020-2022) have historically produced the strongest gold bull markets. Currently tracking 10Y TIPS, 5Y TIPS, Fed Funds real rate, and a GDP-weighted global composite.",
    "Dollar Strength": "Gold is priced in USD globally, so a weaker dollar mechanically lifts gold in other currencies and boosts international demand. The DXY index captures the dollar's strength against major trading partners. A falling DXY is one of the most reliable bullish catalysts for gold — the correlation is roughly -0.8 over long periods.",
    "Global Liquidity": "When central banks expand money supply (M2), more fiat currency chases the same amount of gold. M2 growth acts as a long-term secular driver — gold tends to track global M2 expansion over decades. Rapid M2 growth (like 2020's 25% surge) signals currency debasement that historically precedes gold rallies.",
    "Fiscal & Debt": "Unsustainable government debt levels (US debt/GDP >120%) and persistent fiscal deficits imply future monetization — the government will eventually need to inflate away its obligations. Rising debt service costs (interest payments) force central banks toward financial repression (keeping rates below inflation), which is structurally bullish for gold.",
    "Central Bank Buying": "Central banks (especially China, India, Poland, Turkey) have been net buyers since 2010, purchasing 1,000+ tonnes/year recently. This creates a structural price floor. Central bank buying is a slow-moving but powerful signal — it reflects de-dollarization and reserve diversification trends that take years to play out.",
    "Inflation Expectations": "Gold is traditionally an inflation hedge. When markets expect higher future inflation (via breakevens, consumer surveys, or nowcasts), gold becomes more attractive as a store of value. The 5Y breakeven, UMich consumer expectations, and Cleveland Fed nowcast capture different dimensions of inflation sentiment.",
    "Yield Curve": "A steepening yield curve (long rates rising faster than short rates) often signals economic normalization or inflation expectations building. For gold, an inverted curve signals recession risk (bullish for safe havens), while steepening can signal inflation expectations (also bullish). The curve's shape reflects the market's macro outlook.",
    "Credit Spreads": "Widening high-yield credit spreads signal rising systemic risk and fear in financial markets. When corporate bond spreads blow out, it signals stress that typically drives safe-haven flows into gold. Tight spreads indicate complacency — fewer reasons to hold gold as insurance.",
    "COT Positioning": "CFTC Commitments of Traders data shows who's long and short gold futures. Extremely crowded managed-money longs signal exhaustion (bearish contrarian signal). Swap dealer positioning reflects 'smart money' commercial hedging. Open interest tracks total speculative participation — rising OI with rising price confirms trend strength.",
    "Physical Demand Asia": "China and India account for ~50% of global gold demand. The Shanghai premium (price over London) directly measures Chinese physical appetite. Indian demand is seasonal (wedding/festival season Oct-Feb). When Asian physical demand is strong, it provides a consumption floor that supports prices.",
    "Fed Liquidity": "The Fed's balance sheet operations directly impact liquidity. Reverse repo facility usage drains excess cash from the system (bearish for gold). Commercial bank reserves measure base money availability. When the Fed is injecting liquidity (QE), gold benefits from the expanding monetary base.",
    "Geopolitical Risk": "Gold is the ultimate safe-haven asset during geopolitical crises — wars, sanctions, political instability. The GPR index captures news-based geopolitical tension. While spikes can be sharp, geopolitical premiums tend to fade unless they escalate into sustained economic disruption.",
    "Moving Averages": "Price above both the 50-day and 200-day moving averages confirms a bullish trend. A 'golden cross' (50D crossing above 200D) is a classic technical buy signal. This is a trend-following indicator — it won't catch bottoms but confirms momentum.",
    "ETF & Retail Physical": "GLD and other gold ETF flows track institutional/retail accumulation. US Mint coin sales proxy grassroots physical demand. Retail premiums over spot indicate dealer supply constraints. Strong ETF inflows + high physical premiums = broad-based demand across investor types.",
    "Gold Ratios": "Relative value metrics compare gold to other assets. Gold/Silver ratio above 80 historically reverts (silver catches up). Gold vs Bitcoin tracks competition for 'digital gold' narrative. Gold as % of global financial assets shows allocation room — currently ~1-2% vs historical peaks of 5%+.",
    "Volatility": "VIX spikes (equity fear) trigger safe-haven capital flight into gold. Gold's own volatility (GVZ) rising signals uncertainty. The combination of high equity vol + low gold vol is the ideal setup — it means gold hasn't moved yet but fear is building.",
    "Futures Structure": "Backwardation (near-month futures > far-month) indicates tight physical supply — the market is willing to pay a premium for immediate delivery. Contango is normal. Persistent backwardation is a powerful bullish signal for physical gold.",
    "Mining Fundamentals": "All-in sustaining costs (AISC ~$1,300/oz) create a price floor — miners shut down below this. Mine supply is flat/declining globally (peak gold thesis). Capex underinvestment since 2013 means limited new supply coming online. Reserve depletion forces miners to pay up for acquisitions.",
    "Banking Stress": "TED spread and SOFR-OIS spread measure interbank lending stress. When banks don't trust each other (spreads widen), it signals systemic risk that drives safe-haven demand. These spiked during 2008, 2020, and the 2023 SVB crisis — each time gold rallied.",
    "Basel III": "Basel III NSFR rules reclassified unallocated gold from Tier 1 to Tier 3, requiring banks to hold more capital against gold positions. This structurally reduces paper gold supply and may tighten the physical market over time. A slow-burning regulatory tailwind.",
    "COMEX Inventory": "COMEX registered and eligible gold stocks track the deliverable supply backing futures contracts. Falling inventories signal physical tightness — fewer ounces available to settle contracts. When COMEX stocks plunge (as in early 2022), it can trigger short squeezes.",
    "Mining Equities": "Gold miners (GDX) typically lead the metal — when miners outperform gold, it signals the market expects higher gold prices. Junior miners (GDXJ) outperforming seniors signals risk-on appetite within the precious metals sector. Miners lagging metal = bearish divergence.",
    "Lease Rates": "Gold lease rates reflect the cost of borrowing physical gold. Rising lease rates signal physical market stress — someone urgently needs to borrow gold (possibly to cover short positions or deliver on contracts). Spikes in lease rates have preceded major gold rallies.",
    "EFP Spread": "The Exchange for Physical spread measures the gap between COMEX futures and London spot. A wide EFP signals dislocation between paper and physical markets — arbitrageurs can't close the gap, indicating supply chain stress. This spiked dramatically during COVID.",
    "Bollinger Bands": "When price trades above the upper Bollinger Band (2 standard deviations above 20-day MA), gold is in an extended move. While this can signal overbought conditions, sustained upper-band riding indicates strong trend momentum. Below lower band = oversold bounce potential.",
    "RSI": "The Relative Strength Index measures momentum on a 0-100 scale. RSI above 70 = overbought (pullback risk), below 30 = oversold (bounce potential). RSI divergences (price making new highs while RSI doesn't) signal weakening momentum before reversals.",
    "Options Skew": "Options skew measures the relative cost of call vs put options. Heavy call skew (calls much pricier than puts) implies the market is over-positioned for upside — often a contrarian bearish signal. Put skew indicates hedging demand / fear.",
    "Silver Beta": "Silver has higher beta to gold — it moves more in both directions. When silver outperforms gold (falling gold/silver ratio), it confirms a broad precious metals bull market. Silver lagging while gold rises suggests the rally lacks breadth.",
    "Jewelry Demand": "Jewelry accounts for ~50% of annual gold demand by volume but is price-sensitive — demand falls when gold gets expensive. It provides a consumption floor at lower prices. Less impactful on price direction than investment demand but stabilizes the market.",
    "Dealer Gamma": "Market makers' options gamma exposure affects short-term volatility. Negative gamma means dealers must sell into declines and buy into rallies (amplifying moves). Positive gamma means they dampen moves. This is a short-term flow indicator, not directional.",
    "Retail Sentiment": "Google Trends for 'buy gold' captures retail search interest. Extremely high search volume often marks local tops (everyone who wanted to buy already has). Low search volume near price lows can signal capitulation. A contrarian indicator at extremes.",
    "Scrap Supply": "When gold prices are high, recycling/scrap flows increase as people sell old jewelry. This additional supply can cap price rallies. Scrap supply is price-responsive — it acts as a natural dampener on runaway bull moves.",
    "Sovereign Risk": "US CDS spreads measure the market's perceived probability of US government default. Rising sovereign risk undermines confidence in government bonds, making gold more attractive as a non-sovereign store of value. Spikes during debt ceiling crises boost gold.",
    "Volume Profile": "Trading volume at specific price levels identifies support and resistance zones. High volume at a price level means many positions were established there — creating a 'memory' that acts as support/resistance when price returns. A structural indicator.",
    "Ad Spending": "Gold advertising spending (TV, digital) tends to spike during late-cycle retail mania — companies advertise more when they know retail demand is hot. This is a contrarian indicator: heavy gold ads = late-stage euphoria, often preceding corrections.",
}

if cluster_breakdown:
    # Sort clusters by absolute contribution (largest impact first)
    _sorted_clusters = sorted(cluster_breakdown.items(), key=lambda x: abs(x[1]['score']), reverse=True)
    _max_abs_score = max(abs(c['score']) for c in cluster_breakdown.values()) if cluster_breakdown else 1
    _cb_rows = ""
    for _cg_name, _cb in _sorted_clusters:
        _cb_score = _cb['score']
        _cb_max = _cb['max']
        _cb_mw = _cb['max_weight']
        _norm = _cb_score / _cb_max if _cb_max > 0 else 0
        if _cb_score > 0.5:
            _cb_color = "#22c55e"
            _cb_dir = "Bullish"
        elif _cb_score < -0.5:
            _cb_color = "#ef4444"
            _cb_dir = "Bearish"
        else:
            _cb_color = "#facc15"
            _cb_dir = "Neutral"
        # Bar: proportional to score relative to largest cluster
        _bar_pct = (abs(_cb_score) / _max_abs_score * 100) if _max_abs_score > 0 else 0
        _bar_pct = max(min(_bar_pct, 100), 1)
        _bar_html = f'<div style="display:flex;align-items:center;gap:4px;"><div style="width:{_bar_pct:.0f}%;height:10px;background:{_cb_color};border-radius:2px;min-width:2px;"></div><span style="font-size:0.7rem;color:#888;">{_cb_score:+.1f}</span></div>'
        # Daily change
        _prev_score = _prev_cluster_scores.get(_cg_name, _cb_score)
        _daily_chg = _cb_score - _prev_score if isinstance(_prev_score, (int, float)) else 0.0
        if _daily_chg > 0.05:
            _chg_str = f'<span style="color:#22c55e;">▲ {_daily_chg:+.1f}</span>'
        elif _daily_chg < -0.05:
            _chg_str = f'<span style="color:#ef4444;">▼ {_daily_chg:+.1f}</span>'
        else:
            _chg_str = f'<span style="color:#666;">— 0.0</span>'
        # CSS tooltip description
        _tooltip_text = _CLUSTER_DESCRIPTIONS.get(_cg_name, "No description available.")
        _tooltip_text_escaped = _tooltip_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        _cb_rows += (
            f'<tr class="cluster-row">'
            f'<td style="padding:4px 6px;font-size:0.78rem;position:relative;">'
            f'<span class="cluster-name">{_cg_name}'
            f'<span class="cluster-tooltip">{_tooltip_text_escaped}</span>'
            f'</span></td>'
            f'<td style="padding:4px 6px;font-size:0.78rem;text-align:center;">{_cb_mw:.0f}</td>'
            f'<td style="padding:4px 6px;font-size:0.78rem;text-align:right;color:{_cb_color};font-weight:600;">{_cb_score:+.1f}</td>'
            f'<td style="padding:4px 6px;font-size:0.78rem;text-align:right;">{_cb_max:.0f}</td>'
            f'<td style="padding:4px 6px;width:25%;">{_bar_html}</td>'
            f'<td style="padding:4px 6px;font-size:0.75rem;text-align:center;font-weight:600;">{_chg_str}</td>'
            f'<td style="padding:4px 6px;font-size:0.75rem;text-align:center;color:{_cb_color};">{_cb_dir}</td>'
            f'</tr>'
        )
    # Compute total daily change
    _total_daily_chg = overall_score - sum(_prev_cluster_scores.get(cg, cb['score']) for cg, cb in cluster_breakdown.items() if isinstance(_prev_cluster_scores.get(cg, cb['score']), (int, float)))
    if _total_daily_chg > 0.05:
        _total_chg_str = f'<span style="color:#22c55e;">▲ {_total_daily_chg:+.1f}</span>'
    elif _total_daily_chg < -0.05:
        _total_chg_str = f'<span style="color:#ef4444;">▼ {_total_daily_chg:+.1f}</span>'
    else:
        _total_chg_str = f'<span style="color:#666;">— 0.0</span>'

    _calc_html = (
        '<style>'
        '.cluster-name { cursor: help; text-decoration: underline dotted #555; text-underline-offset: 3px; position: relative; display: inline-block; }'
        '.cluster-tooltip { visibility: hidden; opacity: 0; position: absolute; left: 0; top: 100%; z-index: 9999; '
        'background: #1a1a2e; color: #ccc; padding: 10px 14px; border-radius: 8px; font-size: 0.75rem; '
        'line-height: 1.5; width: 380px; max-width: 90vw; box-shadow: 0 4px 20px rgba(0,0,0,0.5); '
        'border: 1px solid #333; pointer-events: none; transition: opacity 0.15s; white-space: normal; '
        'font-weight: normal; text-decoration: none; }'
        '.cluster-name:hover .cluster-tooltip { visibility: visible; opacity: 1; }'
        '.cluster-row:hover { background: #1e1e30; }'
        '</style>'
        '<div style="padding:8px 0;">'
        '<div style="font-size:0.82rem;color:#aaa;margin-bottom:12px;line-height:1.5;">'
        '<b>Scoring Formula:</b> For each factor: <code style="background:#1a1a2e;padding:2px 4px;border-radius:3px;">raw = z_score × weight</code> '
        '(flipped for bearish-when-high). Capped at <code style="background:#1a1a2e;padding:2px 4px;border-radius:3px;">±weight×3</code>. '
        'Within each cluster, scores are <b>weight-averaged</b>. Then each cluster contributes '
        '<code style="background:#1a1a2e;padding:2px 4px;border-radius:3px;">cluster_avg × max_weight</code> to the total. '
        f'Final: <code style="background:#1a1a2e;padding:2px 4px;border-radius:3px;">{overall_score:+.1f} / {max_possible_score:.0f} = {gold_score:+.3f}</code>'
        '</div>'
        '<table class="factor-table" style="table-layout:auto;">'
        '<tr>'
        '<th style="padding:5px 6px;text-align:left;">Cluster</th>'
        '<th style="padding:5px 6px;text-align:center;">Wt</th>'
        '<th style="padding:5px 6px;text-align:right;">Score</th>'
        '<th style="padding:5px 6px;text-align:right;">Max</th>'
        '<th style="padding:5px 6px;text-align:left;">Contribution</th>'
        '<th style="padding:5px 6px;text-align:center;">Δ Day</th>'
        '<th style="padding:5px 6px;text-align:center;">Signal</th>'
        '</tr>'
        f'{_cb_rows}'
        '<tr style="border-top:2px solid #444;">'
        f'<td style="padding:6px;font-weight:700;">TOTAL</td>'
        f'<td></td>'
        f'<td style="padding:6px;text-align:right;font-weight:700;color:{interp_color};">{overall_score:+.1f}</td>'
        f'<td style="padding:6px;text-align:right;font-weight:700;">{max_possible_score:.0f}</td>'
        f'<td style="padding:6px;font-weight:700;color:{interp_color};">= {gold_score:+.3f}</td>'
        f'<td style="padding:6px;text-align:center;font-weight:600;">{_total_chg_str}</td>'
        f'<td style="padding:6px;text-align:center;font-weight:700;color:{interp_color};">{interpretation}</td>'
        '</tr>'
        '</table></div>'
    )
    with st.expander(f"📊 Score Calculation — {num_clusters} clusters, {total_factors} factors → {gold_score:+.3f}", expanded=False):
        st.markdown(_calc_html, unsafe_allow_html=True)

# ─── Analyst Forecasts Detail Expander ─────────────────────────────────────
if _analyst_forecasts and _gc is not None and len(_gc) > 200:
    _spot_for_table = float(_gc.iloc[-1])
    _n_lbma = sum(1 for a in _analyst_forecasts if a.get("source") == "LBMA")
    _n_bank = sum(1 for a in _analyst_forecasts if a.get("source") == "Bank")
    with st.expander(f"View All {len(_analyst_forecasts)} Analyst Forecasts — {_n_lbma} LBMA + {_n_bank} Wall Street", expanded=False):
        # Build HTML table of all analysts
        _at_html = (
            '<div class="factor-table-wrap">'
            '<table class="factor-table" style="min-width:620px;">'
            '<tr>'
            '<th style="padding:6px 8px;">Analyst</th>'
            '<th style="padding:6px 8px;">Institution</th>'
            '<th style="padding:6px 8px;text-align:right;">Low</th>'
            '<th style="padding:6px 8px;text-align:right;">Average</th>'
            '<th style="padding:6px 8px;text-align:right;">High</th>'
            '<th style="padding:6px 8px;text-align:right;">vs Spot</th>'
            '<th style="padding:6px 8px;text-align:center;">Date</th>'
            '<th style="padding:6px 8px;text-align:center;">Source</th>'
            '</tr>'
        )
        _above_count = 0
        for _af in _analyst_forecasts:
            _vs = (_af["avg"] - _spot_for_table) / _spot_for_table * 100
            _vs_clr = "#22c55e" if _vs > 0.5 else ("#ef4444" if _vs < -0.5 else "#facc15")
            _vs_arrow = "▲" if _vs > 0.5 else ("▼" if _vs < -0.5 else "—")
            # Highlight if spot is within analyst's range
            _in_range = _af["low"] <= _spot_for_table <= _af["high"]
            _row_bg = "background:rgba(96,165,250,0.06);" if _in_range else ""
            if _af["avg"] > _spot_for_table:
                _above_count += 1
            # Source badge color
            _src = _af.get("source", "LBMA")
            if _src == "Bank":
                _src_badge = '<span style="font-size:0.65rem;background:rgba(96,165,250,0.15);color:#60a5fa;padding:1px 6px;border-radius:3px;">Bank</span>'
            else:
                _src_badge = '<span style="font-size:0.65rem;background:rgba(250,204,21,0.15);color:#facc15;padding:1px 6px;border-radius:3px;">LBMA</span>'
            _af_date = _af.get("date", "Jan 2026")
            _at_html += (
                f'<tr style="{_row_bg}">'
                f'<td style="padding:5px 8px;font-weight:500;color:#d0d0d0;">{_af["analyst"]}</td>'
                f'<td style="padding:5px 8px;color:#888;font-size:0.8rem;">{_af["institution"]}</td>'
                f'<td style="padding:5px 8px;text-align:right;font-family:monospace;color:#ef4444;">${_af["low"]:,.0f}</td>'
                f'<td style="padding:5px 8px;text-align:right;font-family:monospace;font-weight:600;color:#60a5fa;">${_af["avg"]:,.0f}</td>'
                f'<td style="padding:5px 8px;text-align:right;font-family:monospace;color:#22c55e;">${_af["high"]:,.0f}</td>'
                f'<td style="padding:5px 8px;text-align:right;font-family:monospace;color:{_vs_clr};">'
                f'{_vs_arrow} {_vs:+.1f}%</td>'
                f'<td style="padding:5px 8px;text-align:center;font-size:0.75rem;color:#888;">{_af_date}</td>'
                f'<td style="padding:5px 8px;text-align:center;">{_src_badge}</td>'
                '</tr>'
            )
        _at_html += (
            '</table></div>'
            f'<div style="margin-top:10px;font-size:0.78rem;color:#888;">'
            f'<b>{_above_count}</b> of {len(_analyst_forecasts)} analysts forecast an average above current spot '
            f'(${_spot_for_table:,.0f}). '
            f'Rows highlighted in blue indicate spot price falls within the analyst\'s low-high range.'
            f'</div>'
            f'<div style="margin-top:6px;font-size:0.65rem;color:#555;">'
            f'<span style="color:#facc15;">LBMA</span> = LBMA Annual Forecast Survey (published Jan 20, {_forecast_year}) &middot; '
            f'<span style="color:#60a5fa;">Bank</span> = Wall Street research notes (Feb/Mar {_forecast_year}). '
            f'Sorted by average forecast ascending.</div>'
        )
        st.markdown(_at_html, unsafe_allow_html=True)

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
        '<div class="spot-card">'
        f'<div class="summary-label">{metal["icon"]} {metal["name"]}</div>'
        f'<div class="spot-price">${current:,.2f}</div>'
        f'<div class="spot-change" style="color:{chg_color};">'
        f'{chg_arrow} ${abs(change):,.2f} ({change_pct:+.2f}%)</div>'
        f'<div style="margin-top:8px;display:flex;justify-content:center;">'
        f'<div style="display:inline-block;">'
        f'<div style="font-size:0.6rem;color:#666;margin-bottom:2px;">50-Day Trend</div>'
        f'{sparkline}</div></div>'
        '</div>'
    )

_spot_html = (
    '<div class="spot-flex">'
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
        height=420,
        autosize=True,
        margin=dict(l=45, r=45, t=25, b=35),
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
                                  div_id="goldChart", config={"displayModeBar": False, "responsive": True})

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
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="color-scheme" content="dark">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    background: #0f0f23;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color-scheme: dark;
    -webkit-text-size-adjust: 100%;
    overflow-x: hidden;
  }

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
    -webkit-overflow-scrolling: touch;
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
    padding: 4px 12px;
    font-size: 0.72rem;
    cursor: pointer;
    display: none;
    transition: background 0.15s;
    -webkit-tap-highlight-color: transparent;
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
  .news-list { flex: 1; overflow-y: auto; -webkit-overflow-scrolling: touch; }

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
    font-size: 0.76rem;
    color: #d0d0d0;
    margin: 4px 0;
    line-height: 1.45;
    padding: 2px 0;
  }
  .headline-item a {
    color: #d0d0d0;
    text-decoration: none;
    transition: color 0.15s;
    -webkit-tap-highlight-color: rgba(96,165,250,0.2);
  }
  .headline-item a:hover, .headline-item a:active { color: #60a5fa; }
  .headline-src { color: #666; }
  .headline-upcoming {
    color: #fbbf24;
    font-style: italic;
  }
  .headline-upcoming::before {
    content: "\\01F4C5 ";
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

  /* ── Mobile: stack vertically ── */
  @media (max-width: 768px) {
    .layout {
      flex-direction: column;
      gap: 10px;
    }
    .chart-panel {
      width: 100%;
      order: 1;
    }
    .news-panel {
      flex: none;
      width: 100%;
      min-width: unset;
      max-height: 300px;
      order: 2;
      border-radius: 8px;
      padding: 10px 12px;
    }
    .news-title { font-size: 0.82rem; }
    .headline-item { font-size: 0.74rem; }
    .hint { font-size: 0.68rem; }
    /* Make plotly chart touch-friendly */
    .chart-panel .js-plotly-plot { cursor: default !important; }
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

    components.html(interactive_html, height=760, scrolling=True)

else:
    st.info("Insufficient data to render dual-axis chart.")

# ─── Comprehensive CSS: Summary + Factor Tables + Mobile ──────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════════════════════════════
   SUMMARY PANEL
   ══════════════════════════════════════════════════════════════════════════════ */
.summary-panel {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    overflow: hidden;
}
.summary-flex {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 20px;
}
.summary-score-block {
    flex: 0 0 auto;
    text-align: center;
    min-width: 160px;
}
.summary-score-value {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.summary-interp {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 4px;
}
.summary-chg {
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 6px;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.summary-breakdown {
    flex: 1 1 240px;
    min-width: 200px;
}
.summary-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.summary-table {
    font-size: 0.82rem;
    color: #d0d0d0;
    border-collapse: collapse;
    width: 100%;
}
.summary-table td {
    padding: 3px 0;
    color: #888;
}
.summary-td-right {
    text-align: right;
    font-family: 'SF Mono', 'Fira Code', monospace;
    padding: 3px 0;
}
.summary-footer {
    margin-top: 10px;
    font-size: 0.65rem;
    color: #555;
    border-top: 1px solid #333;
    padding-top: 6px;
}

/* Targets (Predictions + Pivot) */
.targets-flex {
    margin-top: 16px;
    border-top: 1px solid #333;
    padding-top: 14px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
}
.targets-col {
    flex: 1 1 180px;
    min-width: 150px;
}

/* Gauge */
.gauge-wrap { margin-top: 18px; }
.gauge-bar {
    position: relative;
    height: 18px;
    background: linear-gradient(to right,
        #ef4444 0%, #fca5a5 25%, #facc15 45%, #facc15 55%, #86efac 75%, #22c55e 100%);
    border-radius: 9px;
    overflow: visible;
}
.gauge-needle {
    position: absolute;
    top: -3px;
    transform: translateX(-50%);
    width: 4px;
    height: 24px;
    background: white;
    border-radius: 2px;
    box-shadow: 0 0 6px rgba(255,255,255,0.6);
}
.gauge-center {
    position: absolute;
    left: 50%;
    top: -1px;
    transform: translateX(-50%);
    width: 1px;
    height: 20px;
    background: rgba(0,0,0,0.4);
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #666;
    margin-top: 3px;
}

/* Spot Metal Cards */
.spot-flex {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.spot-card {
    flex: 1 1 200px;
    min-width: 150px;
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 16px 16px;
    text-align: center;
}
.spot-price {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', monospace;
    margin: 4px 0;
}
.spot-change {
    font-size: 0.88rem;
    font-weight: 600;
}

/* Reduce gap between chart and first factor section */
[data-testid="stElementContainer"]:has([data-testid="stIFrame"]) {
    margin-bottom: -2rem !important;
}

/* ══════════════════════════════════════════════════════════════════════════════
   FACTOR TABLES
   ══════════════════════════════════════════════════════════════════════════════ */
.factor-table-wrap {
    width: 100%;
    overflow-x: auto;
    margin-bottom: 1.5rem;
}
.factor-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    table-layout: fixed;
}
.factor-table colgroup .col-indicator { width: 12%; }
.factor-table colgroup .col-ticker { width: 10%; }
.factor-table colgroup .col-value { width: 9%; }
.factor-table colgroup .col-chg { width: 10%; }
.factor-table colgroup .col-spark { width: 13%; }
.factor-table colgroup .col-mean { width: 9%; }
.factor-table colgroup .col-zscore { width: 7%; }
.factor-table colgroup .col-pct { width: 7%; }
.factor-table colgroup .col-signal { width: 10%; }
.factor-table colgroup .col-score { width: 13%; }
.factor-table th {
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 4px 3px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #333;
    font-size: 0.805rem;
    overflow: hidden;
    text-overflow: ellipsis;
    word-break: break-word;
}
.factor-table td {
    padding: 4px 3px;
    border-bottom: 1px solid #2a2a3a;
    vertical-align: top;
    font-size: 0.855rem;
    word-break: break-word;
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
    width: 340px;
    max-width: 85vw;
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
.val-unit { font-size: 0.6rem; color: #666; font-weight: 400; font-family: -apple-system, sans-serif; margin-top: 1px; }
.chg-cell { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.805rem; font-weight: 600; }
.chg-green { color: #22c55e; }
.chg-red { color: #ef4444; }
.chg-gray { color: #888; }
.spark-cell { padding: 2px 2px !important; vertical-align: middle; }
.spark-cell svg { display: block; width: 100%; height: auto; }

/* ══════════════════════════════════════════════════════════════════════════════
   MOBILE BREAKPOINTS
   ══════════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    /* Summary panel */
    .summary-panel {
        padding: 16px 14px;
        border-radius: 8px;
        margin-bottom: 14px;
    }
    .summary-flex {
        flex-direction: column;
        align-items: center;
        gap: 16px;
    }
    .summary-score-block {
        min-width: unset;
        width: 100%;
    }
    .summary-score-value {
        font-size: 2.6rem;
    }
    .summary-interp {
        font-size: 0.92rem;
    }
    .summary-chg {
        font-size: 0.82rem;
    }
    .summary-breakdown {
        min-width: unset;
        width: 100%;
    }
    .summary-table {
        font-size: 0.78rem;
    }

    /* Targets (predictions + pivots) */
    .targets-flex {
        flex-direction: column;
        gap: 14px;
        padding-top: 10px;
    }
    .targets-col {
        min-width: unset;
        width: 100%;
    }

    /* Gauge */
    .gauge-labels {
        font-size: 0.62rem;
    }

    /* Spot cards */
    .spot-flex {
        gap: 10px;
    }
    .spot-card {
        min-width: calc(50% - 8px);
        flex: 1 1 calc(50% - 8px);
        padding: 12px 10px;
        border-radius: 8px;
    }
    .spot-price {
        font-size: 1.3rem;
    }
    .spot-change {
        font-size: 0.78rem;
    }

    /* Factor tables — compact for mobile */
    .factor-table-wrap {
        margin-left: -0.5rem;
        margin-right: -0.5rem;
        padding: 0 0.5rem;
        overflow-x: visible;
    }
    .factor-table {
        font-size: 0.7rem;
        table-layout: auto;
    }
    .factor-table th {
        padding: 4px 4px;
        font-size: 0.65rem;
    }
    .factor-table td {
        padding: 4px 4px;
    }
    .val-cell {
        font-size: 0.78rem;
    }
    .chg-cell {
        font-size: 0.72rem;
    }
    .spark-cell {
        min-width: 60px;
        max-width: 90px;
    }

    /* Tooltips on mobile: wider for readability */
    .ind-cell .tooltip-box {
        width: 280px;
        left: -10px;
        font-size: 0.76rem;
    }
}

/* Extra-small phones (iPhone SE etc.) */
@media (max-width: 400px) {
    .summary-score-value {
        font-size: 2.2rem;
    }
    .spot-card {
        min-width: 100%;
        flex: 1 1 100%;
    }
    .spot-price {
        font-size: 1.2rem;
    }
    .gauge-labels span:nth-child(2) {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)


def render_category_table(cat_df):
    """Render a category table as HTML with hover tooltips on indicator names and sparklines."""
    cols = ["Indicator", "Ticker / Source", "Value", "Change", "Sparkline", "5Y Mean", "Z-Score",
            "Percentile", "Colour Indicator", "Overall Contribution"]

    header_labels = {
        "Colour Indicator": "Signal",
        "Overall Contribution": "Score",
        "Sparkline": "50D Trend",
        "Change": "Chg"
    }

    col_classes = ["col-indicator", "col-ticker", "col-value", "col-chg", "col-spark",
                   "col-mean", "col-zscore", "col-pct", "col-signal", "col-score"]
    html = '<div class="factor-table-wrap"><table class="factor-table"><colgroup>'
    for cc in col_classes:
        html += f'<col class="{cc}">'
    html += '</colgroup><tr>'
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
                label = {"Green": "▲ Bull", "Red": "▼ Bear", "Yellow": "● Neut"}.get(val, val)
                html += f'<td class="{css}">{label}</td>'

            elif c == "Overall Contribution":
                fval = float(val) if val != "N/A" else 0
                css = "score-pos" if fval > 0 else ("score-neg" if fval < 0 else "score-zero")
                html += f'<td class="{css}">{fval:+.1f}</td>'

            elif c == "Value":
                unit_raw = row.get("Unit", "")
                # Build a short descriptor tag from the unit
                unit_descriptors = {
                    "%": "rate %", "% GDP": "% of GDP", "% ann.": "annualized %",
                    "pp": "percentage pts", "$B": "USD billions", "$B/yr": "USD B / year",
                    "$T": "USD trillions", "$M/pt": "USD M / point", "$M/mo": "USD M / month",
                    "$/oz": "USD per troy oz", "pts": "index points", "vol pts": "volatility pts",
                    "bps": "basis points", "x": "ratio multiple", "ratio": "price ratio",
                    "contracts": "net contracts", "M oz": "million troy oz", "oz": "troy ounces",
                    "tonnes": "metric tonnes", "tonnes/yr": "tonnes / year",
                    "bbl/oz": "barrels per oz", "/2": "of 2 MAs", "0-100": "scale 0–100",
                    "index": "index value",
                }
                desc = unit_descriptors.get(unit_raw, unit_raw)
                html += f'<td class="val-cell">{esc(str(val))}'
                if desc:
                    html += f'<div class="val-unit">{esc(desc)}</div>'
                html += '</td>'

            else:
                html += f'<td>{esc(str(val))}</td>'

        html += '</tr>'

    html += '</table></div>'
    return html


for cat in final_df['Category'].unique():
    cat_df = final_df[final_df['Category'] == cat]

    subset_score = cat_df['Overall Contribution'].sum()
    c_color = "🟢" if subset_score > 0 else ("🔴" if subset_score < 0 else "🟡")

    st.subheader(f"{cat}  {c_color} (Subset Score: {subset_score:.1f})")
    st.markdown(render_category_table(cat_df), unsafe_allow_html=True)
