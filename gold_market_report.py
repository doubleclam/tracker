#!/usr/bin/env python3
"""
MI Metals Factors - Full 47-Factor Dynamic Engine
Calculates 5Y Means, Z-Scores, and Percentiles dynamically from historical API data.
"""

import ssl
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
import streamlit as st
from datetime import datetime, timedelta

# ─── Security & API Setup ────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
FRED_API_KEY = "412665086b998f7954423844843240b6"
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

st.set_page_config(page_title="MI Metals Factors", layout="wide")

# ─── Factor Configuration (All 47 Factors + Sub-factors) ─────────────────────
FACTOR_CONFIG = {
    "I. Macro-Monetary Drivers": [
        {"ind": "1. US Real Rates (10Y TIPS)", "ticker": "DFII10", "source": "FRED", "weight": 5, "higher_is_bullish": False, "why": "High real yield competes with metals."},
        {"ind": "1A. Fed Funds Real Rate", "ticker": "FEDFUNDS_REAL", "source": "SIMULATED", "weight": 4, "higher_is_bullish": False, "why": "Negative real rates are historically bullish."},
        {"ind": "1B. 5Y Real Rates", "ticker": "DFII5", "source": "FRED", "weight": 3, "higher_is_bullish": False, "why": "Impacts medium-term carrying costs."},
        {"ind": "1C. Global Real Rates (Weighted)", "ticker": "GLB_REAL", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Global opportunity cost."},
        {"ind": "2. Breakeven Inflation (5Y)", "ticker": "T5YIE", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Market-implied inflation expectation."},
        {"ind": "2A. UMich Inflation Expectations", "ticker": "MICH", "source": "FRED", "weight": 2, "higher_is_bullish": True, "why": "Consumer inflation sentiment."},
        {"ind": "2B. Cleveland Fed Nowcast", "ticker": "CLEV_INFL", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "High-frequency inflation indicator."},
        {"ind": "3. Global Liquidity (M2)", "ticker": "M2SL", "source": "FRED", "weight": 5, "higher_is_bullish": True, "why": "Expansions in M2 debase fiat currencies."},
        {"ind": "3A. US Debt/GDP Ratio", "ticker": "GFDEGDQ188S", "source": "FRED", "weight": 4, "higher_is_bullish": True, "why": "Unsustainable debt implies future monetization."},
        {"ind": "3B. Deficit as % of GDP", "ticker": "FYFSGDA188S", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Larger deficits (more negative) weaken currency, bullish for gold."},
        {"ind": "3C. Debt Service Costs", "ticker": "A091RC1Q027SBEA", "source": "FRED", "weight": 4, "higher_is_bullish": True, "why": "Forces central banks into yield curve control."},
        {"ind": "4. Nominal Yield Curve", "ticker": "T10Y2Y", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Steepening indicates normalization."},
        {"ind": "5. Currency Strength (DXY)", "ticker": "DX-Y.NYB", "source": "YF", "weight": 5, "higher_is_bullish": False, "why": "Inverse correlation to USD pricing."},
        {"ind": "6. Credit Spreads (HY)", "ticker": "BAMLH0A0HYM2", "source": "FRED", "weight": 3, "higher_is_bullish": False, "why": "Tight spreads reflect low systemic fear."}
    ],
    "II. Market Structure & Positioning": [
        {"ind": "7. Managed Money Position", "ticker": "COT_MM", "source": "SIMULATED", "weight": 4, "higher_is_bullish": False, "why": "Crowded longs risk sharp liquidations."},
        {"ind": "8. Swap Dealer Position", "ticker": "COT_SD", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Commercial positioning is the 'smart money'."},
        {"ind": "9. Open Interest", "ticker": "COMEX_OI", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Reflects total speculative participation."},
        {"ind": "10. Futures Structure", "ticker": "FUT_CURVE", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Backwardation indicates tight physical supply."},
        {"ind": "11. Options Skew", "ticker": "OPT_SKEW", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "Heavy call skew implies topside exhaustion."}
    ],
    "III. Physical Fundamentals & Flow": [
        {"ind": "12. Shanghai Premium", "ticker": "SGE_PREM", "source": "SIMULATED", "weight": 4, "higher_is_bullish": True, "why": "Indicates physical demand strength in Asia."},
        {"ind": "12A. Gold in Other Currencies", "ticker": "XAU_BASKET", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Broad cross-currency strength."},
        {"ind": "13. Indian Demand Premium", "ticker": "IN_PREM", "source": "SIMULATED", "weight": 4, "higher_is_bullish": True, "why": "Core cultural and seasonal demand indicator."},
        {"ind": "13A. India Wedding Season", "ticker": "IN_WEDDING", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Cyclical demand boosts."},
        {"ind": "13B. China Lunar New Year", "ticker": "CN_LNY", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Cyclical demand boosts."},
        {"ind": "14. ETF Flows", "ticker": "ETF_FLOWS", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Institutional and retail accumulation."},
        {"ind": "14A. US Mint Coin Sales", "ticker": "US_MINT", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Retail physical demand proxy."},
        {"ind": "14B. Retail Premium Over Spot", "ticker": "RETAIL_PREM", "source": "SIMULATED", "weight": 1, "higher_is_bullish": True, "why": "Dealer physical supply constraints."},
        {"ind": "15. COMEX Inventory", "ticker": "COMEX_INV", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Falling stocks signal physical market tightness."},
        {"ind": "15A. Scrap/Recycling Flows", "ticker": "SCRAP_FLOW", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "High scrap rates cap price rallies."},
        {"ind": "16. Central Bank Buying", "ticker": "CB_BUYING", "source": "SIMULATED", "weight": 5, "higher_is_bullish": True, "why": "Provides a structural floor to prices."}
    ],
    "IV. Supply Side & Mining Economics": [
        {"ind": "17. All-in Sustaining Costs", "ticker": "AISC", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Higher production costs create a price floor."},
        {"ind": "18. Mine Supply Growth", "ticker": "MINE_SUPPLY", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Peak gold dynamics restrict new supply."},
        {"ind": "19. Mining Capex Cycle", "ticker": "MINING_CAPEX", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Underinvestment limits future mine output."},
        {"ind": "20. Miner Hedging Activity", "ticker": "HEDGING", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Forward selling suppresses spot prices."},
        {"ind": "21. Reserve Depletion", "ticker": "RESERVES", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Falling high-grade reserves forces premiums."}
    ],
    "V. Banking System & Liquidity": [
        {"ind": "22. TED Spread", "ticker": "TEDRATE", "source": "FRED", "weight": 2, "higher_is_bullish": False, "why": "Measures perceived credit risk in banking."},
        {"ind": "23. SOFR-OIS Spread", "ticker": "SOFR_OIS", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Interbank liquidity stress."},
        {"ind": "24. Fed Reverse Repo Usage", "ticker": "RRPONTSYD", "source": "FRED", "weight": 3, "higher_is_bullish": False, "why": "Drains excess liquidity from the system."},
        {"ind": "25. Commercial Bank Reserves", "ticker": "TOTRESNS", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Base money supply availability."},
        {"ind": "26. Basel III NSFR Impact", "ticker": "NSFR_IMPACT", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Regulatory unallocated gold constraints."}
    ],
    "VI. Ratios & Relative Value": [
        {"ind": "27. Gold/Silver Ratio", "ticker": "GC=F/SI=F", "source": "YF_RATIO", "weight": 3, "higher_is_bullish": False, "why": "High ratio historically reverts."},
        {"ind": "27A. Gold vs Bitcoin", "ticker": "GC=F/BTC-USD", "source": "YF_RATIO", "weight": 1, "higher_is_bullish": False, "why": "Alternative store of value competition."},
        {"ind": "27B. Gold vs Real Estate", "ticker": "GC=F/VNQ", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": True, "why": "Hard asset relative strength."},
        {"ind": "27C. Gold as % of Global Assets", "ticker": "AU_ALLOCATION", "source": "SIMULATED", "weight": 3, "higher_is_bullish": False, "why": "Under-allocation signals room for institutional buying."},
        {"ind": "28. Gold/Oil Ratio", "ticker": "GC=F/CL=F", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": True, "why": "Purchasing power of gold vs energy."},
        {"ind": "29. Gold/S&P 500", "ticker": "GC=F/^GSPC", "source": "YF_RATIO", "weight": 3, "higher_is_bullish": True, "why": "Relative performance against risk assets."},
        {"ind": "30. Platinum/Gold", "ticker": "PL=F/GC=F", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": False, "why": "Industrial cycle divergences."},
        {"ind": "31. Gold/M2 Ratio", "ticker": "AU_M2", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Value of gold relative to fiat expansion."}
    ],
    "VII. Technical Indicators": [
        {"ind": "32. Moving Averages", "ticker": "MA_BULL", "source": "SIMULATED", "weight": 4, "higher_is_bullish": True, "why": "Trend confirmation."},
        {"ind": "33. Bollinger Bands", "ticker": "BB_EXP", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Volatility envelope extension."},
        {"ind": "34. RSI Divergence", "ticker": "RSI_DIV", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Momentum exhaustion indicator."},
        {"ind": "35. Volume Profile", "ticker": "VOL_PROF", "source": "SIMULATED", "weight": 1, "higher_is_bullish": True, "why": "Support level consolidation."}
    ],
    "VIII. Black Swan & Exogenous Risks": [
        {"ind": "36. Equity and Gold Volatility", "ticker": "^VIX", "source": "YF", "weight": 2, "higher_is_bullish": True, "why": "Spikes trigger safe-haven capital flight."},
        {"ind": "37. Geopolitical Risk", "ticker": "GPR_IDX", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Reflects safe-haven tail risk premiums."},
        {"ind": "38. US CDS", "ticker": "US_CDS", "source": "SIMULATED", "weight": 1, "higher_is_bullish": True, "why": "Sovereign default risk proxies."},
        {"ind": "39. Lease Rates", "ticker": "LEASE_RATE", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Physical market leasing stress."},
        {"ind": "40. Dealer Gamma", "ticker": "DLR_GAMMA", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "Market maker positioning impacts volatility."},
        {"ind": "41. EFP Spread", "ticker": "EFP_SPREAD", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Paper vs Physical arbitrage dislocations."},
        {"ind": "42. Miners vs Metal", "ticker": "GDX/GLD", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": True, "why": "Miners lead the metal in bull markets."},
        {"ind": "43. Junior Speculation", "ticker": "GDXJ/GDX", "source": "YF_RATIO", "weight": 1, "higher_is_bullish": True, "why": "Risk-on appetite in the precious metals sector."},
        {"ind": "44. Silver Beta", "ticker": "SI_BETA", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Silver outperforming signals broad bull market."}
    ],
    "IX. Sentiment & Retail Indicators": [
        {"ind": "45. Google Trends - Buy Gold", "ticker": "GTRENDS", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Retail euphoria often marks local tops."},
        {"ind": "46. Gold Advertising Spending", "ticker": "AD_SPEND", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "Late-cycle retail trapping indicator."},
        {"ind": "47. Jewelry Demand", "ticker": "JEWELRY", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Base consumer physical demand floor."}
    ]
}

# ─── Dynamic Historical Data Engine ──────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_historical_data():
    """Fetches 5 years of daily/monthly history to calculate true Z-Scores for all 47 factors"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    hist_data = {}
    fetch_errors = []

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

    # 2. Fetch Yahoo Finance Data
    yf_tickers = [
        "DX-Y.NYB", "^VIX", "GC=F", "SI=F", "PL=F", "CL=F",
        "^GSPC", "BTC-USD", "VNQ", "GDX", "GLD", "GDXJ",
        "PA=F", "HG=F"
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

def calculate_statistics(series, config):
    """Dynamically calculates 5Y Mean, Std, Z-Score, and Confidence Scores"""
    if series is None or len(series) < 10:
        return {"Value": "N/A", "Colour Indicator": "Yellow", "Total Factor Score": 0}

    current_val = float(series.iloc[-1])
    mean_5y = float(series.mean())
    std_5y = float(series.std())

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
        "Value": f"{current_val:.2f}",
        "5Y Mean": f"{mean_5y:.2f}",
        "5Y Std": f"{std_5y:.2f}",
        "Z-Score": f"{z_score:.2f}",
        "Percentile": f"{percentile:.0f}%",
        "Total Factor Score": round(factor_score, 1),
        "Colour Indicator": color
    }

# ─── Dashboard Execution ─────────────────────────────────────────────────────

st.title("📊 MI Metals Factors")
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
            if item['source'] == 'SIMULATED' and ticker not in [
                "FEDFUNDS_REAL", "AU_M2", "SI_BETA", "MA_BULL", "RSI_DIV", "BB_EXP"
            ]:
                source_label = f"{ticker} (Simulated)"
            elif series is not None and len(series) >= 10:
                source_label = f"{ticker} ({item['source']})"
            else:
                source_label = f"{ticker} (No Data)"

            entry = {
                "Category": category,
                "Indicator": item['ind'],
                "Ticker / Source": source_label,
                "Value": stats["Value"],
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

# ─── Overall Confidence & Scoring Calculations ───────────────────────────────

overall_score = final_df['Total Factor Score'].sum()
total_factors = len(final_df)
decisive_signals = len(final_df[final_df['Colour Indicator'].isin(['Green', 'Red'])])
confidence_pct = (decisive_signals / total_factors) * 100 if total_factors > 0 else 0
overall_bias = "Bullish" if overall_score > 10 else ("Bearish" if overall_score < -10 else "Neutral")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall System Bias", overall_bias)
col2.metric("Aggregate Score (Dynamic)", f"{overall_score:.1f}")
col3.metric("Level of Confidence", f"{confidence_pct:.1f}%")
col4.metric("Live / Simulated Factors", f"{total_factors - simulated_count} / {simulated_count}")

# Clamp progress to [0.0, 1.0]
st.progress(min(max(confidence_pct / 100.0, 0.0), 1.0))
st.divider()

# Show fetch errors in an expander if any
if fetch_errors:
    with st.expander(f"⚠️ Data Fetch Issues ({len(fetch_errors)})", expanded=False):
        for err in fetch_errors:
            st.text(f"  • {err}")

def style_colour_indicator(val):
    colors = {'Green': '#00FF00', 'Red': '#FF0000', 'Yellow': '#FFD700'}
    return f'color: {colors.get(val, "white")}; font-weight: bold;' if val in colors else ''

for cat in final_df['Category'].unique():
    cat_df = final_df[final_df['Category'] == cat]

    subset_score = cat_df['Total Factor Score'].sum()
    c_color = "🟢" if subset_score > 0 else ("🔴" if subset_score < 0 else "🟡")

    st.subheader(f"{cat}  {c_color} (Subset Score: {subset_score:.1f})")

    display_df = cat_df.drop(columns=['Category'])
    st.dataframe(
        display_df.style.map(style_colour_indicator, subset=['Colour Indicator']),
        use_container_width=True, hide_index=True
    )
