#!/usr/bin/env python3
"""
MI Metals Factors - Full 47-Factor Dynamic Engine
Calculates 5Y Means, Z-Scores, and Percentiles dynamically from historical API data.
"""

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

st.set_page_config(page_title="MI Metals Factors", layout="wide")

# ─── Factor Configuration (All 47 Factors + Sub-factors) ─────────────────────
# Each factor has:
#   ind            - display name
#   ticker         - data series identifier
#   source         - FRED / YF / YF_RATIO / SIMULATED
#   weight         - importance weight (1-5)
#   higher_is_bullish - directional interpretation for scoring
#   why            - short reason why it matters for gold
#   definition     - full plain-English explanation (shown on hover)
#   unit           - measurement unit appended to the Value column

FACTOR_CONFIG = {
    "I. Macro-Monetary Drivers": [
        {"ind": "1. US Real Rates (10Y TIPS)", "ticker": "DFII10", "source": "FRED", "weight": 5, "higher_is_bullish": False, "why": "High real yield competes with metals.",
         "definition": "The yield on 10-Year Treasury Inflation-Protected Securities (TIPS). Represents the real (after-inflation) return investors demand for lending to the US government for 10 years. When positive, bonds offer a real return that competes with non-yielding gold. When negative, holding gold has zero opportunity cost vs bonds.",
         "unit": "%"},
        {"ind": "1A. Fed Funds Real Rate", "ticker": "FEDFUNDS_REAL", "source": "SIMULATED", "weight": 4, "higher_is_bullish": False, "why": "Negative real rates are historically bullish.",
         "definition": "The effective Federal Funds Rate minus the 5-Year Breakeven Inflation Rate. Measures whether the Fed's policy rate is above or below expected inflation. Deeply negative real Fed Funds rates (like 2020-2022) historically coincide with gold bull markets because cash loses purchasing power.",
         "unit": "%"},
        {"ind": "1B. 5Y Real Rates", "ticker": "DFII5", "source": "FRED", "weight": 3, "higher_is_bullish": False, "why": "Impacts medium-term carrying costs.",
         "definition": "The yield on 5-Year TIPS. A shorter-duration measure of real interest rates than the 10Y, more sensitive to near-term Fed policy expectations. Reflects the medium-term opportunity cost of holding gold versus inflation-protected bonds.",
         "unit": "%"},
        {"ind": "1C. Global Real Rates (Weighted)", "ticker": "GLB_REAL", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Global opportunity cost.",
         "definition": "A GDP-weighted composite of real policy rates across major economies (US, Eurozone, Japan, UK, China). Gold is priced globally, so if real rates are negative worldwide, the bid for gold is structurally stronger than if only one country has low rates.",
         "unit": "%"},
        {"ind": "2. Breakeven Inflation (5Y)", "ticker": "T5YIE", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Market-implied inflation expectation.",
         "definition": "The difference between the 5-Year nominal Treasury yield and the 5-Year TIPS yield. This spread represents the bond market's consensus expectation for average annual inflation over the next 5 years. Higher breakevens signal rising inflation fears, which drive demand for gold as an inflation hedge.",
         "unit": "%"},
        {"ind": "2A. UMich Inflation Expectations", "ticker": "MICH", "source": "FRED", "weight": 2, "higher_is_bullish": True, "why": "Consumer inflation sentiment.",
         "definition": "The University of Michigan Survey of Consumers' median expectation for inflation over the next 12 months. Unlike breakevens (market-based), this captures household sentiment. Spikes in consumer inflation expectations often precede retail gold buying waves and can become self-fulfilling.",
         "unit": "%"},
        {"ind": "2B. Cleveland Fed Nowcast", "ticker": "CLEV_INFL", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "High-frequency inflation indicator.",
         "definition": "The Cleveland Fed's real-time estimate of current-month inflation using Treasury yields, inflation swaps, and survey data. Provides a more timely read than CPI (which is released with a lag). Persistent above-target readings support the inflation-hedge narrative for gold.",
         "unit": "%"},
        {"ind": "3. Global Liquidity (M2)", "ticker": "M2SL", "source": "FRED", "weight": 5, "higher_is_bullish": True, "why": "Expansions in M2 debase fiat currencies.",
         "definition": "The US M2 money supply: cash, checking deposits, savings, money market funds, and small CDs. Used as a proxy for global liquidity since the USD is the world reserve currency. When M2 expands rapidly (like 2020-2021's 40% surge), more dollars chase finite gold ounces, driving prices higher.",
         "unit": "$B"},
        {"ind": "3A. US Debt/GDP Ratio", "ticker": "GFDEGDQ188S", "source": "FRED", "weight": 4, "higher_is_bullish": True, "why": "Unsustainable debt implies future monetization.",
         "definition": "Total US federal government debt as a percentage of GDP. When this ratio rises above ~100%, markets begin pricing in the risk that the government will need to inflate away its debt (financial repression), rather than repay it through growth or austerity. This structural debasement risk underpins long-term gold demand.",
         "unit": "% GDP"},
        {"ind": "3B. Deficit as % of GDP", "ticker": "FYFSGDA188S", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Larger deficits (more negative) weaken currency, bullish for gold.",
         "definition": "The annual US federal budget deficit expressed as a percentage of GDP (shown as positive = deficit size). Large fiscal deficits mean the government is spending far more than it collects in taxes, which must be financed by borrowing or money creation. Persistent 5%+ deficits outside recessions signal structural fiscal deterioration.",
         "unit": "% GDP"},
        {"ind": "3C. Debt Service Costs", "ticker": "A091RC1Q027SBEA", "source": "FRED", "weight": 4, "higher_is_bullish": True, "why": "Forces central banks into yield curve control.",
         "definition": "Federal government interest payments on outstanding debt, in billions per quarter (annualized). When interest costs consume a large share of tax revenue, the Fed faces pressure to keep rates artificially low (financial repression / yield curve control) to prevent a debt spiral. This environment is extremely bullish for gold.",
         "unit": "$B/yr"},
        {"ind": "4. Nominal Yield Curve", "ticker": "T10Y2Y", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Steepening indicates normalization.",
         "definition": "The spread between the 10-Year and 2-Year US Treasury yields. A positive (steep) curve means long-term rates are higher than short-term rates, which is normal. An inverted curve (negative spread) signals recession expectations. A steepening from inversion is bullish for gold as it often precedes rate cuts and easing cycles.",
         "unit": "pp"},
        {"ind": "5. Currency Strength (DXY)", "ticker": "DX-Y.NYB", "source": "YF", "weight": 5, "higher_is_bullish": False, "why": "Inverse correlation to USD pricing.",
         "definition": "The US Dollar Index, measuring the dollar against a basket of 6 major currencies (EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%). Since gold is priced in USD, a weaker dollar mechanically makes gold cheaper for foreign buyers, boosting demand. The DXY-gold correlation is roughly -0.8 over long periods.",
         "unit": "pts"},
        {"ind": "6. Credit Spreads (HY)", "ticker": "BAMLH0A0HYM2", "source": "FRED", "weight": 3, "higher_is_bullish": False, "why": "Tight spreads reflect low systemic fear.",
         "definition": "The ICE BofA US High Yield Master II Option-Adjusted Spread: the extra yield (in percentage points) that junk bond investors demand over Treasuries. Tight spreads (<3.5%) signal complacency and risk-on sentiment. Blowouts (>6%) signal credit stress and flight-to-safety flows into gold.",
         "unit": "%"}
    ],
    "II. Market Structure & Positioning": [
        {"ind": "7. Managed Money Position", "ticker": "COT_MM", "source": "CFTC", "weight": 4, "higher_is_bullish": False, "why": "Crowded longs risk sharp liquidations.",
         "definition": "Net long/short position of Managed Money traders (hedge funds, CTAs) in COMEX gold futures, from the CFTC Commitments of Traders report. Extreme net long positioning (Z-score > 2.0) signals a crowded trade vulnerable to sharp liquidation. Extreme short positioning signals contrarian bullish potential.",
         "unit": "contracts"},
        {"ind": "8. Swap Dealer Position", "ticker": "COT_SD", "source": "CFTC", "weight": 3, "higher_is_bullish": True, "why": "Commercial positioning is the 'smart money'.",
         "definition": "Net position of Swap Dealers (major banks facilitating OTC gold transactions) in COMEX gold futures. These entities are considered 'smart money' because they have superior information about physical flows. When swap dealers reduce their typical net short, it signals reduced hedging demand or outright bullishness.",
         "unit": "contracts"},
        {"ind": "9. Open Interest", "ticker": "COMEX_OI", "source": "CFTC", "weight": 2, "higher_is_bullish": True, "why": "Reflects total speculative participation.",
         "definition": "Total number of outstanding (unsettled) gold futures contracts on COMEX. Rising OI with rising prices confirms new money entering bullish bets. Rising OI with falling prices signals aggressive new shorts. Falling OI indicates position liquidation regardless of price direction.",
         "unit": "contracts"},
        {"ind": "10. Futures Structure", "ticker": "FUT_CURVE", "source": "YF_CALC", "weight": 2, "higher_is_bullish": True, "why": "Backwardation indicates tight physical supply.",
         "definition": "The shape of the gold futures term structure, measured as the spread between front-month COMEX gold futures (GC=F) and a spot proxy (GLD ETF x conversion factor). Positive = contango (normal), negative = backwardation (rare, bullish). Backwardation signals extreme physical demand or delivery stress.",
         "unit": "$/oz"},
        {"ind": "11. Options Skew", "ticker": "OPT_SKEW", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "Heavy call skew implies topside exhaustion.",
         "definition": "The 25-Delta Risk Reversal: the implied volatility of 25-delta calls minus 25-delta puts on gold options. Positive values (call premium) mean the market is paying more for upside protection, suggesting bullish consensus is already priced in. Negative values (put premium) suggest fear and hedging activity.",
         "unit": "vol pts"}
    ],
    "III. Physical Fundamentals & Flow": [
        {"ind": "12. Shanghai Premium", "ticker": "SGE_PREM", "source": "SGE_API", "weight": 4, "higher_is_bullish": True, "why": "Indicates physical demand strength in Asia.",
         "definition": "The premium (or discount) of gold on the Shanghai Gold Exchange versus COMEX, in $/oz. Calculated from the SGE daily benchmark price (CNY/gram) converted to USD/oz via the CNY/USD exchange rate, minus the COMEX front-month futures price. Positive premiums signal strong Chinese physical demand.",
         "unit": "$/oz"},
        {"ind": "12A. Gold in Other Currencies", "ticker": "XAU_BASKET", "source": "YF_CALC", "weight": 3, "higher_is_bullish": True, "why": "Broad cross-currency strength.",
         "definition": "Gold's performance measured against a basket of non-USD currencies (EUR, GBP, JPY, CNY). Calculated as the equal-weighted average of gold priced in each currency via YF FX rates. When gold makes new highs in most currencies simultaneously, it signals genuine global repricing rather than just USD weakness.",
         "unit": "index"},
        {"ind": "13. Indian Demand Premium", "ticker": "IN_PREM", "source": "SIMULATED", "weight": 4, "higher_is_bullish": True, "why": "Core cultural and seasonal demand indicator.",
         "definition": "The premium or discount at which gold trades in India versus international benchmarks. India is the world's second-largest gold consumer. Premiums indicate strong local demand (often seasonal around Diwali and wedding season). Discounts signal weak demand or import restrictions.",
         "unit": "$/oz"},
        {"ind": "13A. India Wedding Season", "ticker": "IN_WEDDING", "source": "CALENDAR", "weight": 2, "higher_is_bullish": True, "why": "Cyclical demand boosts.",
         "definition": "A seasonal indicator tracking India's wedding season intensity. Score 100 = peak season (Oct-Feb, especially Diwali and post-harvest weddings), 50 = shoulder (Mar, Sep), 25 = off-season (Apr-Aug). Indian weddings drive enormous gold jewelry demand, adding 100-200 tonnes of incremental annual demand during peak months.",
         "unit": "0-100"},
        {"ind": "13B. China Lunar New Year", "ticker": "CN_LNY", "source": "CALENDAR", "weight": 2, "higher_is_bullish": True, "why": "Cyclical demand boosts.",
         "definition": "A seasonal indicator for Chinese gold demand. Score 100 = peak gifting/buying season (Jan-Feb around Lunar New Year and Golden Week in Oct), 50 = shoulder months, 25 = off-season. Gold gifting is a deeply embedded cultural tradition; demand typically surges 4-6 weeks before the holiday.",
         "unit": "0-100"},
        {"ind": "14. ETF Flows (GLD Volume)", "ticker": "ETF_FLOWS", "source": "YF_CALC", "weight": 3, "higher_is_bullish": True, "why": "Institutional and retail accumulation.",
         "definition": "20-day average daily dollar volume of the SPDR Gold Trust (GLD), the largest physically-backed gold ETF. Rising volume indicates increasing institutional and retail interest in gold exposure. Used as a proxy for ETF flow momentum since actual tonnage flow data requires paid subscriptions.",
         "unit": "$B"},
        {"ind": "14A. US Mint Coin Sales", "ticker": "US_MINT", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Retail physical demand proxy.",
         "definition": "Monthly sales of American Gold Eagle and American Gold Buffalo coins by the US Mint. A direct measure of retail/individual investor demand for physical gold. Spikes in coin sales (like March 2020) often coincide with financial stress or loss of confidence in the monetary system.",
         "unit": "oz"},
        {"ind": "14B. Retail Premium Over Spot", "ticker": "RETAIL_PREM", "source": "SIMULATED", "weight": 1, "higher_is_bullish": True, "why": "Dealer physical supply constraints.",
         "definition": "The percentage premium that retail gold products (coins, small bars) trade above the spot price. Normal premiums are 3-5%. Premiums above 8-10% indicate physical supply constraints at the retail level, often a sign of panic buying or minting bottlenecks.",
         "unit": "%"},
        {"ind": "15. COMEX Inventory", "ticker": "COMEX_INV", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Falling stocks signal physical market tightness.",
         "definition": "Total registered (eligible for delivery) gold inventory in COMEX-approved warehouses, measured in troy ounces. Falling registered inventory means physical gold is being withdrawn for delivery rather than remaining as exchange collateral. Rapid drawdowns signal that paper claims are being converted to physical metal.",
         "unit": "M oz"},
        {"ind": "15A. Scrap/Recycling Flows", "ticker": "SCRAP_FLOW", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "High scrap rates cap price rallies.",
         "definition": "The volume of recycled gold entering the market from old jewelry, electronics, and dental scrap. High prices incentivize consumers and recyclers to sell, creating a natural price ceiling. Scrap supply typically represents 25-30% of annual gold supply.",
         "unit": "tonnes"},
        {"ind": "16. Central Bank Buying", "ticker": "CB_BUYING", "source": "SIMULATED", "weight": 5, "higher_is_bullish": True, "why": "Provides a structural floor to prices.",
         "definition": "Net gold purchases by central banks worldwide, reported quarterly by the World Gold Council. Since 2010, central banks have been net buyers, with purchases accelerating to 1,000+ tonnes/year in 2022-2023. This represents a structural shift in reserve management away from USD assets and toward gold.",
         "unit": "tonnes/yr"}
    ],
    "IV. Supply Side & Mining Economics": [
        {"ind": "17. All-in Sustaining Costs", "ticker": "AISC", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Higher production costs create a price floor.",
         "definition": "The All-In Sustaining Cost per ounce to produce gold, including mining, processing, refining, sustaining capex, corporate G&A, and exploration. Published quarterly by mining companies. The global average AISC (~$1,300-1,400/oz in 2024) acts as a price floor: below this, mines shut down, reducing supply.",
         "unit": "$/oz"},
        {"ind": "18. Mine Supply Growth", "ticker": "MINE_SUPPLY", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Peak gold dynamics restrict new supply.",
         "definition": "Year-over-year growth in global gold mine production, in tonnes. Mine supply has plateaued at ~3,600 tonnes/year as new discoveries decline and ore grades fall. Flat-to-declining supply growth combined with rising demand creates a structural deficit.",
         "unit": "tonnes/yr"},
        {"ind": "19. Mining Capex Cycle", "ticker": "MINING_CAPEX", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Underinvestment limits future mine output.",
         "definition": "Aggregate capital expenditure by major gold miners on new mine development, expansion, and exploration. After the 2013-2015 downturn, the industry drastically cut capex. It takes 10-15 years from discovery to first production, so today's underinvestment guarantees constrained supply in the late 2020s and 2030s.",
         "unit": "$B/yr"},
        {"ind": "20. Miner Hedging Activity", "ticker": "HEDGING", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Forward selling suppresses spot prices.",
         "definition": "The total volume of gold production that miners have sold forward in futures/options markets. Heavy hedging (forward selling) locks in current prices but creates selling pressure in the futures market. Low hedging signals miner confidence that prices will rise, and reduces overhead supply pressure.",
         "unit": "M oz"},
        {"ind": "21. Reserve Depletion", "ticker": "RESERVES", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Falling high-grade reserves forces premiums.",
         "definition": "Total global proven and probable gold reserves in the ground, measured in million ounces. The gold mining industry has struggled to replace depleted reserves through new discoveries. Declining reserve life (years of reserves remaining at current production rates) implies future scarcity premiums.",
         "unit": "M oz"}
    ],
    "V. Banking System & Liquidity": [
        {"ind": "22. TED Spread", "ticker": "TEDRATE", "source": "FRED", "weight": 2, "higher_is_bullish": False, "why": "Measures perceived credit risk in banking.",
         "definition": "The spread between 3-Month LIBOR (interbank lending rate) and the 3-Month Treasury bill yield. Widening TED spread signals that banks are charging each other more to lend, reflecting rising counterparty risk fears. Spikes (like 2008's 4.5%) trigger safe-haven flows into gold. Note: LIBOR was discontinued in 2023; SOFR-based spreads are the modern equivalent.",
         "unit": "%"},
        {"ind": "23. SOFR-OIS Spread", "ticker": "SOFR_OIS", "source": "FRED_CALC", "weight": 2, "higher_is_bullish": False, "why": "Interbank liquidity stress.",
         "definition": "The spread between SOFR (Secured Overnight Financing Rate) and the effective Federal Funds Rate (OIS proxy), in basis points. Both from FRED. The modern replacement for the TED Spread after LIBOR's discontinuation. Widening signals stress in short-term secured funding markets.",
         "unit": "bps"},
        {"ind": "24. Fed Reverse Repo Usage", "ticker": "RRPONTSYD", "source": "FRED", "weight": 3, "higher_is_bullish": False, "why": "Drains excess liquidity from the system.",
         "definition": "The total dollar amount parked overnight at the Federal Reserve's Reverse Repo Facility (RRP). Money market funds and banks use the RRP to earn a risk-free return. High RRP usage (>$2T in 2022-2023) means excess liquidity is being drained from the financial system. As the RRP drains toward zero, that liquidity re-enters markets.",
         "unit": "$T"},
        {"ind": "25. Commercial Bank Reserves", "ticker": "TOTRESNS", "source": "FRED", "weight": 3, "higher_is_bullish": True, "why": "Base money supply availability.",
         "definition": "Total reserves held by commercial banks at the Federal Reserve, in billions. These reserves are the foundation of the money multiplier: banks lend against them. Rising reserves mean the banking system has ample liquidity for credit creation. When reserves fall below a critical threshold (~$3T), funding stress emerges.",
         "unit": "$B"},
        {"ind": "26. Basel III NSFR Impact", "ticker": "NSFR_IMPACT", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Regulatory unallocated gold constraints.",
         "definition": "The estimated impact of Basel III's Net Stable Funding Ratio (NSFR) requirement on bank gold trading. Under Basel III, unallocated gold (paper gold held on bank balance sheets) requires 85% stable funding backing, making it far more expensive for banks to maintain. This forces a shift from paper to physical gold, tightening the physical market.",
         "unit": "index"}
    ],
    "VI. Ratios & Relative Value": [
        {"ind": "27. Gold/Silver Ratio", "ticker": "GC=F/SI=F", "source": "YF_RATIO", "weight": 3, "higher_is_bullish": False, "why": "High ratio historically reverts.",
         "definition": "The number of ounces of silver it takes to buy one ounce of gold. Historical average is ~60-65x. Above 80x signals silver is extremely cheap relative to gold (and precious metals may be underowned). Below 50x signals silver overvaluation or late-stage bull market euphoria. The ratio tends to mean-revert violently.",
         "unit": "x"},
        {"ind": "27A. Gold vs Bitcoin", "ticker": "GC=F/BTC-USD", "source": "YF_RATIO", "weight": 1, "higher_is_bullish": False, "why": "Alternative store of value competition.",
         "definition": "Gold price divided by Bitcoin price, showing gold's value in BTC terms. A rising ratio means gold is outperforming Bitcoin as a store of value. A falling ratio signals capital rotating from gold into crypto. This rivalry is most relevant for younger institutional allocators choosing between 'digital gold' and physical gold.",
         "unit": "ratio"},
        {"ind": "27B. Gold vs Real Estate", "ticker": "GC=F/VNQ", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": True, "why": "Hard asset relative strength.",
         "definition": "Gold price divided by the Vanguard Real Estate ETF (VNQ) price. Measures gold's purchasing power relative to real estate assets. A rising ratio means gold is outperforming property, often during periods of financial stress when real estate is under pressure but gold benefits from safe-haven flows.",
         "unit": "ratio"},
        {"ind": "27C. Gold as % of Global Assets", "ticker": "AU_ALLOCATION", "source": "SIMULATED", "weight": 3, "higher_is_bullish": False, "why": "Under-allocation signals room for institutional buying.",
         "definition": "Gold's total market capitalization as a percentage of total global financial assets (~$500T including equities, bonds, real estate). Currently around 1-2%. At past cycle peaks (1980, 2011), gold's share reached 3-5%. Even a small reallocation from bonds to gold by pension funds and sovereign wealth funds would require enormous physical purchases.",
         "unit": "%"},
        {"ind": "28. Gold/Oil Ratio", "ticker": "GC=F/CL=F", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": True, "why": "Purchasing power of gold vs energy.",
         "definition": "The number of barrels of oil that one ounce of gold can buy. Long-term average is ~15-20 barrels. A high ratio (>25) means gold is expensive vs energy, often reflecting safe-haven demand. An extremely low ratio (<10) signals gold is undervalued relative to the real economy's energy needs.",
         "unit": "bbl/oz"},
        {"ind": "29. Gold/S&P 500", "ticker": "GC=F/^GSPC", "source": "YF_RATIO", "weight": 3, "higher_is_bullish": True, "why": "Relative performance against risk assets.",
         "definition": "Gold price divided by the S&P 500 index level. Measures whether gold or equities are leading. A rising ratio means gold is outperforming stocks, typically during recessions, financial crises, or inflationary regimes. A falling ratio signals risk-on conditions where equities dominate returns.",
         "unit": "ratio"},
        {"ind": "30. Platinum/Gold", "ticker": "PL=F/GC=F", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": False, "why": "Industrial cycle divergences.",
         "definition": "Platinum price divided by gold price. Historically, platinum traded at a premium to gold due to its rarity and industrial uses. Since 2015, this ratio has collapsed below 0.5x, signaling either extreme platinum undervaluation or that gold's monetary premium has decoupled from industrial metals. A low ratio often signals industrial weakness.",
         "unit": "ratio"},
        {"ind": "31. Gold/M2 Ratio", "ticker": "AU_M2", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Value of gold relative to fiat expansion.",
         "definition": "Gold price per ounce divided by US M2 money supply (in billions). Measures whether gold has kept pace with monetary expansion. A rising ratio means gold is outpacing money printing. A falling ratio means gold is getting cheaper relative to the amount of dollars in circulation, signaling potential undervaluation.",
         "unit": "ratio"}
    ],
    "VII. Technical Indicators": [
        {"ind": "32. Moving Averages", "ticker": "MA_BULL", "source": "SIMULATED", "weight": 4, "higher_is_bullish": True, "why": "Trend confirmation.",
         "definition": "A composite signal: counts how many key moving averages (50-day, 200-day) gold's price is currently above. Value of 2 = above both (strong uptrend), 1 = above one (mixed), 0 = below both (downtrend). A 'Golden Cross' (50D crossing above 200D) is a widely-followed bullish signal; a 'Death Cross' is the inverse.",
         "unit": "/2"},
        {"ind": "33. Bollinger Bands", "ticker": "BB_EXP", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Volatility envelope extension.",
         "definition": "The width of the 20-day Bollinger Bands as a percentage of the middle band (20-day SMA). Measures volatility expansion/contraction. Narrow bands ('squeeze') signal low volatility preceding a breakout. Wide bands signal high volatility and potential exhaustion. Expanding width during a trend confirms momentum; extreme width can signal overbought conditions.",
         "unit": "%"},
        {"ind": "34. RSI Divergence", "ticker": "RSI_DIV", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Momentum exhaustion indicator.",
         "definition": "The 14-day Relative Strength Index for gold. RSI oscillates 0-100, measuring the speed and magnitude of recent price changes. Above 70 = overbought (potential pullback risk). Below 30 = oversold (potential bounce). Most useful when RSI diverges from price: price makes new highs but RSI doesn't, signaling weakening momentum.",
         "unit": "0-100"},
        {"ind": "35. Volume Profile", "ticker": "VOL_PROF", "source": "YF_CALC", "weight": 1, "higher_is_bullish": True, "why": "Support level consolidation.",
         "definition": "The 20-day average daily trading volume of GC=F gold futures, normalized as a ratio to the 5Y average volume. Values above 1.0 indicate above-average participation. Rising volume during price advances confirms trend strength and institutional conviction. Falling volume during rallies warns of weak hands.",
         "unit": "ratio"}
    ],
    "VIII. Black Swan & Exogenous Risks": [
        {"ind": "36. Equity and Gold Volatility", "ticker": "^VIX", "source": "YF", "weight": 2, "higher_is_bullish": True, "why": "Spikes trigger safe-haven capital flight.",
         "definition": "The CBOE Volatility Index (VIX), derived from S&P 500 option prices. Represents the market's expectation of 30-day annualized equity volatility. Known as the 'fear gauge.' VIX spikes above 30 signal acute market stress and typically trigger safe-haven flows into gold. Sustained elevated VIX (>25) supports gold's risk premium.",
         "unit": "vol pts"},
        {"ind": "37. Geopolitical Risk", "ticker": "GPR_IDX", "source": "WEB_CSV", "weight": 3, "higher_is_bullish": True, "why": "Reflects safe-haven tail risk premiums.",
         "definition": "The Caldara-Iacoviello Geopolitical Risk Index, constructed by counting newspaper articles related to geopolitical tensions, wars, and terrorism. Baseline 100 = 1985-2019 average. Published monthly by Matteo Iacoviello (Federal Reserve Board). Gold has historically rallied 5-15% during acute geopolitical crises.",
         "unit": "index"},
        {"ind": "38. US CDS", "ticker": "US_CDS", "source": "SIMULATED", "weight": 1, "higher_is_bullish": True, "why": "Sovereign default risk proxies.",
         "definition": "The cost of credit default swap (CDS) protection on US sovereign debt, in basis points. CDS pricing reflects the market's perceived probability of a US government default. Spikes (like during the 2023 debt ceiling crisis) signal sovereign risk concerns that directly benefit gold as the ultimate 'money of last resort.'",
         "unit": "bps"},
        {"ind": "39. Lease Rates", "ticker": "LEASE_RATE", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Physical market leasing stress.",
         "definition": "The annualized rate that central banks and institutions charge to lend physical gold (historically derived from GOFO minus LIBOR, now from SOFR-based benchmarks). Rising lease rates signal that physical gold is in high demand for borrowing, often because shorts need to locate metal for delivery. Elevated rates precede supply squeezes.",
         "unit": "% ann."},
        {"ind": "40. Dealer Gamma", "ticker": "DLR_GAMMA", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "Market maker positioning impacts volatility.",
         "definition": "Estimated aggregate gamma exposure (GEX) of market makers in gold options. When dealers are 'long gamma,' they dampen volatility by buying dips and selling rips. When dealers are 'short gamma,' they must sell into declines and buy into rallies, amplifying moves. Short gamma environments produce outsized gold price swings.",
         "unit": "$M/pt"},
        {"ind": "41. EFP Spread", "ticker": "EFP_SPREAD", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Paper vs Physical arbitrage dislocations.",
         "definition": "The Exchange-for-Physical (EFP) spread: the difference between COMEX gold futures and London spot gold. Normally a small, stable spread reflecting financing costs. Blowouts (like March 2020's $70+ spread) signal severe dislocation between paper and physical markets, delivery stress, or logistical bottlenecks in moving gold between vaults.",
         "unit": "$/oz"},
        {"ind": "42. Miners vs Metal", "ticker": "GDX/GLD", "source": "YF_RATIO", "weight": 2, "higher_is_bullish": True, "why": "Miners lead the metal in bull markets.",
         "definition": "The ratio of the VanEck Gold Miners ETF (GDX) to the SPDR Gold Trust (GLD). Gold miners are leveraged plays on gold price — they should outperform bullion in a bull market. When GDX/GLD is rising, miners are confirming the gold rally with operational leverage. When miners lag (ratio falling), it signals skepticism about the move's sustainability.",
         "unit": "ratio"},
        {"ind": "43. Junior Speculation", "ticker": "GDXJ/GDX", "source": "YF_RATIO", "weight": 1, "higher_is_bullish": True, "why": "Risk-on appetite in the precious metals sector.",
         "definition": "The ratio of the VanEck Junior Gold Miners ETF (GDXJ) to the senior Gold Miners ETF (GDX). Junior miners are smaller, higher-risk explorers and developers. When GDXJ outperforms GDX, it signals speculative risk appetite within the gold sector — investors are willing to take on more risk for higher returns, a hallmark of early-to-mid bull markets.",
         "unit": "ratio"},
        {"ind": "44. Silver Beta", "ticker": "SI_BETA", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Silver outperforming signals broad bull market.",
         "definition": "The 60-day rolling regression beta of silver futures returns against gold futures returns. A beta > 1.0 means silver moves more than gold on a percentage basis. Silver typically leads gold in strong precious metals rallies (beta 1.5-3x). Expanding beta signals growing speculative enthusiasm across the sector.",
         "unit": "x"}
    ],
    "IX. Sentiment & Retail Indicators": [
        {"ind": "45. Google Trends - Buy Gold", "ticker": "GTRENDS", "source": "PYTRENDS", "weight": 2, "higher_is_bullish": False, "why": "Retail euphoria often marks local tops.",
         "definition": "Google search interest for 'buy gold' over the past 5 years, indexed 0-100 (weekly data via pytrends). Extreme spikes in search interest often coincide with retail FOMO buying near local price tops. Conversely, low search interest during rising prices suggests the move is under-owned and has further to run. A contrarian indicator.",
         "unit": "0-100"},
        {"ind": "46. Gold Advertising Spending", "ticker": "AD_SPEND", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "Late-cycle retail trapping indicator.",
         "definition": "Estimated aggregate spending on gold-related advertising (TV, online, radio) by dealers, mints, and bullion companies. Heavy advertising spend typically targets retail investors during euphoric price runs, extracting high premiums from late buyers. Peak ad spend often correlates with local price tops — a classic contrarian sell signal.",
         "unit": "$M/mo"},
        {"ind": "47. Jewelry Demand", "ticker": "JEWELRY", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Base consumer physical demand floor.",
         "definition": "Global gold jewelry consumption in tonnes, reported quarterly by the World Gold Council. Jewelry represents ~50% of annual gold demand (~2,000 tonnes). Jewelry demand is price-sensitive (falls when prices spike) but provides a structural floor. Strong jewelry demand at elevated prices signals acceptance of a new price regime and broad consumer demand.",
         "unit": "tonnes/yr"}
    ]
}

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
        "Value": format_value_with_unit(current_val, unit),
        "5Y Mean": format_value_with_unit(mean_5y, unit),
        "5Y Std": f"{std_5y:.2f}",
        "Z-Score": f"{z_score:.2f}",
        "Percentile": f"{percentile:.0f}%",
        "Total Factor Score": round(factor_score, 1),
        "Colour Indicator": color
    }

def compute_rolling_scores(historical_data, n_days=50):
    """
    Compute a daily Gold Score time series over the last n_days.
    For each day t in the window, we treat each factor's value at day t as
    'current' and compute its z-score against the full 5Y history up to day t.
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

    max_possible_score = sum(
        item['weight'] * 3
        for indicators in FACTOR_CONFIG.values()
        for item in indicators
    )

    daily_scores = {}
    for d in sampled_dates:
        day_total = 0.0
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
                day_total += factor_score
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
                'url': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
            })

    # Sort by date descending
    headlines.sort(key=lambda x: x.get('datetime', pd.Timestamp('2000-01-01')), reverse=True)
    return headlines


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
            is_computed_live = ticker in historical_data and not (
                item['source'] == 'SIMULATED' and ticker not in [
                    "FEDFUNDS_REAL", "AU_M2", "SI_BETA", "MA_BULL", "RSI_DIV", "BB_EXP"
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
                "Definition": item.get('definition', ''),
                "Ticker / Source": source_label,
                "Value": stats["Value"],
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

# ─── Overall Confidence & Scoring Calculations ───────────────────────────────

overall_score = final_df['Total Factor Score'].sum()
total_factors = len(final_df)
decisive_signals = len(final_df[final_df['Colour Indicator'].isin(['Green', 'Red'])])
confidence_pct = (decisive_signals / total_factors) * 100 if total_factors > 0 else 0

# Compute maximum possible bullish score: sum of (weight * 3) for every factor
max_possible_score = sum(
    item['weight'] * 3
    for indicators in FACTOR_CONFIG.values()
    for item in indicators
)

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
    f'{total_factors} factors &middot; {total_factors - simulated_count} live &middot; {simulated_count} simulated</div>'
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
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_scores.index, y=rolling_scores.values,
            name="Gold Score", mode="lines",
            line=dict(color="#60a5fa", width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0, secondary_y=True, line_dash="dash",
                  line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0e0e1a",
        height=480,
        margin=dict(l=55, r=55, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True)
    fig.update_yaxes(title_text="Gold $/oz", gridcolor="rgba(255,255,255,0.05)",
                     showgrid=True, secondary_y=False, tickformat="$,.0f",
                     title_font=dict(color="#FFD700", size=11))
    fig.update_yaxes(title_text="Score", secondary_y=True, range=[-1.05, 1.05],
                     tickformat="+.2f", title_font=dict(color="#60a5fa", size=11),
                     gridcolor="rgba(255,255,255,0.03)")

    chart_div_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                                  div_id="goldChart", config={"displayModeBar": False})

    headlines_json = json.dumps([
        {"date": h["date"],
         "title": html_mod.escape(h.get("title", ""), quote=True),
         "source": html_mod.escape(h.get("source", ""), quote=True),
         "url": h.get("url", "")}
        for h in headlines
    ])

    interactive_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

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
    <div class="news-footer">Sources: Google News RSS &middot; Federal Reserve FOMC Calendar</div>
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

    // Group by date descending
    var groups = {};
    var dateOrder = [];
    filtered.forEach(function(h){
        if (!groups[h.date]) { groups[h.date] = []; dateOrder.push(h.date); }
        groups[h.date].push(h);
    });
    // Sort dates descending (most recent first)
    dateOrder.sort(function(a,b){ return b.localeCompare(a); });

    var out = "";
    var shown = 0;
    for (var i = 0; i < dateOrder.length && shown < 20; i++) {
        var d = dateOrder[i];
        var items = groups[d];
        var dt = new Date(d + "T12:00:00");
        var nice = dt.toLocaleDateString("en-US", {year:"numeric",month:"short",day:"numeric",weekday:"short"});

        out += '<div class="day-header">' + nice + '</div>';
        var mx = Math.min(items.length, 5);
        for (var j = 0; j < mx; j++) {
            var h = items[j];
            var src = h.source ? ' <span class="headline-src">&mdash; ' + h.source + '</span>' : "";
            if (h.url) {
                out += '<div class="headline-item">&bull; <a href="' + h.url + '" target="_blank">' + h.title + '</a>' + src + '</div>';
            } else {
                out += '<div class="headline-item">&bull; ' + h.title + src + '</div>';
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
.spark-cell { padding: 4px 6px !important; vertical-align: middle; min-width: 130px; }
.spark-cell svg { display: block; }
</style>
""", unsafe_allow_html=True)


def render_category_table(cat_df):
    """Render a category table as HTML with hover tooltips on indicator names and sparklines."""
    cols = ["Indicator", "Ticker / Source", "Value", "Sparkline", "5Y Mean", "Z-Score",
            "Percentile", "Colour Indicator", "Total Factor Score"]

    header_labels = {
        "Colour Indicator": "Signal",
        "Total Factor Score": "Score",
        "Sparkline": "50D Trend"
    }

    html = '<table class="factor-table">'
    html += '<tr>'
    for c in cols:
        label = header_labels.get(c, c)
        html += f'<th>{label}</th>'
    html += '</tr>'

    for _, row in cat_df.iterrows():
        ind_name = row["Indicator"]
        definition = row.get("Definition", "")
        why = row.get("Why It Matters", "")

        # Escape HTML entities in text
        def esc(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

        html += '<tr>'
        for c in cols:
            val = row.get(c, "")

            if c == "Indicator":
                # Indicator cell with tooltip
                html += '<td class="ind-cell">'
                html += f'<span class="ind-name">{esc(ind_name)}</span>'
                if definition:
                    html += '<div class="tooltip-box">'
                    html += f'<div class="tt-title">{esc(ind_name)}</div>'
                    html += f'<div class="tt-def">{esc(definition)}</div>'
                    if why:
                        html += f'<div class="tt-why">Why it matters: {esc(why)}</div>'
                    html += '</div>'
                html += '</td>'

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
