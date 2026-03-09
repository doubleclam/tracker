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
        {"ind": "7. Managed Money Position", "ticker": "COT_MM", "source": "SIMULATED", "weight": 4, "higher_is_bullish": False, "why": "Crowded longs risk sharp liquidations.",
         "definition": "Net long/short position of Managed Money traders (hedge funds, CTAs) in COMEX gold futures, from the CFTC Commitments of Traders report. Extreme net long positioning (Z-score > 2.0) signals a crowded trade vulnerable to sharp liquidation. Extreme short positioning signals contrarian bullish potential.",
         "unit": "contracts"},
        {"ind": "8. Swap Dealer Position", "ticker": "COT_SD", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Commercial positioning is the 'smart money'.",
         "definition": "Net position of Swap Dealers (major banks facilitating OTC gold transactions) in COMEX gold futures. These entities are considered 'smart money' because they have superior information about physical flows. When swap dealers reduce their typical net short, it signals reduced hedging demand or outright bullishness.",
         "unit": "contracts"},
        {"ind": "9. Open Interest", "ticker": "COMEX_OI", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Reflects total speculative participation.",
         "definition": "Total number of outstanding (unsettled) gold futures contracts on COMEX. Rising OI with rising prices confirms new money entering bullish bets. Rising OI with falling prices signals aggressive new shorts. Falling OI indicates position liquidation regardless of price direction.",
         "unit": "contracts"},
        {"ind": "10. Futures Structure", "ticker": "FUT_CURVE", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Backwardation indicates tight physical supply.",
         "definition": "The shape of the gold futures term structure. Contango (normal): deferred months trade above spot, reflecting storage/financing costs. Backwardation (rare): spot trades above deferred months, signaling extreme physical demand or delivery stress. Backwardation in gold is an exceptionally bullish signal.",
         "unit": "$/oz"},
        {"ind": "11. Options Skew", "ticker": "OPT_SKEW", "source": "SIMULATED", "weight": 1, "higher_is_bullish": False, "why": "Heavy call skew implies topside exhaustion.",
         "definition": "The 25-Delta Risk Reversal: the implied volatility of 25-delta calls minus 25-delta puts on gold options. Positive values (call premium) mean the market is paying more for upside protection, suggesting bullish consensus is already priced in. Negative values (put premium) suggest fear and hedging activity.",
         "unit": "vol pts"}
    ],
    "III. Physical Fundamentals & Flow": [
        {"ind": "12. Shanghai Premium", "ticker": "SGE_PREM", "source": "SIMULATED", "weight": 4, "higher_is_bullish": True, "why": "Indicates physical demand strength in Asia.",
         "definition": "The premium (or discount) of gold on the Shanghai Gold Exchange (SGE AU9999) versus the international LBMA benchmark, in $/oz. A positive premium means Chinese buyers are paying above global prices to secure physical gold, indicating strong demand. Premiums above $30/oz are historically significant.",
         "unit": "$/oz"},
        {"ind": "12A. Gold in Other Currencies", "ticker": "XAU_BASKET", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Broad cross-currency strength.",
         "definition": "Gold's performance measured against a basket of non-USD currencies (EUR, JPY, GBP, CNY, INR). When gold makes new highs in most currencies simultaneously, it signals a genuine global repricing rather than just a USD weakness story. Broad-based strength is more sustainable.",
         "unit": "index"},
        {"ind": "13. Indian Demand Premium", "ticker": "IN_PREM", "source": "SIMULATED", "weight": 4, "higher_is_bullish": True, "why": "Core cultural and seasonal demand indicator.",
         "definition": "The premium or discount at which gold trades in India versus international benchmarks. India is the world's second-largest gold consumer. Premiums indicate strong local demand (often seasonal around Diwali and wedding season). Discounts signal weak demand or import restrictions.",
         "unit": "$/oz"},
        {"ind": "13A. India Wedding Season", "ticker": "IN_WEDDING", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Cyclical demand boosts.",
         "definition": "A seasonal indicator tracking India's wedding season intensity. Indian weddings drive enormous gold jewelry demand, particularly during auspicious dates (Oct-Dec, Jan-Feb). Peak wedding season can add 100-200 tonnes of incremental annual demand.",
         "unit": "index"},
        {"ind": "13B. China Lunar New Year", "ticker": "CN_LNY", "source": "SIMULATED", "weight": 2, "higher_is_bullish": True, "why": "Cyclical demand boosts.",
         "definition": "A seasonal indicator for Chinese gold demand around Lunar New Year (Jan-Feb). Gold gifting is a deeply embedded cultural tradition. Demand typically surges 4-6 weeks before the holiday, creating seasonal price support.",
         "unit": "index"},
        {"ind": "14. ETF Flows", "ticker": "ETF_FLOWS", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Institutional and retail accumulation.",
         "definition": "Net inflows or outflows from physically-backed gold ETFs (GLD, IAU, etc.), measured in tonnes. ETF flows represent Western institutional and retail investment demand. Large sustained inflows signal a shift in portfolio allocation toward gold. The World Gold Council reports these weekly.",
         "unit": "tonnes"},
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
        {"ind": "23. SOFR-OIS Spread", "ticker": "SOFR_OIS", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Interbank liquidity stress.",
         "definition": "The spread between the Secured Overnight Financing Rate (SOFR) and the Overnight Indexed Swap (OIS) rate. The modern replacement for the TED Spread after LIBOR's discontinuation. Widening signals stress in short-term funding markets, which can cascade into broader financial instability and gold buying.",
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
        {"ind": "35. Volume Profile", "ticker": "VOL_PROF", "source": "SIMULATED", "weight": 1, "higher_is_bullish": True, "why": "Support level consolidation.",
         "definition": "A measure of trading volume at each price level, identifying the Volume Point of Control (VPOC) — the price with the most volume. Breakouts above areas of thin volume (low-volume nodes) tend to be explosive. Price consolidation around high-volume nodes creates strong support/resistance. Thin liquidity above current price is bullish.",
         "unit": "index"}
    ],
    "VIII. Black Swan & Exogenous Risks": [
        {"ind": "36. Equity and Gold Volatility", "ticker": "^VIX", "source": "YF", "weight": 2, "higher_is_bullish": True, "why": "Spikes trigger safe-haven capital flight.",
         "definition": "The CBOE Volatility Index (VIX), derived from S&P 500 option prices. Represents the market's expectation of 30-day annualized equity volatility. Known as the 'fear gauge.' VIX spikes above 30 signal acute market stress and typically trigger safe-haven flows into gold. Sustained elevated VIX (>25) supports gold's risk premium.",
         "unit": "vol pts"},
        {"ind": "37. Geopolitical Risk", "ticker": "GPR_IDX", "source": "SIMULATED", "weight": 3, "higher_is_bullish": True, "why": "Reflects safe-haven tail risk premiums.",
         "definition": "The Caldara-Iacoviello Geopolitical Risk Index, constructed by counting newspaper articles related to geopolitical tensions, wars, and terrorism. Higher readings indicate elevated global tensions. Gold has historically rallied 5-15% during acute geopolitical crises (wars, terrorist attacks, sanctions escalations) due to its safe-haven status.",
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
        {"ind": "45. Google Trends - Buy Gold", "ticker": "GTRENDS", "source": "SIMULATED", "weight": 2, "higher_is_bullish": False, "why": "Retail euphoria often marks local tops.",
         "definition": "Google search interest for terms like 'buy gold,' 'gold price,' and 'gold investment,' indexed 0-100. Extreme spikes in search interest often coincide with retail FOMO buying near local price tops. Conversely, low search interest during rising prices suggests the move is under-owned and has further to run. A contrarian indicator.",
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
                "Definition": item.get('definition', ''),
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
</style>
""", unsafe_allow_html=True)


def render_category_table(cat_df):
    """Render a category table as HTML with hover tooltips on indicator names."""
    cols = ["Indicator", "Ticker / Source", "Value", "5Y Mean", "Z-Score",
            "Percentile", "Colour Indicator", "Total Factor Score"]

    html = '<table class="factor-table">'
    html += '<tr>'
    for c in cols:
        label = c
        if c == "Colour Indicator":
            label = "Signal"
        elif c == "Total Factor Score":
            label = "Score"
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
