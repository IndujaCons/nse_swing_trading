"""
Momentum Scanner Backtest Engine
EOD backtest engine for swing trading strategies.

Strategy J — Weekly Close Support Bounce:
  Entry support: Lowest weekly CLOSE of last 26 weeks (~6 months)
  Stop support: Lowest daily LOW of 120 days
  Entry: Daily low within 1% of weekly close support AND close > support AND IBS > 0.5
         AND green candle AND CCI(20) > -100 AND no gap-down (open >= prev close)
  Exit 1: Close >= Entry+5% → sell 50%
  Exit 2: Close >= Entry+10% → sell remaining
  Stop: Close < daily low support (6-month low)
  Trailing: Chandelier exit — Highest High (since entry) - 3x ATR(14)

Strategy T — Keltner Channel Pullback:
  Entry: Price pulls back to EMA(20) midline (was at upper Keltner in last 10 days)
         AND green candle AND no gap-down (open >= prev close)
  Exit (2-stage): +6% sell 1/3, upper Keltner sell remaining 2/3
  Stop: 5% hard SL (tightens to 3% after first partial)

Strategy R — Bullish RSI Divergence (Regular + Hidden):
  Regular: Price makes lower low but RSI(14) makes higher low (reversal)
           AND RSI(14) < 40 AND RSI divergence >= 3pt
  Hidden:  Price makes higher low but RSI(14) makes lower low (continuation)
           AND close > EMA(50) AND RSI(14) < 60 AND RSI divergence >= 5pt
  Entry: Green candle AND no gap-down AND min 2% stop distance
  Exit (2-stage like T): structural SL (1% below swing low), +6% sell 1/3,
         tight 3% SL after first exit, upper Keltner sell remaining

Strategy MW — Weekly ADX Trend Entry:
  Signal: Weekly ADX crosses above 20 (trend confirmed) AND DI+ > DI- (bullish)
  Entry: Green candle AND no gap-down on the daily bar
  Exit (2-stage like T): 8% hard SL, +6% sell 1/3, +10% sell 1/3,
         tight 3% SL after first partial, upper Keltner sell remaining
"""

import os
import sys
import random
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import load_config


# Actual Nifty Next 50 constituents (Nifty 100 = Nifty 50 + Next 50)
NIFTY_NEXT50_TICKERS = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
    "ATGL", "AUROPHARMA", "BAJAJHLDNG", "BANKBARODA", "BHEL",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DABUR",
    "DLF", "GAIL", "GODREJCP", "HAL", "HAVELLS",
    "ICICIPRULI", "ICICIGI", "INDIGO", "INDUSTOWER", "IOC",
    "IRFC", "JIOFIN", "JSWENERGY", "LICI", "LODHA",
    "LUPIN", "MANKIND", "MARICO", "MOTHERSON", "NAUKRI",
    "NHPC", "PFC", "PNB", "POLYCAB", "RECLTD",
    "SHREECEM", "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM",
    "TRENT", "TVSMOTOR", "UNITDSPR", "VEDL", "ZOMATO",
]

# Nifty 200 constituents (next 100 beyond Nifty 100)
NIFTY_200_NEXT100_TICKERS = [
    "360ONE", "ABCAPITAL", "ALKEM", "APLAPOLLO", "ASHOKLEY",
    "ASTRAL", "AUBANK", "BAJAJHFL", "BANKINDIA", "BDL",
    "BHARATFORG", "BHARTIHEXA", "BIOCON", "BLUESTARCO", "BSE",
    "CGPOWER", "COCHINSHIP", "COFORGE", "CONCOR", "COROMANDEL",
    "CUMMINSIND", "DIVISLAB", "DIXON", "DMART", "ENRIN",
    "EXIDEIND", "FEDERALBNK", "FORTIS", "GLENMARK", "GMRAIRPORT",
    "GODFRYPHLP", "GODREJPROP", "HDFCAMC", "HINDZINC", "HUDCO",
    "HYUNDAI", "IDEA", "IDFCFIRSTB", "IGL", "INDHOTEL",
    "INDIANB", "IRCTC", "IREDA", "ITCHOTELS", "JINDALSTEL",
    "JUBLFOOD", "KALYANKJIL", "KEI", "KPITTECH", "LICHSGFIN",
    "LTF", "LTM", "M&MFIN", "MAXHEALTH", "MAZDOCK",
    "MFSL", "MOTILALOFS", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NATIONALUM", "NMDC", "NTPCGREEN", "NYKAA", "OBEROIRLTY",
    "OFSS", "OIL", "PAGEIND", "PATANJALI", "PAYTM",
    "PERSISTENT", "PHOENIXLTD", "PIIND", "POLICYBZR", "POWERINDIA",
    "PREMIERENE", "PRESTIGE", "HINDPETRO", "IRB", "RVNL",
    "SAIL", "SBICARD", "SHRIRAMFIN", "SOLARINDS", "SONACOMS",
    "SUPREMEIND", "SUZLON", "SWIGGY", "TATACOMM", "TATAELXSI",
    "TATATECH", "TIINDIA", "TMPV", "TORNTPOWER", "UNIONBANK",
    "UPL", "VBL", "VMM", "VOLTAS", "WAAREEENER",
    "YESBANK", "ZYDUSLIFE",
]

# Ticker renames: new_name -> old_name (for yfinance historical data fallback)
TICKER_ALIASES = {
    "ETERNAL": "ZOMATO",      # Zomato -> Eternal (renamed 2025)
    "TMPV": "TATAMOTORS",     # Tata Motors -> TMPV (demerged Oct 2025)
}

# Nifty 500 constituents beyond Nifty 200 (next 300 by market cap)
NIFTY_500_BEYOND200_TICKERS = [
    "3MINDIA", "AADHARHFC", "AARTIIND", "AAVAS", "ABBOTINDIA",
    "ABFRL", "ABLBL", "ABREL", "ABSLAMC", "ACC",
    "ACE", "ACMESOLAR", "AEGISLOG", "AEGISVOPAK", "AFCONS",
    "AFFLE", "AGARWALEYE", "AIAENG", "AIIL", "AJANTPHARM",
    "AKUMS", "AKZOINDIA", "ALKYLAMINE", "ALOKINDS", "AMBER",
    "ANANDRATHI", "ANANTRAJ", "ANGELONE", "APARINDS", "APLLTD",
    "APOLLOTYRE", "APTUS", "ASAHIINDIA", "ASTERDM",
    "ASTRAZEN", "ATHERENERG", "ATUL", "AWL", "BALKRISIND",
    "BALRAMCHIN", "BANDHANBNK", "BASF", "BATAINDIA", "BAYERCROP",
    "BBTC", "BEML", "BERGEPAINT", "BIKAJI", "BLS",
    "BLUEDART", "BLUEJET", "BRIGADE", "BSOFT", "CAMPUS",
    "CAMS", "CANFINHOME", "CAPLIPOINT", "CARBORUNIV", "CASTROLIND",
    "CCL", "CDSL", "CEATLTD", "CENTRALBK", "CENTURYPLY",
    "CERA", "CESC", "CGCL", "CHALET", "CHAMBLFERT",
    "CHENNPETRO", "CHOICEIN", "CHOLAHLDNG", "CLEAN", "COHANCE",
    "CONCORDBIO", "CRAFTSMAN", "CREDITACC", "CRISIL", "CROMPTON",
    "CUB", "CYIENT", "DALBHARAT", "DATAPATTNS", "DBREALTY",
    "DCMSHRIRAM", "DEEPAKFERT", "DEEPAKNTR", "DELHIVERY", "DEVYANI",
    "DOMS", "ECLERX", "EIDPARRY", "EIHOTEL", "ELECON",
    "ELGIEQUIP", "EMAMILTD", "EMCURE", "ENDURANCE", "ENGINERSIN",
    "ERIS", "ESCORTS", "FACT", "FINCABLES", "FINPIPE",
    "FIRSTCRY", "FIVESTAR", "FLUOROCHEM", "FORCEMOT", "FSL",
    "GESHIP", "GICRE", "GILLETTE", "GLAND", "GLAXO",
    "GMDCLTD", "GODIGIT", "GODREJAGRO", "GODREJIND", "GPIL",
    "GRANULES", "GRAPHITE", "GRAVITA", "GRSE", "GSPL",
    "GUJGASLTD", "HAPPSTMNDS", "HBLENGINE", "HEG",
    "HEXT", "HFCL", "HINDCOPPER", "HOMEFIRST", "HONASA",
    "HONAUT", "HSCL", "IDBI", "IEX", "IFCI",
    "IGIL", "IIFL", "INDIACEM",
    "INDIAMART", "INOXINDIA", "INOXWIND", "INTELLECT", "IOB",
    "IPCALAB", "IRCON", "ITI", "JBCHEPHARM",
    "JBMA", "JINDALSAW", "JKCEMENT", "JKTYRE", "JMFINANCIL",
    "JPPOWER", "JSL", "JSWCEMENT", "JSWINFRA", "JUBLINGREA",
    "JUBLPHARMA", "JWL", "JYOTHYLAB", "JYOTICNC", "KAJARIACER",
    "KARURVYSYA", "KAYNES", "KEC", "KFINTECH", "KIMS",
    "KIRLOSBROS", "KIRLOSENG", "KPIL", "KPRMILL", "KSB",
    "LALPATHLAB", "LATENTVIEW", "LAURUSLABS", "LEMONTREE", "LINDEINDIA",
    "LLOYDSME", "LTFOODS", "LTTS", "MAHABANK", "MAHSCOOTER",
    "MAHSEAMLES", "MANAPPURAM", "MANYAVAR", "MAPMYINDIA", "MCX",
    "MEDANTA", "METROPOLIS", "MGL", "MINDACORP", "MMTC",
    "MRPL", "MSUMI", "NATCOPHARM", "NAVA",
    "NAVINFLUOR", "NBCC", "NCC", "NETWEB", "NEULANDLAB",
    "NEWGEN", "NH", "NIACL", "NIVABUPA", "NLCINDIA",
    "NSLNISP", "NUVAMA", "NUVOCO", "OLAELEC", "OLECTRA",
    "ONESOURCE", "PCBL", "PETRONET", "PFIZER", "PGEL",
    "PGHH", "PNBHOUSING", "POLYMED", "POONAWALLA", "PPLPHARMA",
    "PRAJIND", "PTCIL", "PVRINOX", "RADICO", "RAILTEL",
    "RAINBOW", "RAMCOCEM", "RBLBANK", "RCF", "REDINGTON",
    "RELINFRA", "RHIM", "RITES", "RKFORGE", "RPOWER",
    "RRKABEL", "SAGILITY", "SAILIFE", "SAMMAANCAP", "SAPPHIRE",
    "SARDAEN", "SAREGAMA", "SBFC", "SCHAEFFLER", "SCHNEIDER",
    "SCI", "SHYAMMETL", "SIGNATURE", "SJVN", "SOBHA",
    "SONATSOFTW", "STARHEALTH", "SUMICHEM", "SUNDARMFIN", "SUNDRMFAST",
    "SUNTV", "SWANCORP", "SYNGENE", "SYRMA", "TARIL",
    "TATACHEM", "TATAINVEST", "TBOTEK", "TECHNOE", "TEJASNET",
    "THELEELA", "THERMAX", "TIMKEN", "TITAGARH", "TRIDENT",
    "TRITURBINE", "TRIVENI", "TTML", "UBL", "UCOBANK",
    "UNOMINDA", "USHAMART", "UTIAMC", "VENTIVE", "VGUARD",
    "VIJAYA", "VTL", "WELCORP", "WELSPUNLIV", "WHIRLPOOL",
    "WOCKPHARMA", "ZEEL", "ZENSARTECH", "ZENTEC", "ZFCVINDIA",
]

BATCH_VARIANTS = [
    ("J", None, "J: Weekly Support"),
    ("T", None, "T: Keltner Pullback"),
    ("R", None, "R: RSI Divergence"),
    ("MW", None, "MW: Weekly ADX"),
]


class MomentumBacktester:
    """Single-stock daily backtest for swing trading strategies (J, T, R, MW)."""

    def _calculate_rsi_series(self, closes: pd.Series, period: int) -> pd.Series:
        """Calculate full RSI series using Wilder's smoothing."""
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _find_swing_lows(lows: pd.Series, left: int = 5, right: int = 3) -> list:
        """Find swing low positions: low[i] is min of surrounding window.

        Returns list of (iloc_pos, low_value) tuples, confirmed `right` bars
        after the actual low.
        """
        result = []
        vals = lows.values
        n = len(vals)
        for i in range(left, n - right):
            window = vals[i - left: i + right + 1]
            if vals[i] == np.min(window):
                result.append((i, float(vals[i])))
        return result

    @staticmethod
    def _calculate_adx_series(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14):
        """Calculate ADX, +DI, and -DI series using Wilder's smoothing.

        Returns (adx, plus_di, minus_di) as pd.Series.
        """
        prev_high = highs.shift(1)
        prev_low = lows.shift(1)
        prev_close = closes.shift(1)

        # True Range
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = highs - prev_high
        down_move = prev_low - lows
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=highs.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=highs.index)

        # Wilder's smoothing (EWM with alpha=1/period)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=period).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def _rsi_near_swing(rsi14_vals, idx, window=3):
        """Get min RSI within ±window bars of a swing low index.

        Price low and RSI low often don't land on the same bar —
        use the lowest RSI in the zone to match how traders read charts.
        """
        start = max(0, idx - window)
        end = min(len(rsi14_vals), idx + window + 1)
        chunk = rsi14_vals[start:end]
        valid = [v for v in chunk if not np.isnan(v)]
        return min(valid) if valid else rsi14_vals[idx]

    @staticmethod
    def _detect_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                                   max_lookback=50, min_sep=5,
                                   rsi_threshold=40, min_rsi_divergence=3,
                                   min_price_drop=0.0, max_curr_age=None):
        """Check for bullish RSI divergence at bar i.

        Looks for two swing lows in the last max_lookback bars where:
        - Price: current swing low < previous * (1 - min_price_drop) (meaningful lower low)
        - RSI(14): current > previous + min_rsi_divergence (meaningful higher low)
        - RSI(14) < rsi_threshold at the current swing low (oversold zone)
        - If max_curr_age set, current swing low must be within that many bars of i.

        Uses min RSI within ±3 bars of each swing low to match visual chart reading.

        Returns (True, swing_low_price) or (False, None).
        """
        # Gather recent swing lows confirmed by bar i (confirmed = idx + right <= i)
        recent = [(idx, val) for idx, val in swing_lows
                  if idx <= i and i - idx <= max_lookback]
        if len(recent) < 2:
            return False, None

        # Check pairs: most recent first
        for k in range(len(recent) - 1, 0, -1):
            curr_idx, curr_low = recent[k]
            prev_idx, prev_low = recent[k - 1]
            # Current swing low must be fresh if max_curr_age is set
            if max_curr_age is not None and i - curr_idx > max_curr_age:
                continue
            if curr_idx - prev_idx < min_sep:
                continue
            # Meaningful lower low in price (>= min_price_drop)
            if curr_low >= prev_low * (1 - min_price_drop):
                continue
            # Use min RSI near each swing low (±3 bars)
            curr_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, curr_idx)
            prev_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, prev_idx)
            if np.isnan(curr_rsi) or np.isnan(prev_rsi):
                continue
            if curr_rsi - prev_rsi < min_rsi_divergence:
                continue
            # RSI below threshold at current swing low (oversold zone)
            if curr_rsi >= rsi_threshold:
                continue
            return True, curr_low
        return False, None

    @staticmethod
    def _detect_hidden_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                                          max_lookback=50, min_sep=5,
                                          rsi_threshold=60, min_rsi_divergence=5,
                                          max_curr_age=None):
        """Check for hidden bullish RSI divergence at bar i.

        Hidden bullish divergence (uptrend continuation):
        - Price: current swing low > previous swing low (higher low)
        - RSI(14): current < previous - min_rsi_divergence (lower low in RSI)
        - RSI(14) < rsi_threshold at current swing low (relaxed from 40)
        - If max_curr_age set, current swing low must be within that many bars of i.

        Uses min RSI within ±3 bars of each swing low to match visual chart reading.

        Returns (True, swing_low_price) or (False, None).
        """
        recent = [(idx, val) for idx, val in swing_lows
                  if idx <= i and i - idx <= max_lookback]
        if len(recent) < 2:
            return False, None

        for k in range(len(recent) - 1, 0, -1):
            curr_idx, curr_low = recent[k]
            prev_idx, prev_low = recent[k - 1]
            if max_curr_age is not None and i - curr_idx > max_curr_age:
                continue
            if curr_idx - prev_idx < min_sep:
                continue
            # Higher low in price (uptrend continuation)
            if curr_low <= prev_low:
                continue
            # Use min RSI near each swing low (±3 bars)
            curr_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, curr_idx)
            prev_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, prev_idx)
            if np.isnan(curr_rsi) or np.isnan(prev_rsi):
                continue
            if prev_rsi - curr_rsi < min_rsi_divergence:
                continue
            # RSI below relaxed threshold
            if curr_rsi >= rsi_threshold:
                continue
            return True, curr_low
        return False, None

    @staticmethod
    def _explain_hidden_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                                            max_lookback=50, min_sep=5,
                                            rsi_threshold=60, min_rsi_divergence=5):
        """Like _detect_hidden_bullish_divergence but returns full detail.

        Returns (True, detail_dict) or (False, None).
        detail_dict has: prev_idx, prev_low, prev_rsi, curr_idx, curr_low, curr_rsi
        """
        recent = [(idx, val) for idx, val in swing_lows
                  if idx <= i and i - idx <= max_lookback]
        if len(recent) < 2:
            return False, None

        for k in range(len(recent) - 1, 0, -1):
            curr_idx, curr_low = recent[k]
            prev_idx, prev_low = recent[k - 1]
            if curr_idx - prev_idx < min_sep:
                continue
            if curr_low <= prev_low:
                continue
            curr_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, curr_idx)
            prev_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, prev_idx)
            if np.isnan(curr_rsi) or np.isnan(prev_rsi):
                continue
            if prev_rsi - curr_rsi < min_rsi_divergence:
                continue
            if curr_rsi >= rsi_threshold:
                continue
            return True, {
                "prev_idx": prev_idx, "prev_low": float(prev_low),
                "prev_rsi": float(prev_rsi),
                "curr_idx": curr_idx, "curr_low": float(curr_low),
                "curr_rsi": float(curr_rsi),
            }
        return False, None

    @staticmethod
    def _explain_bullish_divergence(lows_vals, rsi14_vals, i, swing_lows,
                                     max_lookback=50, min_sep=5,
                                     rsi_threshold=40, min_rsi_divergence=3,
                                     min_price_drop=0.0):
        """Like _detect_bullish_divergence but returns full detail for explanation.

        Uses min RSI within ±3 bars of each swing low to match visual chart reading.

        Returns (True, detail_dict) or (False, None).
        detail_dict has: prev_idx, prev_low, prev_rsi, curr_idx, curr_low, curr_rsi
        """
        recent = [(idx, val) for idx, val in swing_lows
                  if idx <= i and i - idx <= max_lookback]
        if len(recent) < 2:
            return False, None

        for k in range(len(recent) - 1, 0, -1):
            curr_idx, curr_low = recent[k]
            prev_idx, prev_low = recent[k - 1]
            if curr_idx - prev_idx < min_sep:
                continue
            if curr_low >= prev_low * (1 - min_price_drop):
                continue
            curr_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, curr_idx)
            prev_rsi = MomentumBacktester._rsi_near_swing(rsi14_vals, prev_idx)
            if np.isnan(curr_rsi) or np.isnan(prev_rsi):
                continue
            if curr_rsi - prev_rsi < min_rsi_divergence:
                continue
            if curr_rsi >= rsi_threshold:
                continue
            return True, {
                "prev_idx": prev_idx, "prev_low": float(prev_low),
                "prev_rsi": float(prev_rsi),
                "curr_idx": curr_idx, "curr_low": float(curr_low),
                "curr_rsi": float(curr_rsi),
            }
        return False, None

    @staticmethod
    def _calculate_cci_series(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI) series."""
        tp = (highs + lows + closes) / 3
        sma_tp = tp.rolling(window=period, min_periods=period).mean()
        mean_dev = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        return cci

    def run(self, symbol: str, period_days: int, strategy: str = "B",
            capital: int = 100000, exit_target: str = None,
            _daily_data=None, end_date=None,
            _stop_tolerance: float = 0.0) -> Dict:
        """
        Run daily-only backtest for a single symbol.

        Args:
            symbol: NSE ticker (e.g. "ICICIBANK")
            period_days: backtest lookback in calendar days (30, 90, 180, 365)
            strategy: "J", "T", or "R"
            capital: starting capital in INR
            exit_target: profit target for Strategy J variants

        Returns:
            Dict with trades list, summary stats, and metadata.
        """
        config = load_config()
        rsi2_entry = config.get("momentum_rsi2_threshold", 75)

        end_date = end_date or datetime.now()

        if _daily_data is not None:
            daily = _daily_data
        else:
            nse_symbol = f"{symbol}.NS"
            # Fetch daily data (need 500 days warmup for EMA200 to stabilize)
            daily_start = end_date - timedelta(days=period_days + 500)
            try:
                daily = yf.Ticker(nse_symbol).history(start=daily_start, end=end_date)
            except Exception:
                daily = pd.DataFrame()

            # Fallback to old ticker name if renamed
            if (daily.empty or len(daily) < 210) and symbol in TICKER_ALIASES:
                old_symbol = f"{TICKER_ALIASES[symbol]}.NS"
                try:
                    daily = yf.Ticker(old_symbol).history(start=daily_start, end=end_date)
                except Exception:
                    daily = pd.DataFrame()

        if daily.empty or len(daily) < 210:
            return {"error": f"Insufficient daily data for {symbol}",
                    "trades": [], "summary": self._empty_summary()}

        closes = daily["Close"]
        opens = daily["Open"]
        highs = daily["High"]
        lows = daily["Low"]

        # IBS = (Close - Low) / (High - Low), clamped to [0, 1]
        hl_range = highs - lows
        ibs_series = ((closes - lows) / hl_range).where(hl_range > 0, 0.5)

        # NR7: True if today's range is the narrowest of the last 7 days
        rolling_min_range = hl_range.rolling(window=7, min_periods=7).min()
        nr7_series = (hl_range == rolling_min_range) & (hl_range > 0)

        # Support levels (exclude current bar via shift)
        support_6m = lows.rolling(window=120, min_periods=120).min().shift(1)   # 6-month
        support_1y = lows.rolling(window=252, min_periods=252).min().shift(1)   # 1-year

        # Volume for daily use
        volumes = daily["Volume"].astype(float)

        # RS Rating filter for Strategy J
        rs_filter_series = None  # True/False: outperforming Nifty (for J)
        nifty_close = None
        if strategy in ("J",):
            try:
                nifty_data = yf.Ticker("^NSEI").history(
                    start=daily.index[0], end=end_date)
                if not nifty_data.empty:
                    nifty_close = nifty_data["Close"].reindex(daily.index, method="ffill")

                    def _weighted_rs_score(stock_closes):
                        r3m = stock_closes.pct_change(periods=63)
                        r6m = stock_closes.pct_change(periods=126)
                        r9m = stock_closes.pct_change(periods=189)
                        r12m = stock_closes.pct_change(periods=252)
                        return 0.4 * r3m + 0.3 * r6m + 0.2 * r9m + 0.1 * r12m
                    stock_ws = _weighted_rs_score(closes)
                    nifty_ws = _weighted_rs_score(nifty_close)
                    rs_filter_series = stock_ws > nifty_ws

            except Exception:
                pass

        # Weekly support for Strategy J
        # Entry: lowest weekly CLOSE of last 26 weeks
        # Stop: lowest weekly LOW of last 26 weeks
        weekly_support_series = None
        weekly_low_stop_series = None
        if strategy == "J":
            weekly = daily.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            # 26-week rolling min of weekly CLOSE, skip last 2 weeks (use proven support, not recent noise)
            w_support = weekly["Close"].rolling(window=26, min_periods=26).min().shift(2)
            weekly_support_series = w_support.reindex(daily.index, method="ffill")
            # 26-week rolling min of weekly LOW (for stop-loss), also skip last 2 weeks
            w_low_stop = weekly["Low"].rolling(window=26, min_periods=26).min().shift(2)
            weekly_low_stop_series = w_low_stop.reindex(daily.index, method="ffill")


        # Pre-compute indicator series over entire dataset
        rsi2_series = self._calculate_rsi_series(closes, 2)
        rsi3_series = self._calculate_rsi_series(closes, 3)
        rsi14_series = self._calculate_rsi_series(closes, 14)
        ema5_series = closes.ewm(span=5, adjust=False).mean()
        ema8_series = closes.ewm(span=8, adjust=False).mean()
        ema10_series = closes.ewm(span=10, adjust=False).mean()
        ema20_series = closes.ewm(span=20, adjust=False).mean()
        ema50_series = closes.ewm(span=50, adjust=False).mean()
        ema200_series = closes.ewm(span=200, adjust=False).mean()
        vol_avg20_series = volumes.rolling(window=20, min_periods=20).mean()
        cci20_series = self._calculate_cci_series(highs, lows, closes, 20)

        # ATR(14) for Strategy T (Keltner Channel)
        prev_close_series = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close_series).abs()
        tr3 = (lows - prev_close_series).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14_series = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # Tracking variables for Strategy T (2-stage: +6% partial, Keltner remaining)
        t_partial_stage = 0  # 0=none, 1=sold 1/3 at +6%, 2=sold 2/3 at +10%
        t_remaining = 0

        # Tracking variables for Strategy R (RSI divergence, 2-stage like T)
        r_partial_stage = 0
        r_remaining = 0
        r_swing_low_stop = 0.0  # structural SL: 1% below divergence swing low

        # Tracking variables for Strategy MW (Weekly MACD, 2-stage like T)
        mw_partial_stage = 0
        mw_remaining = 0

        # Pre-compute swing lows for Strategy R
        swing_lows = []
        if strategy == "R":
            swing_lows = self._find_swing_lows(lows)

        # Pre-compute weekly ADX for Strategy MW
        mw_weekly = None
        mw_weekly_adx_vals = None
        mw_weekly_plus_di_vals = None
        mw_weekly_minus_di_vals = None
        if strategy == "MW":
            mw_weekly = daily.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            mw_adx, mw_plus_di, mw_minus_di = self._calculate_adx_series(
                mw_weekly["High"], mw_weekly["Low"], mw_weekly["Close"])
            mw_weekly_adx_vals = mw_adx.values
            mw_weekly_plus_di_vals = mw_plus_di.values
            mw_weekly_minus_di_vals = mw_minus_di.values

        # Pre-compute weekly data for Strategy RW / TW
        rw_weekly = None
        rw_weekly_rsi14_vals = None
        rw_weekly_lows_vals = None
        rw_weekly_swing_lows = None
        if strategy == "RW":
            rw_weekly = daily.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            rw_weekly_rsi14 = self._calculate_rsi_series(rw_weekly["Close"], 14)
            rw_weekly_rsi14_vals = rw_weekly_rsi14.values
            rw_weekly_lows_vals = rw_weekly["Low"].values
            rw_weekly_swing_lows = self._find_swing_lows(rw_weekly["Low"], left=3, right=2)

        # Determine backtest start index (need 200+ bars warmup)
        bt_start_date = (end_date - timedelta(days=period_days)).date()
        bt_indices = [i for i, ts in enumerate(daily.index)
                      if ts.date() >= bt_start_date and i >= 200]

        if not bt_indices:
            return {"error": "No trading days in selected period",
                    "trades": [], "summary": self._empty_summary()}

        # Backtest loop
        in_position = False
        entry_price = 0.0
        entry_date = None
        shares = 0
        remaining_shares = 0  # For scale-out: shares left after partial exit
        partial_exit_done = False  # For scale-out: has first leg exited?
        trades = []
        entry_support_j = 0.0    # For Strategy J: weekly open support (entry level)
        entry_stop_j = 0.0       # For Strategy J: weekly open support (stop level)
        entry_bar = 0            # For Strategy J: bar index at entry
        nifty_at_entry = 0.0     # For Strategy J: Nifty close at entry (for SL filter)
        j_highest_high = 0.0     # For Strategy J: highest high since entry (for chandelier)

        for i in bt_indices:
            price = float(closes.iloc[i])
            rsi2 = float(rsi2_series.iloc[i])
            rsi2_prev = float(rsi2_series.iloc[i - 1]) if i > 0 else np.nan
            rsi3 = float(rsi3_series.iloc[i])
            rsi3_prev = float(rsi3_series.iloc[i - 1]) if i > 0 else np.nan
            ema5 = float(ema5_series.iloc[i])
            ema8 = float(ema8_series.iloc[i])
            ema10 = float(ema10_series.iloc[i])
            ema20 = float(ema20_series.iloc[i])
            ema50 = float(ema50_series.iloc[i])
            ema200 = float(ema200_series.iloc[i])
            vol_today = float(volumes.iloc[i])
            vol_avg20 = float(vol_avg20_series.iloc[i]) if not pd.isna(vol_avg20_series.iloc[i]) else 0.0
            ibs = float(ibs_series.iloc[i])
            nr7 = bool(nr7_series.iloc[i])
            open_price = float(opens.iloc[i])
            high = float(highs.iloc[i])
            low = float(lows.iloc[i])
            prior_sup_6m = float(support_6m.iloc[i]) if not pd.isna(support_6m.iloc[i]) else None
            prior_sup_1y = float(support_1y.iloc[i]) if not pd.isna(support_1y.iloc[i]) else None
            # Today's low can update support only if candle is green
            is_green_candle = price > open_price
            sup_6m = min(prior_sup_6m, low) if (is_green_candle and prior_sup_6m is not None) else prior_sup_6m
            sup_1y = min(prior_sup_1y, low) if (is_green_candle and prior_sup_1y is not None) else prior_sup_1y
            day = daily.index[i].date()

            if pd.isna(rsi2):
                continue

            # Previous bar values for crossover detection
            prev_close = float(closes.iloc[i - 1]) if i > 0 else None
            prev_ema20 = float(ema20_series.iloc[i - 1]) if i > 0 else None

            if strategy == "J":
                # Strategy J: Weekly Close Support Bounce (scale-out)
                # Entry: daily low within 1% of 26-week min weekly CLOSE
                # Stop: close < 26-week min weekly LOW
                # Exit 1: +5% → sell 50%   Exit 2: +10% → sell remaining

                w_support = float(weekly_support_series.iloc[i]) if weekly_support_series is not None and not pd.isna(weekly_support_series.iloc[i]) else None
                w_low_stop = float(weekly_low_stop_series.iloc[i]) if weekly_low_stop_series is not None and not pd.isna(weekly_low_stop_series.iloc[i]) else None

                entry_signal = False
                if w_support is not None and not in_position:
                    close_near = ((price - w_support) / w_support) * 100 if w_support > 0 else 999
                    cci_val = float(cci20_series.iloc[i]) if not pd.isna(cci20_series.iloc[i]) else 0.0
                    no_gap_down_j = (prev_close is None or open_price >= prev_close)
                    entry_signal = (close_near >= 0        # close above support
                                    and close_near <= 3.0  # close within +3% of support
                                    and ibs > 0.5          # bounce
                                    and price > open_price # green candle
                                    and cci_val > -100     # CCI(20) not deeply oversold
                                    and no_gap_down_j)     # no gap-down

                # Exits (scale-out: 50% at +5%, remaining at +8%)
                t1_price = entry_price * 1.05 if in_position else 0
                t2_price = entry_price * 1.10 if in_position else 0
                days_since_entry = (i - entry_bar) if in_position else 0

                def _j_trade(ep, shares_, exit_day, exit_px, reason):
                    t = self._make_trade(entry_date, ep, shares_, exit_day, exit_px, reason)
                    t["zone_bottom"] = round(entry_support_j, 2)
                    t["zone_top"] = None
                    t["zone_formed"] = None
                    t["rally_pct"] = None
                    t["vol_ratio"] = None
                    t["entry_dist_pct"] = round(((ep - entry_support_j) / entry_support_j) * 100, 2) if entry_support_j > 0 else 0
                    trades.append(t)

                def _j_stop(sz):
                    """Check stop, return True if stopped.
                    Skip support break if Nifty fell same or more since entry."""
                    nifty_shields = False
                    if nifty_at_entry > 0 and nifty_close is not None:
                        nifty_now = float(nifty_close.iloc[i])
                        nifty_pct = (nifty_now - nifty_at_entry) / nifty_at_entry
                        stock_pct = (price - entry_price) / entry_price
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            nifty_shields = True

                    if nifty_shields:
                        return False  # Ignore stop — market-wide fall

                    if price < entry_stop_j:
                        _j_trade(entry_price, sz, day, price, "SUPPORT_BREAK")
                        return True
                    return False

                # Track highest high for chandelier exit
                if in_position:
                    j_highest_high = max(j_highest_high, high)

                if in_position and not partial_exit_done:
                    if _j_stop(shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    if price >= t1_price:
                        half = shares // 2
                        _j_trade(entry_price, half, day, price, "5PCT_PARTIAL")
                        remaining_shares = shares - half
                        partial_exit_done = True
                        if price >= t2_price:
                            _j_trade(entry_price, remaining_shares, day, price, "10PCT")
                            in_position = False
                            partial_exit_done = False
                        continue

                if in_position and partial_exit_done:
                    if _j_stop(remaining_shares):
                        in_position = False
                        partial_exit_done = False
                        continue
                    # Chandelier exit: highest high since entry - 3x ATR(14)
                    atr14_val = float(atr14_series.iloc[i]) if not pd.isna(atr14_series.iloc[i]) else 0.0
                    chandelier_stop = j_highest_high - 3.0 * atr14_val
                    if atr14_val > 0 and price < chandelier_stop:
                        _j_trade(entry_price, remaining_shares, day, price, "CHANDELIER_EXIT")
                        in_position = False
                        partial_exit_done = False
                        continue
                    if price >= t2_price:
                        _j_trade(entry_price, remaining_shares, day, price, "10PCT")
                        in_position = False
                        partial_exit_done = False
                        continue

                # Underwater exit: 10 trading days below entry
                if in_position and (i - entry_bar) >= 10 and price < entry_price:
                    sz = remaining_shares if partial_exit_done else shares
                    _j_trade(entry_price, sz, day, price, "UNDERWATER_EXIT")
                    in_position = False
                    partial_exit_done = False
                    continue

                # Entry (only when fully out)
                if entry_signal and not in_position:
                    shares = int(capital // price)
                    if shares > 0:
                        entry_price = price
                        entry_date = day
                        entry_bar = i
                        entry_support_j = w_support  # Weekly close support (for display)
                        entry_stop_j = w_low_stop if w_low_stop is not None else w_support
                        nifty_at_entry = float(nifty_close.iloc[i]) if nifty_close is not None else 0.0
                        j_highest_high = high
                        in_position = True
                        partial_exit_done = False
                continue

            elif strategy == "T":
                # Strategy T: Keltner Channel Pullback (2-stage exit)
                # Entry: Price near EMA(20) (within 1%) AND was at upper Keltner in last 10 bars
                #        AND green candle AND no gap-down
                # Stage 0: 5% SL. +6% → sell 1/3, stage=1.
                # Stage 1: 3% SL. +10% → sell 1/3, stage=2.
                # Stage 2: 3% SL. Upper Keltner → sell remaining.
                # Underwater: 10 trading days below entry → exit all remaining.
                atr14 = float(atr14_series.iloc[i]) if not pd.isna(atr14_series.iloc[i]) else 0.0
                upper_keltner = ema20 + 2 * atr14

                if in_position:
                    third = shares // 3
                    t_sl_pct = 0.03 if t_partial_stage >= 1 else 0.05
                    exited = False

                    # Hard SL (tightens to 3% after first partial)
                    if price <= entry_price * (1 - t_sl_pct):
                        sl_label = f"HARD_SL_{int(t_sl_pct*100)}PCT"
                        trades.append(self._make_trade(
                            entry_date, entry_price, t_remaining, day,
                            price, sl_label))
                        exited = True

                    # 2-stage partial exits (+6% sell 1/3, Keltner sells rest)
                    if not exited and t_partial_stage == 0 and price >= entry_price * 1.06 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_6PCT_1of3"))
                        t_remaining = shares - third
                        t_partial_stage = 1
                    elif not exited and t_partial_stage == 1 and price >= entry_price * 1.10 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_10PCT_2of3"))
                        t_remaining = shares - 2 * third
                        t_partial_stage = 2

                    # Upper Keltner exit on remaining
                    if not exited and price >= upper_keltner:
                        trades.append(self._make_trade(
                            entry_date, entry_price, t_remaining, day,
                            price, "KELTNER_UPPER_EXIT"))
                        exited = True

                    # Underwater exit: 10 trading days below entry
                    if not exited and (i - entry_bar) >= 10 and price < entry_price:
                        trades.append(self._make_trade(
                            entry_date, entry_price, t_remaining, day,
                            price, "UNDERWATER_EXIT"))
                        exited = True

                    if exited:
                        in_position = False
                        t_partial_stage = 0
                        t_remaining = 0
                    continue

                # Entry check (not in position)
                if atr14 > 0:
                    near_ema20 = abs(price - ema20) / ema20 <= 0.01
                    was_at_upper = False
                    for lookback_j in range(max(0, i - 10), i):
                        past_high = float(highs.iloc[lookback_j])
                        past_ema20 = float(ema20_series.iloc[lookback_j])
                        past_atr14 = float(atr14_series.iloc[lookback_j]) if not pd.isna(atr14_series.iloc[lookback_j]) else 0.0
                        past_upper = past_ema20 + 2 * past_atr14
                        if past_high >= past_upper:
                            was_at_upper = True
                            break
                    no_gap_down_t = (prev_close is None or open_price >= prev_close)
                    if near_ema20 and was_at_upper and price > open_price and no_gap_down_t and ibs > 0.5:
                        shares = int(capital // price)
                        if shares > 0:
                            entry_price = price
                            entry_date = day
                            entry_bar = i
                            in_position = True
                            t_partial_stage = 0
                            t_remaining = shares
                continue

            elif strategy == "R":
                # Strategy R: Bullish RSI Divergence (2-stage exit like T)
                atr14 = float(atr14_series.iloc[i]) if not pd.isna(atr14_series.iloc[i]) else 0.0
                upper_keltner = ema20 + 2 * atr14

                if in_position:
                    third = shares // 3
                    r_sl_pct = 0.03 if r_partial_stage >= 1 else 0.01
                    structural_stop = r_swing_low_stop
                    exited = False

                    # Nifty crash shield for R: skip SL if Nifty fell same or more
                    r_nifty_shields = False
                    if nifty_at_entry > 0 and nifty_close is not None:
                        nifty_now = float(nifty_close.iloc[i])
                        nifty_pct = (nifty_now - nifty_at_entry) / nifty_at_entry
                        stock_pct = (price - entry_price) / entry_price
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            r_nifty_shields = True

                    # Structural SL (1% below divergence swing low)
                    if not r_nifty_shields and price <= structural_stop:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "STRUCTURAL_SL"))
                        exited = True

                    # 2-stage partial exits (+6% sell 1/3, Keltner sells rest)
                    if not exited and r_partial_stage == 0 and price >= entry_price * 1.06 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_6PCT_1of3"))
                        r_remaining = shares - third
                        r_partial_stage = 1
                    elif not exited and r_partial_stage == 1 and price >= entry_price * 1.10 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_10PCT_2of3"))
                        r_remaining = shares - 2 * third
                        r_partial_stage = 2

                    # Tight SL after first partial (3%) — also shielded
                    if not exited and not r_nifty_shields and r_partial_stage >= 1 and price <= entry_price * (1 - 0.03):
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "HARD_SL_3PCT"))
                        exited = True

                    # Upper Keltner exit on remaining
                    if not exited and price >= upper_keltner:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "KELTNER_UPPER_EXIT"))
                        exited = True

                    # Underwater exit: 10 trading days below entry
                    if not exited and (i - entry_bar) >= 10 and price < entry_price:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "UNDERWATER_EXIT"))
                        exited = True

                    if exited:
                        in_position = False
                        r_partial_stage = 0
                        r_remaining = 0
                    continue

                # Entry check (not in position)
                no_gap_down_r = (prev_close is None or open_price >= prev_close)
                is_green_r = price > open_price
                if is_green_r and no_gap_down_r and ibs > 0.5:
                    rsi14_vals = rsi14_series.values
                    lows_vals = lows.values
                    divergence, swing_low_val = self._detect_bullish_divergence(
                        lows_vals, rsi14_vals, i, swing_lows,
                        rsi_threshold=35)
                    is_hidden_div = False
                    if not divergence:
                        # Try hidden bullish divergence if price > EMA50 (uptrend)
                        if price > ema50:
                            divergence, swing_low_val = self._detect_hidden_bullish_divergence(
                                lows_vals, rsi14_vals, i, swing_lows)
                            if divergence:
                                is_hidden_div = True
                    if divergence and swing_low_val is not None:
                        r_swing_low_stop_cand = swing_low_val * 0.99
                        r_stop_pct_cand = (price - r_swing_low_stop_cand) / price * 100 if price > 0 else 99.0
                        r_min_stop = 2.0 if is_hidden_div else 0.0
                        if r_min_stop < r_stop_pct_cand <= 6.0 + _stop_tolerance:
                            shares = int(capital // price)
                            if shares > 0:
                                entry_price = price
                                entry_date = day
                                entry_bar = i
                                in_position = True
                                r_partial_stage = 0
                                r_remaining = shares
                                r_swing_low_stop = r_swing_low_stop_cand
                                nifty_at_entry = float(nifty_close.iloc[i]) if nifty_close is not None else 0.0
                continue

            elif strategy == "RW":
                # Strategy RW: Weekly RSI Divergence (same exits as R, 50d underwater)
                atr14 = float(atr14_series.iloc[i]) if not pd.isna(atr14_series.iloc[i]) else 0.0
                upper_keltner = ema20 + 2 * atr14

                if in_position:
                    third = shares // 3
                    structural_stop = r_swing_low_stop
                    exited = False

                    if price <= structural_stop:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "STRUCTURAL_SL"))
                        exited = True

                    if not exited and r_partial_stage == 0 and price >= entry_price * 1.06 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_6PCT_1of3"))
                        r_remaining = shares - third
                        r_partial_stage = 1
                    elif not exited and r_partial_stage == 1 and price >= entry_price * 1.10 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_10PCT_2of3"))
                        r_remaining = shares - 2 * third
                        r_partial_stage = 2

                    if not exited and r_partial_stage >= 1 and price <= entry_price * 0.97:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "HARD_SL_3PCT"))
                        exited = True

                    if not exited and price >= upper_keltner:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "KELTNER_UPPER_EXIT"))
                        exited = True

                    # 50-day underwater exit for RW
                    if not exited and (i - entry_bar) >= 50 and price < entry_price:
                        trades.append(self._make_trade(
                            entry_date, entry_price, r_remaining, day,
                            price, "UNDERWATER_EXIT"))
                        exited = True

                    if exited:
                        in_position = False
                        r_partial_stage = 0
                        r_remaining = 0
                    continue

                # RW entry: weekly RSI divergence (>=3pt RSI div)
                no_gap_down_rw = (prev_close is None or open_price >= prev_close)
                is_green_rw = price > open_price
                if is_green_rw and no_gap_down_rw and rw_weekly is not None:
                    w_dates = rw_weekly.index
                    day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                    w_before = w_dates[w_dates < day_ts]
                    if len(w_before) >= 2:
                        w_idx = len(w_before) - 1
                        divergence, swing_low_val = self._detect_bullish_divergence(
                            rw_weekly_lows_vals, rw_weekly_rsi14_vals, w_idx, rw_weekly_swing_lows,
                            max_lookback=13, min_sep=2,
                            rsi_threshold=100, min_rsi_divergence=3)
                        if not divergence:
                            divergence, swing_low_val = self._detect_hidden_bullish_divergence(
                                rw_weekly_lows_vals, rw_weekly_rsi14_vals, w_idx, rw_weekly_swing_lows,
                                max_lookback=13, min_sep=2,
                                rsi_threshold=100, min_rsi_divergence=3)
                        if divergence and swing_low_val is not None:
                            rw_stop_cand = swing_low_val * 0.99
                            rw_stop_pct = (price - rw_stop_cand) / price * 100 if price > 0 else 99.0
                            if 2.0 <= rw_stop_pct <= 8.0:
                                shares = int(capital // price)
                                if shares > 0:
                                    entry_price = price
                                    entry_date = day
                                    entry_bar = i
                                    in_position = True
                                    r_partial_stage = 0
                                    r_remaining = shares
                                    r_swing_low_stop = rw_stop_cand
                continue

            elif strategy == "MW":
                # Strategy MW: Weekly ADX Trend Entry (2-stage exit, weekly Keltner)
                # Weekly upper Keltner for exit
                mw_upper_keltner = 0.0
                if mw_weekly is not None:
                    w_dates = mw_weekly.index
                    day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                    w_before = w_dates[w_dates < day_ts]
                    if len(w_before) >= 1:
                        w_idx = len(w_before) - 1
                        w_closes = mw_weekly["Close"]
                        w_ema20 = float(w_closes.ewm(span=20, adjust=False).mean().iloc[w_idx])
                        w_highs = mw_weekly["High"]
                        w_lows = mw_weekly["Low"]
                        w_prev_close = w_closes.shift(1)
                        w_tr = pd.concat([w_highs - w_lows, (w_highs - w_prev_close).abs(), (w_lows - w_prev_close).abs()], axis=1).max(axis=1)
                        w_atr14 = float(w_tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[w_idx])
                        mw_upper_keltner = w_ema20 + 2 * w_atr14

                if in_position:
                    third = shares // 3
                    exited = False

                    # Nifty crash shield for MW: skip SL if Nifty fell same or more
                    mw_nifty_shields = False
                    if nifty_at_entry > 0 and nifty_close is not None:
                        nifty_now = float(nifty_close.iloc[i])
                        nifty_pct = (nifty_now - nifty_at_entry) / nifty_at_entry
                        stock_pct = (price - entry_price) / entry_price
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            mw_nifty_shields = True

                    # Hard SL: 8% initial, 3% after P1, breakeven after P2
                    if mw_partial_stage >= 2:
                        mw_sl_price = entry_price
                        sl_label = "BREAKEVEN_SL"
                    elif mw_partial_stage >= 1:
                        mw_sl_price = entry_price * 0.97
                        sl_label = "HARD_SL_3PCT"
                    else:
                        mw_sl_price = entry_price * 0.94
                        sl_label = "HARD_SL_6PCT"
                    if not mw_nifty_shields and price <= mw_sl_price:
                        trades.append(self._make_trade(
                            entry_date, entry_price, mw_remaining, day,
                            price, sl_label))
                        exited = True

                    # 2-stage partial exits (+6% sell 1/3, +10% sell 1/3)
                    if not exited and mw_partial_stage == 0 and price >= entry_price * 1.06 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_6PCT_1of3"))
                        mw_remaining = shares - third
                        mw_partial_stage = 1
                    elif not exited and mw_partial_stage == 1 and price >= entry_price * 1.10 and third > 0:
                        trades.append(self._make_trade(
                            entry_date, entry_price, third, day,
                            price, "PARTIAL_10PCT_2of3"))
                        mw_remaining = shares - 2 * third
                        mw_partial_stage = 2

                    # Weekly upper Keltner exit on remaining (only after first partial)
                    if not exited and mw_partial_stage >= 1 and mw_upper_keltner > 0 and price >= mw_upper_keltner:
                        trades.append(self._make_trade(
                            entry_date, entry_price, mw_remaining, day,
                            price, "KELTNER_UPPER_EXIT"))
                        exited = True

                    # Underwater exit: 25 trading days below entry
                    if not exited and (i - entry_bar) >= 25 and price < entry_price:
                        trades.append(self._make_trade(
                            entry_date, entry_price, mw_remaining, day,
                            price, "UNDERWATER_EXIT"))
                        exited = True

                    if exited:
                        in_position = False
                        mw_partial_stage = 0
                        mw_remaining = 0
                    continue

                # Entry check: weekly ADX crosses above 20 with DI+ > DI-
                no_gap_down_mw = (prev_close is None or open_price >= prev_close)
                is_green_mw = price > open_price
                if is_green_mw and no_gap_down_mw and ibs > 0.5 and mw_weekly is not None:
                    w_dates = mw_weekly.index
                    day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                    w_before = w_dates[w_dates < day_ts]
                    if len(w_before) >= 2:
                        w_idx = len(w_before) - 1
                        curr_adx = mw_weekly_adx_vals[w_idx]
                        prev_adx = mw_weekly_adx_vals[w_idx - 1]
                        plus_di = mw_weekly_plus_di_vals[w_idx]
                        minus_di = mw_weekly_minus_di_vals[w_idx]
                        if (not np.isnan(curr_adx) and not np.isnan(prev_adx)
                                and curr_adx >= 25 and curr_adx > prev_adx
                                and plus_di > minus_di):
                                shares = int(capital // price)
                                if shares > 0:
                                    entry_price = price
                                    entry_date = day
                                    entry_bar = i
                                    in_position = True
                                    mw_partial_stage = 0
                                    mw_remaining = shares
                                    nifty_at_entry = float(nifty_close.iloc[i]) if nifty_close is not None else 0.0
                continue

        # Close open position at end of backtest
        if in_position:
            last_day = daily.index[bt_indices[-1]].date()
            last_price = float(closes.iloc[bt_indices[-1]])
            if strategy == "T":
                trades.append(self._make_trade(
                    entry_date, entry_price, t_remaining, last_day,
                    last_price, "BACKTEST_END"))
            elif strategy in ("R", "RW"):
                trades.append(self._make_trade(
                    entry_date, entry_price, r_remaining, last_day,
                    last_price, "BACKTEST_END"))
            elif strategy == "MW":
                trades.append(self._make_trade(
                    entry_date, entry_price, mw_remaining, last_day,
                    last_price, "BACKTEST_END"))
            elif partial_exit_done:
                trades.append(self._make_trade(
                    entry_date, entry_price, remaining_shares, last_day,
                    last_price, "BACKTEST_END"))
            else:
                trades.append(self._make_trade(
                    entry_date, entry_price, shares, last_day, last_price,
                    "BACKTEST_END"))

        trading_days_dates = [daily.index[i].date() for i in bt_indices]
        summary = self._calculate_summary(trades, capital)

        return {
            "symbol": symbol,
            "strategy": strategy,
            "exit_target": str(exit_target) if strategy == "J" and exit_target else None,
            "start_date": trading_days_dates[0].isoformat(),
            "end_date": trading_days_dates[-1].isoformat(),
            "trading_days": len(bt_indices),
            "capital": capital,
            "trades": trades,
            "summary": summary,
        }

    def _make_trade(self, entry_date, entry_price, shares, exit_date,
                    exit_price, reason):
        pnl = round((exit_price - entry_price) * shares, 2)
        pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
        holding = (exit_date - entry_date).days
        return {
            "entry_date": entry_date.isoformat(),
            "entry_time": "15:15",
            "entry_price": round(entry_price, 2),
            "shares": shares,
            "exit_date": exit_date.isoformat(),
            "exit_time": "15:15",
            "exit_price": round(exit_price, 2),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding,
            "exit_reason": reason,
        }

    def _empty_summary(self):
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "total_pnl": 0, "total_return_pct": 0,
            "avg_win": 0, "avg_loss": 0, "largest_win": 0, "largest_loss": 0,
            "profit_factor": 0, "avg_holding_days": 0,
        }

    def _calculate_summary(self, trades, capital):
        if not trades:
            return self._empty_summary()

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)

        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

        gross_wins = sum(t["pnl"] for t in wins)
        gross_losses = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 999.99

        # Position-level WR: group trades by (symbol, entry_date)
        from collections import defaultdict
        positions = defaultdict(float)
        for t in trades:
            key = (t.get("symbol", t.get("ticker", "")), t["entry_date"])
            positions[key] += t["pnl"]
        pos_wins = sum(1 for pnl in positions.values() if pnl > 0)
        pos_losses = sum(1 for pnl in positions.values() if pnl <= 0)
        pos_wr = round(pos_wins / len(positions) * 100, 1) if positions else 0

        return {
            "total_trades": len(positions),
            "winning_trades": pos_wins,
            "losing_trades": pos_losses,
            "win_rate": pos_wr,
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / capital * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(max((t["pnl"] for t in trades), default=0), 2),
            "largest_loss": round(min((t["pnl"] for t in trades), default=0), 2),
            "profit_factor": round(min(profit_factor, 999.99), 2),
            "avg_holding_days": round(
                sum(t["holding_days"] for t in trades) / len(trades), 1),
        }

    REASON_LABELS = {
        "5PCT_PARTIAL": "Hit +5% target, sold 50%",
        "10PCT": "Hit +10% target, sold remaining",
        "SUPPORT_BREAK": "Close broke below weekly support stop",
        "CHANDELIER_EXIT": "Chandelier trailing stop triggered",
        "UNDERWATER_EXIT": "Held 10+ days underwater, cut loss",
        "BACKTEST_END": "Position still open at backtest end",
        "PARTIAL_6PCT_1of3": "Hit +6% target, sold 1/3",
        "PARTIAL_8PCT_1of3": "Hit +8% target, sold 1/3",
        "PARTIAL_10PCT_2of3": "Hit +10% target, sold 1/3",
        "PARTIAL_15PCT_2of3": "Hit +15% target, sold 1/3",
        "KELTNER_UPPER_EXIT": "Upper Keltner band reached, sold remaining",
        "HARD_SL_5PCT": "Hard 5% stop-loss triggered",
        "HARD_SL_3PCT": "Tight 3% stop-loss triggered (post partial)",
        "STRUCTURAL_SL": "Structural stop (1% below swing low) hit",
        "PARTIAL_5PCT": "Hit +5% target, sold partial",
    }

    def explain_trade(self, symbol, strategy, entry_date_str):
        """Explain a single trade: setup indicators, entry, exit progression, P&L.

        Args:
            symbol: NSE ticker
            strategy: "J", "T", "R", or "MW"
            entry_date_str: "YYYY-MM-DD"

        Returns dict with {symbol, strategy, setup, entry, exits, result} or {error}.
        """
        from datetime import datetime as dt

        target_date = dt.strptime(entry_date_str, "%Y-%m-%d").date()

        # Run backtest with a 120-day window around target date
        window_start = target_date - timedelta(days=60)
        window_end = target_date + timedelta(days=120)
        if window_end.date() if hasattr(window_end, 'date') else window_end > datetime.now().date():
            window_end = datetime.now()
        else:
            window_end = datetime.combine(window_end, datetime.min.time())

        period_days = (window_end.date() - window_start).days if hasattr(window_end, 'date') else (window_end - window_start).days
        result = self.run(symbol, period_days, strategy=strategy, end_date=window_end,
                          _stop_tolerance=0.5)

        if "error" in result:
            return {"error": result["error"]}

        trades = result.get("trades", [])
        if not trades:
            return {"error": f"No {strategy} trades found for {symbol} in the window around {entry_date_str}"}

        # Find trades matching entry_date — exact match first, then ±1 day fallback
        matched = []
        for t in trades:
            t_entry = dt.strptime(t["entry_date"], "%Y-%m-%d").date()
            if t_entry == target_date:
                matched.append(t)

        if not matched:
            # ±1 day fallback (weekends / holidays)
            for t in trades:
                t_entry = dt.strptime(t["entry_date"], "%Y-%m-%d").date()
                if abs((t_entry - target_date).days) <= 1:
                    matched.append(t)

        if not matched:
            # Wider fallback: find nearest trade within ±10 days
            # (portfolio backtest may enter on different day than single-stock)
            candidates = []
            for t in trades:
                t_entry = dt.strptime(t["entry_date"], "%Y-%m-%d").date()
                gap = abs((t_entry - target_date).days)
                if gap <= 10:
                    candidates.append((gap, t))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                matched = [candidates[0][1]]

        if not matched:
            # List available entry dates for debugging
            available = sorted(set(t["entry_date"] for t in trades))
            return {"error": f"No trade on {entry_date_str}. Nearby entry dates: {', '.join(available[:10])}"}

        entry_date_bt = dt.strptime(matched[0]["entry_date"], "%Y-%m-%d").date()
        entry_price = matched[0]["entry_price"]
        shares_total = sum(t["shares"] for t in matched)

        # Display date: use user's requested date (real entry day) when
        # backtest trade was matched via ±1 fallback.  The backtest fires
        # signals at the signal-bar close (e.g. Feb 25) but the actual
        # trade entry happens the next trading day (Feb 26).
        entry_date_display = target_date

        # Fetch raw OHLCV for indicator snapshots
        nse_symbol = f"{symbol}.NS"
        fetch_start = entry_date_bt - timedelta(days=300)
        try:
            raw = yf.Ticker(nse_symbol).history(start=fetch_start, end=window_end)
        except Exception:
            raw = pd.DataFrame()

        # Fallback to old ticker name if renamed
        if (raw.empty or len(raw) < 50) and symbol in TICKER_ALIASES:
            old_symbol = f"{TICKER_ALIASES[symbol]}.NS"
            try:
                raw = yf.Ticker(old_symbol).history(start=fetch_start, end=window_end)
            except Exception:
                raw = pd.DataFrame()

        if raw.empty or len(raw) < 50:
            return {"error": f"Could not fetch price data for {symbol}"}

        # Find entry bar index — prefer user's requested date, fall back to backtest date
        entry_i = None
        for idx_i, ts in enumerate(raw.index):
            if ts.date() == target_date:
                entry_i = idx_i
                break
        if entry_i is None:
            for idx_i, ts in enumerate(raw.index):
                if ts.date() == entry_date_bt:
                    entry_i = idx_i
                    break

        if entry_i is None:
            return {"error": f"Entry date {target_date} not found in price data"}

        # Signal bar index: the backtest evaluates entry conditions on this bar
        # (entry_i may be +1 day ahead for display purposes)
        signal_i = None
        for idx_i, ts in enumerate(raw.index):
            if ts.date() == entry_date_bt:
                signal_i = idx_i
                break
        if signal_i is None:
            signal_i = entry_i  # fallback

        closes = raw["Close"]
        opens = raw["Open"]
        highs = raw["High"]
        lows = raw["Low"]

        # Build setup explanation
        setup = {"strategy": strategy}

        if strategy == "J":
            # Weekly support, IBS, CCI, distance
            weekly = raw.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            w_support = weekly["Close"].rolling(window=26, min_periods=26).min().shift(2)
            w_support_daily = w_support.reindex(raw.index, method="ffill")
            w_low_stop = weekly["Low"].rolling(window=26, min_periods=26).min().shift(2)
            w_low_stop_daily = w_low_stop.reindex(raw.index, method="ffill")

            ws = float(w_support_daily.iloc[signal_i]) if not pd.isna(w_support_daily.iloc[signal_i]) else 0
            wls = float(w_low_stop_daily.iloc[signal_i]) if not pd.isna(w_low_stop_daily.iloc[signal_i]) else 0

            # Find which week formed the support/stop levels
            signal_date_val = raw.index[signal_i]
            weekly_idx = weekly.index.searchsorted(signal_date_val)
            ws_formed_week = ""
            wls_formed_week = ""
            if weekly_idx >= 2:
                end_idx = weekly_idx - 2  # shift(2) means we skip last 2
                start_idx = max(0, end_idx - 26)
                window = weekly.iloc[start_idx:end_idx]
                if len(window) > 0:
                    ws_week_idx = window["Close"].idxmin()
                    ws_formed_week = str(ws_week_idx.date()) if pd.notna(ws_week_idx) else ""
                    wls_week_idx = window["Low"].idxmin()
                    wls_formed_week = str(wls_week_idx.date()) if pd.notna(wls_week_idx) else ""

            # Use signal bar (backtest date) for setup indicators, not display date
            hl_range = highs.iloc[signal_i] - lows.iloc[signal_i]
            ibs_val = float((closes.iloc[signal_i] - lows.iloc[signal_i]) / hl_range) if hl_range > 0 else 0.5
            cci_series = self._calculate_cci_series(highs, lows, closes, 20)
            cci_val = float(cci_series.iloc[signal_i]) if not pd.isna(cci_series.iloc[signal_i]) else 0

            sig_price = float(closes.iloc[signal_i])
            dist_pct = round(((sig_price - ws) / ws) * 100, 2) if ws > 0 else 0
            stop_dist = round(((entry_price - wls) / entry_price) * 100, 2) if wls > 0 else 0

            setup["description"] = f"{symbol} was trading near its 26-week support level (weekly close support at {ws:,.2f})"
            setup["weekly_support"] = round(ws, 2)
            setup["weekly_support_formed"] = ws_formed_week
            setup["weekly_low_stop"] = round(wls, 2)
            setup["weekly_low_stop_formed"] = wls_formed_week
            setup["ibs"] = round(ibs_val, 2)
            setup["cci20"] = round(cci_val, 1)
            setup["distance_from_support"] = f"{dist_pct}%"
            setup["stop_distance"] = f"{stop_dist}%"

        elif strategy == "T":
            # EMA20, upper Keltner, last touch, pullback depth, ATR14
            ema20_series = closes.ewm(span=20, adjust=False).mean()
            prev_close_s = closes.shift(1)
            tr1 = highs - lows
            tr2 = (highs - prev_close_s).abs()
            tr3 = (lows - prev_close_s).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr14_series = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

            ema20_val = float(ema20_series.iloc[signal_i])
            atr14_val = float(atr14_series.iloc[signal_i]) if not pd.isna(atr14_series.iloc[signal_i]) else 0
            upper_k = ema20_val + 2 * atr14_val

            # Find last touch of upper Keltner
            last_touch_date = None
            pullback_high = 0
            for lb in range(signal_i - 1, max(signal_i - 11, -1), -1):
                past_h = float(highs.iloc[lb])
                past_ema20 = float(ema20_series.iloc[lb])
                past_atr14 = float(atr14_series.iloc[lb]) if not pd.isna(atr14_series.iloc[lb]) else 0
                past_upper = past_ema20 + 2 * past_atr14
                if past_h >= past_upper:
                    last_touch_date = raw.index[lb].strftime("%b %d")
                    pullback_high = past_h
                    break

            pullback_pct = round(((pullback_high - entry_price) / pullback_high) * 100, 2) if pullback_high > 0 else 0

            setup["description"] = f"{symbol} pulled back to EMA(20) after touching upper Keltner band"
            setup["ema20"] = round(ema20_val, 2)
            setup["upper_keltner"] = round(upper_k, 2)
            setup["atr14"] = round(atr14_val, 2)
            setup["last_keltner_touch"] = last_touch_date or "N/A"
            setup["pullback_depth"] = f"{pullback_pct}%"
            setup["stop"] = f"{round(entry_price * 0.95, 2)} (5% hard SL)"

        elif strategy == "R":
            # Swing lows, RSI values, divergence detail
            rsi14_series = self._calculate_rsi_series(closes, 14)
            swing_lows = self._find_swing_lows(lows)
            rsi14_vals = rsi14_series.values
            lows_vals = lows.values

            # Try regular divergence first, then hidden
            found, detail = self._explain_bullish_divergence(
                lows_vals, rsi14_vals, signal_i, swing_lows)
            div_type = "regular"

            if not found:
                found, detail = self._explain_hidden_bullish_divergence(
                    lows_vals, rsi14_vals, signal_i, swing_lows)
                div_type = "hidden"

            if found and detail:
                prev_date = raw.index[detail["prev_idx"]].strftime("%b %d")
                curr_date = raw.index[detail["curr_idx"]].strftime("%b %d")
                struct_stop = round(detail["curr_low"] * 0.99, 2)
                stop_pct = round(((entry_price - struct_stop) / entry_price) * 100, 1)

                if div_type == "hidden":
                    rsi_div = round(detail["prev_rsi"] - detail["curr_rsi"], 1)
                    price_rise = round(((detail["curr_low"] - detail["prev_low"]) / detail["prev_low"]) * 100, 1)
                    setup["description"] = f"{symbol} showed hidden bullish RSI divergence — price made higher low but RSI made lower low (continuation in uptrend)"
                    setup["swing_low_1"] = {
                        "date": prev_date,
                        "price": round(detail["prev_low"], 2),
                        "rsi": round(detail["prev_rsi"], 1),
                    }
                    setup["swing_low_2"] = {
                        "date": curr_date,
                        "price": round(detail["curr_low"], 2),
                        "rsi": round(detail["curr_rsi"], 1),
                    }
                    setup["rsi_divergence"] = f"-{rsi_div} points"
                    setup["price_rise"] = f"+{price_rise}%"
                    setup["div_type"] = "hidden"
                else:
                    rsi_div = round(detail["curr_rsi"] - detail["prev_rsi"], 1)
                    price_drop = round(((detail["curr_low"] - detail["prev_low"]) / detail["prev_low"]) * 100, 1)
                    setup["description"] = f"{symbol} showed bullish RSI divergence — price made lower low but RSI made higher low"
                    setup["swing_low_1"] = {
                        "date": prev_date,
                        "price": round(detail["prev_low"], 2),
                        "rsi": round(detail["prev_rsi"], 1),
                    }
                    setup["swing_low_2"] = {
                        "date": curr_date,
                        "price": round(detail["curr_low"], 2),
                        "rsi": round(detail["curr_rsi"], 1),
                    }
                    setup["rsi_divergence"] = f"+{rsi_div} points"
                    setup["price_drop"] = f"{price_drop}%"
                    setup["div_type"] = "regular"

                setup["structural_stop"] = struct_stop
                setup["stop_distance"] = f"{stop_pct}%"
            else:
                setup["description"] = f"{symbol} had bullish RSI divergence signal (detail unavailable for exact entry bar)"

        elif strategy == "MW":
            # Weekly ADX trend entry explanation
            weekly_explain = raw.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            mw_adx_e, mw_pdi_e, mw_mdi_e = self._calculate_adx_series(
                weekly_explain["High"], weekly_explain["Low"], weekly_explain["Close"])

            signal_date_val = raw.index[signal_i]
            w_dates = weekly_explain.index
            day_ts = pd.Timestamp(signal_date_val).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(signal_date_val)
            w_before = w_dates[w_dates < day_ts]
            if len(w_before) >= 2:
                w_idx = len(w_before) - 1
                adx_val = float(mw_adx_e.iloc[w_idx])
                pdi_val = float(mw_pdi_e.iloc[w_idx])
                mdi_val = float(mw_mdi_e.iloc[w_idx])
                signal_week = w_dates[w_idx].strftime("%b %d")
                setup["description"] = f"{symbol} weekly ADX crossed above 25 with DI+ > DI- (bullish trend confirmed)"
                setup["signal_week"] = signal_week
                setup["adx"] = round(adx_val, 1)
                setup["plus_di"] = round(pdi_val, 1)
                setup["minus_di"] = round(mdi_val, 1)
                setup["stop_distance"] = "8.0%"
            else:
                setup["description"] = f"{symbol} had weekly ADX trend signal (detail unavailable)"

        # Entry details — use signal bar (backtest evaluates and enters at signal bar close)
        o = round(float(opens.iloc[signal_i]), 2)
        h = round(float(highs.iloc[signal_i]), 2)
        l = round(float(lows.iloc[signal_i]), 2)
        c = round(float(closes.iloc[signal_i]), 2)

        entry_info = {
            "date": entry_date_display.strftime("%b %d, %Y"),
            "price": entry_price,
            "open": o, "high": h, "low": l, "close": c,
            "shares": shares_total,
            "capital": round(entry_price * shares_total, 2),
        }
        if strategy == "J" and "weekly_low_stop" in setup:
            entry_info["stop"] = setup["weekly_low_stop"]
            entry_info["stop_type"] = "Weekly Low Support"
            entry_info["stop_pct"] = setup.get("stop_distance", "")
        elif strategy == "T":
            entry_info["stop"] = round(entry_price * 0.95, 2)
            entry_info["stop_type"] = "Hard 5% SL"
            entry_info["stop_pct"] = "5.0%"
        elif strategy == "R" and "structural_stop" in setup:
            entry_info["stop"] = setup["structural_stop"]
            entry_info["stop_type"] = "Structural (1% below swing low)"
            entry_info["stop_pct"] = setup.get("stop_distance", "")
        elif strategy == "MW":
            entry_info["stop"] = round(entry_price * 0.94, 2)
            entry_info["stop_type"] = "Hard 8% SL"
            entry_info["stop_pct"] = "8.0%"

        # Exit progression
        exits = []
        for idx, t in enumerate(matched):
            pnl_pct = t["pnl_pct"]
            label = self.REASON_LABELS.get(t["exit_reason"], t["exit_reason"])
            exits.append({
                "stage": idx + 1,
                "date": dt.strptime(t["exit_date"], "%Y-%m-%d").date().strftime("%b %d"),
                "price": t["exit_price"],
                "shares": t["shares"],
                "pnl_pct": f"{'+' if pnl_pct >= 0 else ''}{pnl_pct}%",
                "pnl": t["pnl"],
                "reason": label,
                "reason_code": t["exit_reason"],
            })

        # Result summary
        total_pnl = round(sum(t["pnl"] for t in matched), 2)
        total_capital = round(entry_price * shares_total, 2)
        total_pnl_pct = round((total_pnl / total_capital) * 100, 1) if total_capital > 0 else 0
        last_exit = dt.strptime(matched[-1]["exit_date"], "%Y-%m-%d").date()
        holding_days = (last_exit - entry_date_display).days

        result_info = {
            "winner": total_pnl > 0,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "holding_days": holding_days,
        }

        return {
            "symbol": symbol,
            "strategy": strategy,
            "setup": setup,
            "entry": entry_info,
            "exits": exits,
            "result": result_info,
        }

    def run_portfolio_backtest(self, period_days, universe=50,
                              capital_lakhs=10, per_stock=50000,
                              strategies=None, entries_per_day=1,
                              progress_callback=None, end_date=None,
                              three_stage_exit=True, seed=42,
                              no_gap_down=True, rank_by_risk=True,
                              t_target1=0.06, t_target2=0.10,
                              underwater_exit_days=None,
                              t_tight_sl=None,
                              rank_by_sector_momentum=False,
                              mw_adx_threshold=25,
                              rs_entry_filters=None,
                              rs_regime_mode="asymmetric",
                              rs_hard_sl=0.90,
                              rs_uw_days=0,
                              rs_dist_high_pct=0.03,
                              rs_entry_mode="default",
                              rs_ibd_lookback=12,
                              rs_ibd_filters=None,
                              rs_sl_cooldown=20,
                              rs_ibd_min_rating=80,
                              rs_ibd_max_rating=99,
                              rs_ibd_skip_top=0,
                              rs_ibd_consec_days=0,
                              rs_ibd_rank_by_rating=True,
                              rs_underperform_thresh=0):
        """
        Portfolio-level backtest with configurable capital and strategies.
        capital_lakhs: 10 or 20 (total capital in lakhs)
        strategies: list of strategy codes, e.g. ["J"], ["T"], ["J","T"]
        ₹50K per stock, max positions = capital / 50K.
        """
        from data.momentum_engine import NIFTY_50_TICKERS

        if strategies is None:
            strategies = ["J", "T"]

        TOTAL_CAPITAL = capital_lakhs * 100000
        PER_STOCK = per_stock
        MAX_POSITIONS = TOTAL_CAPITAL // PER_STOCK

        if universe <= 50:
            tickers = NIFTY_50_TICKERS
        elif universe <= 100:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS
        elif universe <= 200:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS
        else:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS + NIFTY_500_BEYOND200_TICKERS

        end_date = end_date or datetime.now()
        daily_start = end_date - timedelta(days=period_days + 500)
        bt_start_date = (end_date - timedelta(days=period_days)).date()

        # --- Phase 1: Fetch all data ---
        stock_data = {}  # ticker -> DataFrame
        total = len(tickers)

        # Fetch Nifty index data once
        try:
            nifty_raw = yf.Ticker("^NSEI").history(start=daily_start, end=end_date)
        except Exception:
            nifty_raw = pd.DataFrame()

        # Fetch benchmark index for RS calculation — Nifty 200 for universe > 100
        bench_symbol = "^NSEI" if universe <= 100 else "^CNX200"
        try:
            if bench_symbol == "^NSEI" and not nifty_raw.empty:
                bench_raw = nifty_raw
            else:
                bench_raw = yf.Ticker(bench_symbol).history(start=daily_start, end=end_date)
        except Exception:
            bench_raw = nifty_raw
        bench_ret123 = pd.Series(dtype=float)
        bench_ret21 = pd.Series(dtype=float)
        if not bench_raw.empty:
            bench_ret123 = (bench_raw["Close"] / bench_raw["Close"].shift(123) - 1) * 100
            bench_ret21 = (bench_raw["Close"] / bench_raw["Close"].shift(21) - 1) * 100

        # Fetch sector index data for sector momentum ranking
        sector_index_data = {}  # sector_name -> closes Series
        if rank_by_sector_momentum:
            from sector_mapping import STOCK_SECTOR_MAP
            from data.live_signals_engine import SECTORAL_INDICES
            for sec_name, sec_symbol in SECTORAL_INDICES.items():
                try:
                    sec_df = yf.Ticker(sec_symbol).history(start=daily_start, end=end_date)
                    if not sec_df.empty and len(sec_df) > 30:
                        sector_index_data[sec_name] = sec_df["Close"]
                except Exception:
                    pass

        for idx, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(idx + 1, total, f"Fetching {ticker}")

            nse_symbol = f"{ticker}.NS"
            try:
                daily = yf.Ticker(nse_symbol).history(start=daily_start, end=end_date)
            except Exception:
                daily = pd.DataFrame()

            # Fallback to old ticker name if renamed
            if (daily.empty or len(daily) < 210) and ticker in TICKER_ALIASES:
                old_symbol = f"{TICKER_ALIASES[ticker]}.NS"
                try:
                    daily = yf.Ticker(old_symbol).history(start=daily_start, end=end_date)
                except Exception:
                    daily = pd.DataFrame()

            if daily.empty or len(daily) < 210:
                continue

            stock_data[ticker] = daily

        if not stock_data:
            return {"error": "No valid stock data", "trades": [],
                    "summary": self._empty_summary()}

        # --- Phase 2: Pre-compute indicators per stock ---
        indicators = {}  # ticker -> dict of series

        for ticker, daily in stock_data.items():
            closes = daily["Close"]
            opens = daily["Open"]
            highs = daily["High"]
            lows = daily["Low"]

            hl_range = highs - lows
            ibs_series = ((closes - lows) / hl_range).where(hl_range > 0, 0.5)

            # Support levels
            support_6m = lows.rolling(window=120, min_periods=120).min().shift(1)
            support_1y = lows.rolling(window=252, min_periods=252).min().shift(1)
            # Weekly support for Strategy J
            weekly = daily.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            # Skip last 2 weeks of support (use proven support, not recent noise)
            w_support = weekly["Close"].rolling(window=26, min_periods=26).min().shift(2)
            w_low_stop = weekly["Low"].rolling(window=26, min_periods=26).min().shift(2)
            weekly_support_series = w_support.reindex(daily.index, method="ffill")
            weekly_low_stop_series = w_low_stop.reindex(daily.index, method="ffill")

            # Volume
            vol_series = daily["Volume"].astype(float)
            vol_avg20 = vol_series.rolling(window=20, min_periods=20).mean()

            # EMAs
            ema5_series = closes.ewm(span=5, adjust=False).mean()
            ema8_series = closes.ewm(span=8, adjust=False).mean()
            ema10_series = closes.ewm(span=10, adjust=False).mean()
            ema20_series = closes.ewm(span=20, adjust=False).mean()
            ema50_series = closes.ewm(span=50, adjust=False).mean()
            ema200_series = closes.ewm(span=200, adjust=False).mean()

            # RSI(2), RSI(3), RSI(14)
            rsi2_series = self._calculate_rsi_series(closes, 2)
            rsi3_series = self._calculate_rsi_series(closes, 3)
            rsi14_series = self._calculate_rsi_series(closes, 14)

            # Swing lows for Strategy R
            swing_lows = self._find_swing_lows(lows)

            # Weekly ADX for Strategy MW
            mw_adx_s, mw_plus_di_s, mw_minus_di_s = self._calculate_adx_series(
                weekly["High"], weekly["Low"], weekly["Close"])
            mw_weekly_adx_vals = mw_adx_s.values
            mw_weekly_plus_di_vals = mw_plus_di_s.values
            mw_weekly_minus_di_vals = mw_minus_di_s.values

            # Weekly upper Keltner for Strategy MW exit
            w_closes_k = weekly["Close"]
            w_ema20_k = w_closes_k.ewm(span=20, adjust=False).mean()
            w_prev_close_k = w_closes_k.shift(1)
            w_tr_k = pd.concat([weekly["High"] - weekly["Low"],
                                (weekly["High"] - w_prev_close_k).abs(),
                                (weekly["Low"] - w_prev_close_k).abs()], axis=1).max(axis=1)
            w_atr14_k = w_tr_k.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            mw_weekly_upper_keltner = w_ema20_k + 2 * w_atr14_k

            # Weekly RSI14 and swing lows for Strategy RW
            weekly_rsi14_series = self._calculate_rsi_series(weekly["Close"], 14)
            weekly_lows_for_swings = weekly["Low"]
            weekly_swing_lows = self._find_swing_lows(weekly_lows_for_swings, left=3, right=2)
            # Pre-compute weekly arrays for divergence detection (avoid recomputing per day)
            weekly_lows_vals = weekly["Low"].values
            weekly_rsi14_vals = weekly_rsi14_series.values

            # Weekly indicators for Strategy WT (Weekly Trend)
            wt_w_closes = weekly["Close"]
            wt_w_ema20 = wt_w_closes.ewm(span=20, adjust=False).mean()
            wt_w_ema50 = wt_w_closes.ewm(span=50, adjust=False).mean()
            wt_w_high20 = wt_w_closes.rolling(20).max()
            wt_w_slope = (wt_w_ema50 / wt_w_ema50.shift(4) - 1) * 100 / 4
            wt_w_gap = wt_w_ema20 - wt_w_ema50
            wt_w_gap_pct = (wt_w_ema20 - wt_w_ema50) / wt_w_ema50 * 100
            wt_w_vol = weekly["Volume"]
            wt_w_vol_avg20 = wt_w_vol.rolling(20).mean()
            wt_w_return = (wt_w_closes / wt_w_closes.shift(1) - 1) * 100

            # CCI(20) for Strategy J entry confirmation
            cci20_series = self._calculate_cci_series(highs, lows, closes, 20)

            # ATR(14) for Strategy T (Keltner Channel)
            prev_close_s = closes.shift(1)
            tr1 = highs - lows
            tr2 = (highs - prev_close_s).abs()
            tr3 = (lows - prev_close_s).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr14_series = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

            # Backtest indices (need 200+ bars warmup)
            bt_indices = [i for i, ts in enumerate(daily.index)
                          if ts.date() >= bt_start_date and i >= 200]

            indicators[ticker] = {
                "daily": daily,
                "closes": closes,
                "opens": opens,
                "highs": highs,
                "lows": lows,
                "ibs": ibs_series,
                "support_6m": support_6m,
                "support_1y": support_1y,
                "weekly_support": weekly_support_series,
                "weekly_low_stop": weekly_low_stop_series,
                "rsi2": rsi2_series,
                "rsi3": rsi3_series,
                "rsi14": rsi14_series,
                "swing_lows": swing_lows,
                "mw_weekly_adx_vals": mw_weekly_adx_vals,
                "mw_weekly_plus_di_vals": mw_weekly_plus_di_vals,
                "mw_weekly_minus_di_vals": mw_weekly_minus_di_vals,
                "mw_weekly_upper_keltner": mw_weekly_upper_keltner,
                "mw_weekly_close": w_closes_k,
                "mw_weekly_ema20": w_ema20_k,
                "cci20": cci20_series,
                "volume": vol_series,
                "vol_avg20": vol_avg20,
                "atr14": atr14_series,
                "ema5": ema5_series,
                "ema8": ema8_series,
                "ema10": ema10_series,
                "ema20": ema20_series,
                "ema50": ema50_series,
                "ema200": ema200_series,
                "bt_indices": bt_indices,
                "weekly_raw": weekly,
                "weekly_swing_lows": weekly_swing_lows,
                "weekly_lows_vals": weekly_lows_vals,
                "weekly_rsi14_vals": weekly_rsi14_vals,
                # WT (Weekly Trend) indicators — all shifted by 1 week for confirmed candle
                "wt_w_ema50": wt_w_ema50,  # unshifted for exit
                "wt_w_close_prev": wt_w_closes.shift(1),
                "wt_w_ema20_prev": wt_w_ema20.shift(1),
                "wt_w_ema50_prev": wt_w_ema50.shift(1),
                "wt_w_high20_prev": wt_w_high20.shift(1),
                "wt_w_slope_prev": wt_w_slope.shift(1),
                "wt_w_gap_prev": wt_w_gap.shift(1),
                "wt_w_gap_prev2": wt_w_gap.shift(2),
                "wt_w_gap_pct_prev": wt_w_gap_pct.shift(1),
                "wt_w_vol_prev": wt_w_vol.shift(1),
                "wt_w_vol_avg_prev": wt_w_vol_avg20.shift(1),
                "wt_w_return_prev": wt_w_return.shift(1),
                # Stock 123-day (6-month) return for RS filter
                "stock_ret123": (closes / closes.shift(123) - 1) * 100,
                # Benchmark 123-day return aligned to stock's dates
                "bench_ret123": bench_ret123.reindex(daily.index, method="ffill") if not bench_ret123.empty else pd.Series(0.0, index=daily.index),
                # RS Rotation strategy indicators
                "rs_21d": (closes / closes.shift(21) - 1) * 100 - (bench_ret21.reindex(daily.index, method="ffill") if not bench_ret21.empty else pd.Series(0.0, index=daily.index)),
                "rs_123d": (closes / closes.shift(123) - 1) * 100 - (bench_ret123.reindex(daily.index, method="ffill") if not bench_ret123.empty else pd.Series(0.0, index=daily.index)),
                "rs_123d_smooth": (lambda sc5, bc5: (sc5 / sc5.shift(123) - 1) * 100 - ((bc5 / bc5.shift(123) - 1) * 100 if not bench_ret123.empty else pd.Series(0.0, index=daily.index)))(closes.rolling(5, center=True, min_periods=3).mean(), bench_raw["Close"].reindex(daily.index, method="ffill").rolling(5, center=True, min_periods=3).mean() if not bench_raw.empty else pd.Series(0.0, index=daily.index)),
                "rs_123d_ma100": ((closes / closes.shift(123) - 1) * 100 - (bench_ret123.reindex(daily.index, method="ffill") if not bench_ret123.empty else pd.Series(0.0, index=daily.index))).rolling(100).mean(),
                # IBD-style weighted RS: 40% Q4 (0-63d) + 20% Q3 (63-126d) + 20% Q2 (126-189d) + 20% Q1 (189-252d)
                "rs_weighted": (
                    0.4 * (closes / closes.shift(63) - 1) * 100 +
                    0.2 * (closes.shift(63) / closes.shift(126) - 1) * 100 +
                    0.2 * (closes.shift(126) / closes.shift(189) - 1) * 100 +
                    0.2 * (closes.shift(189) / closes.shift(252) - 1) * 100
                ),
                # IBD 6-month weighted RS: 50% latest quarter + 30% prior quarter + 20% quarter before
                "rs_weighted_6m": (
                    0.5 * (closes / closes.shift(63) - 1) * 100 +
                    0.3 * (closes.shift(63) / closes.shift(126) - 1) * 100 +
                    0.2 * (closes.shift(126) / closes.shift(189) - 1) * 100
                ),
                # 20-week EMA reindexed to daily for RS exit
                "rs_ema20w_daily": weekly["Close"].ewm(span=20, adjust=False).mean().reindex(daily.index, method="ffill"),
                # 30-week EMA reindexed to daily for RS entry/exit
                "rs_ema30w_daily": weekly["Close"].ewm(span=30, adjust=False).mean().reindex(daily.index, method="ffill"),
            }

        # Pre-compute rolling sector RS momentum (20-day RS, 5-day delta)
        sector_momentum_by_date = {}  # date -> {sector_name: momentum_score}
        if rank_by_sector_momentum and not nifty_raw.empty and sector_index_data:
            from sector_mapping import STOCK_SECTOR_MAP as _SECTOR_MAP
            nifty_closes = nifty_raw["Close"]
            rs_period = 20
            for sec_name, sec_closes in sector_index_data.items():
                common = nifty_closes.index.intersection(sec_closes.index)
                if len(common) < rs_period + 10:
                    continue
                n_al = nifty_closes.reindex(common)
                s_al = sec_closes.reindex(common)
                # Compute rolling RS at each date
                n_ret = n_al.pct_change(rs_period) * 100
                s_ret = s_al.pct_change(rs_period) * 100
                rs_series = s_ret - n_ret
                rs_delta_5 = rs_series - rs_series.shift(5)
                rs_delta_10 = rs_series - rs_series.shift(10)
                # momentum = delta_5d * 3 + delta_10d * 2
                mom_series = rs_delta_5 * 3 + rs_delta_10 * 2
                for ts, mom_val in mom_series.items():
                    d = ts.date()
                    if pd.isna(mom_val):
                        continue
                    if d not in sector_momentum_by_date:
                        sector_momentum_by_date[d] = {}
                    sector_momentum_by_date[d][sec_name] = float(mom_val)

        # --- Phase 3: Build union of all trading dates ---
        all_dates = set()
        for ticker, ind in indicators.items():
            for i in ind["bt_indices"]:
                all_dates.add(ind["daily"].index[i].date())
        all_dates = sorted(all_dates)

        if not all_dates:
            return {"error": "No trading days in selected period",
                    "trades": [], "summary": self._empty_summary()}

        # Build Nifty close lookup by date (for SL filter & 3-day low weakness)
        nifty_close_by_date = {}  # date -> close price
        if not nifty_raw.empty:
            for ts, row in nifty_raw.iterrows():
                nifty_close_by_date[ts.date()] = float(row["Close"])

        # Pre-compute Nifty regime filter for RS strategy
        nifty_rs_regime_ok = {}  # date -> bool
        if not nifty_raw.empty:
            nifty_weekly = nifty_raw["Close"].resample("W-FRI").last().dropna()
            nifty_20w_ema = nifty_weekly.ewm(span=20, adjust=False).mean()
            nifty_10w_ema = nifty_weekly.ewm(span=10, adjust=False).mean()
            # Forward-fill weekly EMAs to daily
            nifty_20w_ema_daily = nifty_20w_ema.reindex(nifty_raw.index, method="ffill")
            nifty_10w_ema_daily = nifty_10w_ema.reindex(nifty_raw.index, method="ffill")
            # Weekly data for regime detection
            nifty_weekly_closes = nifty_weekly  # already computed above
            nifty_weekly_lows = nifty_raw["Low"].resample("W-FRI").min().dropna() if "Low" in nifty_raw.columns else pd.Series(dtype=float)
            nifty_10w_ema_weekly = nifty_10w_ema  # weekly 10w EMA

            regime_on = True  # start optimistic
            for ts in nifty_raw.index:
                d = ts.date()
                close_val = nifty_raw["Close"].get(ts)
                ema20 = nifty_20w_ema_daily.get(ts)
                ema10 = nifty_10w_ema_daily.get(ts)
                if pd.notna(close_val) and pd.notna(ema20) and pd.notna(ema10):
                    cv = float(close_val)
                    e20 = float(ema20)
                    e10 = float(ema10)
                    if rs_regime_mode == "simple":
                        # Simple: ON when close > 20w EMA, OFF otherwise
                        regime_on = cv >= e20
                    elif rs_regime_mode == "early_10w_2":
                        # OFF below 20w EMA, early ON: above 10w + 10w rising 2 weeks
                        if cv >= e20:
                            regime_on = True
                        elif regime_on and cv < e20:
                            regime_on = False
                        elif not regime_on and cv > e10:
                            # Check 10w EMA rising for 2 consecutive weeks
                            wi = nifty_10w_ema_weekly.index.get_indexer([ts], method="ffill")[0]
                            if wi >= 2:
                                e10_now = float(nifty_10w_ema_weekly.iloc[wi])
                                e10_1w = float(nifty_10w_ema_weekly.iloc[wi-1])
                                e10_2w = float(nifty_10w_ema_weekly.iloc[wi-2])
                                if e10_now > e10_1w > e10_2w:
                                    regime_on = True
                    elif rs_regime_mode == "early_10w_3":
                        # Same but 10w rising 3 consecutive weeks
                        if cv >= e20:
                            regime_on = True
                        elif regime_on and cv < e20:
                            regime_on = False
                        elif not regime_on and cv > e10:
                            wi = nifty_10w_ema_weekly.index.get_indexer([ts], method="ffill")[0]
                            if wi >= 3:
                                e10_now = float(nifty_10w_ema_weekly.iloc[wi])
                                e10_1w = float(nifty_10w_ema_weekly.iloc[wi-1])
                                e10_2w = float(nifty_10w_ema_weekly.iloc[wi-2])
                                e10_3w = float(nifty_10w_ema_weekly.iloc[wi-3])
                                if e10_now > e10_1w > e10_2w > e10_3w:
                                    regime_on = True
                    elif rs_regime_mode == "gap_3pct":
                        # OFF below 20w, early ON: within 3% of 20w + 10w rising
                        if cv >= e20:
                            regime_on = True
                        elif regime_on and cv < e20:
                            regime_on = False
                        elif not regime_on:
                            gap_pct = (e20 - cv) / e20 * 100
                            wi = nifty_10w_ema_weekly.index.get_indexer([ts], method="ffill")[0]
                            if wi >= 1 and gap_pct <= 3.0:
                                e10_now = float(nifty_10w_ema_weekly.iloc[wi])
                                e10_1w = float(nifty_10w_ema_weekly.iloc[wi-1])
                                if e10_now > e10_1w:
                                    regime_on = True
                    elif rs_regime_mode == "higher_lows":
                        # OFF below 20w, early ON: 2 consecutive weekly higher lows + above 10w
                        if cv >= e20:
                            regime_on = True
                        elif regime_on and cv < e20:
                            regime_on = False
                        elif not regime_on and cv > e10:
                            wi = nifty_weekly_lows.index.get_indexer([ts], method="ffill")[0]
                            if wi >= 2:
                                lo0 = float(nifty_weekly_lows.iloc[wi])
                                lo1 = float(nifty_weekly_lows.iloc[wi-1])
                                lo2 = float(nifty_weekly_lows.iloc[wi-2])
                                if lo0 > lo1 > lo2:
                                    regime_on = True
                    else:
                        # Asymmetric: instant OFF when < 20w, resume only on 10w > 20w
                        if regime_on and cv < e20:
                            regime_on = False
                        elif not regime_on and e10 > e20:
                            regime_on = True
                nifty_rs_regime_ok[d] = regime_on

        # Build per-stock date->index mapping
        date_to_idx = {}  # ticker -> {date: iloc_index}
        for ticker, ind in indicators.items():
            mapping = {}
            for i in ind["bt_indices"]:
                d = ind["daily"].index[i].date()
                mapping[d] = i
            date_to_idx[ticker] = mapping

        # Pre-compute IBD-style RS Rating percentile (1-99) for each stock on each date
        rs_rating_by_date = {}  # date -> {ticker: percentile_1_99}
        if "RS" in strategies:
            for day in all_dates:
                day_scores = {}
                for ticker, ind in indicators.items():
                    if day not in date_to_idx.get(ticker, {}):
                        continue
                    ci = date_to_idx[ticker][day]
                    rs_w_key = "rs_weighted_6m" if rs_ibd_lookback == 6 else "rs_weighted"
                    rs_w = ind.get(rs_w_key)
                    if rs_w is not None and ci < len(rs_w) and not pd.isna(rs_w.iloc[ci]):
                        day_scores[ticker] = float(rs_w.iloc[ci])
                if len(day_scores) >= 10:
                    sorted_tickers = sorted(day_scores.keys(), key=lambda t: day_scores[t])
                    n = len(sorted_tickers)
                    ratings = {}
                    for rank_i, t in enumerate(sorted_tickers):
                        ratings[t] = max(1, min(99, int(round((rank_i + 1) / n * 99))))
                    rs_rating_by_date[day] = ratings

        # --- Phase 4: Day-by-day simulation ---
        positions: List[Dict] = []  # open positions
        trades: List[Dict] = []     # completed trades
        total_signals = 0
        missed_signals = 0
        max_positions_used = 0
        positions_over_time = []
        running_pnl = 0  # cumulative realized PnL
        rs_cooldown = {}  # ticker -> day_idx when cooldown expires (RS rotation)
        liquid_fund_income = 0.0  # cumulative income from idle capital in liquid fund
        LIQUID_FUND_DAILY_RATE = 0.04 / 252  # ~4% annualized, per trading day

        for day_idx, day in enumerate(all_dates):
            if progress_callback and day_idx % 20 == 0:
                progress_callback(
                    total + day_idx, total + len(all_dates),
                    f"Simulating {day}")

            # Check if Nifty is weak today (close < 3-day low close)
            nifty_weak_today = False
            nifty_today_close = nifty_close_by_date.get(day, 0.0)
            if day in nifty_close_by_date and day_idx >= 3:
                past_nifty_closes = [nifty_close_by_date[d]
                                     for d in all_dates[max(0, day_idx-3):day_idx]
                                     if d in nifty_close_by_date]
                if past_nifty_closes:
                    nifty_3day_low_close = min(past_nifty_closes)
                    nifty_weak_today = nifty_today_close < nifty_3day_low_close

            # === 1. Process exits first (frees capital) ===
            still_open = []
            for pos in positions:
                ticker = pos["symbol"]
                if day not in date_to_idx.get(ticker, {}):
                    still_open.append(pos)
                    continue

                i = date_to_idx[ticker][day]
                ind = indicators[ticker]
                price = float(ind["closes"].iloc[i])
                high = float(ind["highs"].iloc[i])
                low = float(ind["lows"].iloc[i])
                open_price = float(ind["opens"].iloc[i])

                exited = False

                if pos["strategy"] == "J":
                    entry_stop = pos["entry_stop_j"]

                    # Nifty drop shield: skip support break if Nifty fell same or more
                    j_nifty_shields = False
                    j_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if j_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - j_nifty_entry) / j_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            j_nifty_shields = True

                    # J exits: support break, +5% partial, chandelier trailing / +10% full
                    t1 = pos["entry_price"] * 1.05
                    t2 = pos["entry_price"] * 1.10
                    # Track highest high since entry for Chandelier exit
                    if high > pos.get("j_highest_high", high):
                        pos["j_highest_high"] = high
                    elif "j_highest_high" not in pos:
                        pos["j_highest_high"] = high

                    if not pos["partial_exit_done"]:
                        if not j_nifty_shields and price < entry_stop:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["shares"], day, price, "SUPPORT_BREAK"))
                            exited = True
                        elif price >= t1:
                            half = pos["shares"] // 2
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "5PCT_PARTIAL"))
                            pos["remaining_shares"] = pos["shares"] - half
                            pos["partial_exit_done"] = True
                            if price >= t2:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price, "10PCT"))
                                exited = True
                    else:
                        if not j_nifty_shields and price < entry_stop:
                            trades.append(self._make_portfolio_trade(
                                pos, pos["remaining_shares"], day, price, "SUPPORT_BREAK"))
                            exited = True
                        else:
                            # Chandelier exit: highest high since entry - 3x ATR(14)
                            atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
                            chandelier_stop = pos["j_highest_high"] - 3.0 * atr14_val
                            if price < chandelier_stop:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price, "CHANDELIER_EXIT"))
                                exited = True
                            if not exited and price >= t2:
                                trades.append(self._make_portfolio_trade(
                                    pos, pos["remaining_shares"], day, price, "10PCT"))
                                exited = True

                elif pos["strategy"] == "T":
                    # T exits: 5% SL, +6% sell 1/3, upper Keltner sell remaining
                    ema20_val = float(ind["ema20"].iloc[i])
                    atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
                    upper_keltner = ema20_val + 2 * atr14_val
                    # Tighten SL after first partial exit (stage >= 1)
                    t_sl_pct = t_tight_sl if (t_tight_sl and pos.get("partial_stage", 0) >= 1) else 0.05
                    if price <= pos["entry_price"] * (1 - t_sl_pct):
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        sl_label = f"HARD_SL_{int(t_sl_pct*100)}PCT" if t_sl_pct != 0.05 else "HARD_SL_5PCT"
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, sl_label))
                        exited = True
                    if not exited and not pos["partial_exit_done"]:
                        if three_stage_exit:
                            stage = pos.get("partial_stage", 0)
                            third = pos["shares"] // 3
                            if stage == 0 and price >= pos["entry_price"] * (1 + t_target1) and third > 0:
                                trades.append(self._make_portfolio_trade(
                                    pos, third, day, price, f"PARTIAL_{int(t_target1*100)}PCT_1of3"))
                                pos["remaining_shares"] = pos["shares"] - third
                                pos["partial_stage"] = 1
                            elif stage == 1 and price >= pos["entry_price"] * (1 + t_target2) and third > 0:
                                trades.append(self._make_portfolio_trade(
                                    pos, third, day, price, f"PARTIAL_{int(t_target2*100)}PCT_2of3"))
                                pos["remaining_shares"] = pos["shares"] - 2 * third
                                pos["partial_exit_done"] = True
                                pos["partial_stage"] = 2
                        elif price >= pos["entry_price"] * 1.05:
                            half = pos["shares"] // 2
                            if half > 0:
                                trades.append(self._make_portfolio_trade(
                                    pos, half, day, price, "PARTIAL_5PCT"))
                                pos["remaining_shares"] = pos["shares"] - half
                                pos["partial_exit_done"] = True
                    if not exited and price >= upper_keltner:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "KELTNER_UPPER_EXIT"))
                        exited = True

                elif pos["strategy"] == "R":
                    # R exits: structural SL → 3% after P1 → partials → Keltner upper
                    ema20_val = float(ind["ema20"].iloc[i])
                    atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
                    upper_keltner = ema20_val + 2 * atr14_val
                    structural_stop = pos.get("r_swing_low_stop", 0)
                    stage = pos.get("partial_stage", 0)

                    # Nifty crash shield: skip SL if Nifty fell same or more
                    r_nifty_shields = False
                    r_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if r_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - r_nifty_entry) / r_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            r_nifty_shields = True

                    # Structural SL: 1% below divergence swing low — exit all
                    if not r_nifty_shields and structural_stop > 0 and price <= structural_stop:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "STRUCTURAL_SL"))
                        exited = True

                    # Tight SL after first partial (3%) — also shielded by Nifty
                    if not exited and not r_nifty_shields and stage >= 1 and price <= pos["entry_price"] * 0.97:
                        trades.append(self._make_portfolio_trade(
                            pos, pos["remaining_shares"], day, price, "HARD_SL_3PCT"))
                        exited = True

                    # 2-stage partial exits (+8% sell 1/3, +15% sell 1/3)
                    if not exited:
                        third = pos["shares"] // 3
                        if stage == 0 and price >= pos["entry_price"] * 1.08 and third > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, third, day, price, "PARTIAL_8PCT_1of3"))
                            pos["remaining_shares"] = pos["shares"] - third
                            pos["partial_stage"] = 1
                        elif stage == 1 and price >= pos["entry_price"] * 1.15 and third > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, third, day, price, "PARTIAL_15PCT_2of3"))
                            pos["remaining_shares"] = pos["shares"] - 2 * third
                            pos["partial_exit_done"] = True
                            pos["partial_stage"] = 2

                    # Upper Keltner exit on remaining
                    if not exited and price >= upper_keltner:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "KELTNER_UPPER_EXIT"))
                        exited = True

                elif pos["strategy"] == "RW":
                    # RW exits: same as R (structural SL, 2-stage partials, Keltner upper)
                    ema20_val = float(ind["ema20"].iloc[i])
                    atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
                    upper_keltner = ema20_val + 2 * atr14_val
                    structural_stop = pos.get("r_swing_low_stop", 0)
                    stage = pos.get("partial_stage", 0)

                    # Nifty crash shield for RW
                    rw_nifty_shields = False
                    rw_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if rw_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - rw_nifty_entry) / rw_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            rw_nifty_shields = True

                    if not rw_nifty_shields and structural_stop > 0 and price <= structural_stop:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "STRUCTURAL_SL"))
                        exited = True

                    if not exited and not rw_nifty_shields and stage >= 1 and price <= pos["entry_price"] * 0.97:
                        trades.append(self._make_portfolio_trade(
                            pos, pos["remaining_shares"], day, price, "HARD_SL_3PCT"))
                        exited = True

                    if not exited:
                        third = pos["shares"] // 3
                        if stage == 0 and price >= pos["entry_price"] * 1.08 and third > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, third, day, price, "PARTIAL_8PCT_1of3"))
                            pos["remaining_shares"] = pos["shares"] - third
                            pos["partial_stage"] = 1
                        elif stage == 1 and price >= pos["entry_price"] * 1.15 and third > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, third, day, price, "PARTIAL_15PCT_2of3"))
                            pos["remaining_shares"] = pos["shares"] - 2 * third
                            pos["partial_exit_done"] = True
                            pos["partial_stage"] = 2

                    if not exited and price >= upper_keltner:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "KELTNER_UPPER_EXIT"))
                        exited = True

                elif pos["strategy"] == "MW":
                    # MW exits: 8% SL, 3% after P1, breakeven after P2, 2-stage partials, weekly Keltner upper
                    # Compute weekly upper Keltner from pre-computed weekly series
                    upper_keltner = 0.0
                    weekly_raw = ind["weekly_raw"]
                    w_dates = weekly_raw.index
                    day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                    w_before = w_dates[w_dates < day_ts]
                    if len(w_before) >= 1:
                        w_idx = len(w_before) - 1
                        wk_val = ind["mw_weekly_upper_keltner"].iloc[w_idx]
                        if not pd.isna(wk_val):
                            upper_keltner = float(wk_val)
                    stage = pos.get("partial_stage", 0)

                    # Nifty crash shield for MW
                    mw_nifty_shields = False
                    mw_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if mw_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - mw_nifty_entry) / mw_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            mw_nifty_shields = True

                    # Hard SL: 8% initial, 3% after partial 1, breakeven after partial 2
                    if stage >= 2:
                        mw_sl_price = pos["entry_price"]
                        sl_label = "BREAKEVEN_SL"
                    elif stage >= 1:
                        mw_sl_price = pos["entry_price"] * 0.97
                        sl_label = "HARD_SL_3PCT"
                    else:
                        mw_sl_price = pos["entry_price"] * 0.94
                        sl_label = "HARD_SL_6PCT"
                    if not mw_nifty_shields and price <= mw_sl_price:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, sl_label))
                        exited = True

                    # 2-stage partial exits (+8% sell 1/3, +15% sell 1/3)
                    if not exited:
                        third = pos["shares"] // 3
                        if stage == 0 and price >= pos["entry_price"] * 1.08 and third > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, third, day, price, "PARTIAL_8PCT_1of3"))
                            pos["remaining_shares"] = pos["shares"] - third
                            pos["partial_stage"] = 1
                        elif stage == 1 and price >= pos["entry_price"] * 1.15 and third > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, third, day, price, "PARTIAL_15PCT_2of3"))
                            pos["remaining_shares"] = pos["shares"] - 2 * third
                            pos["partial_exit_done"] = True
                            pos["partial_stage"] = 2

                    # Upper Keltner exit on remaining (only after first partial)
                    if not exited and pos.get("partial_stage", 0) >= 1 and price >= upper_keltner:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "KELTNER_UPPER_EXIT"))
                        exited = True

                elif pos["strategy"] == "WT":
                    # WT exits: S1 +15% sell 50%, daily close < weekly EMA50, 12% trail
                    stage = pos.get("partial_stage", 0)
                    remaining = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]

                    # Nifty crash shield for WT
                    wt_nifty_shields = False
                    wt_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if wt_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - wt_nifty_entry) / wt_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            wt_nifty_shields = True

                    # S1: +15% partial — sell 50%
                    if not exited and stage == 0 and price >= pos["entry_price"] * 1.15:
                        half = remaining // 2
                        if half > 0:
                            trades.append(self._make_portfolio_trade(
                                pos, half, day, price, "WT_PARTIAL_15PCT"))
                            pos["remaining_shares"] = remaining - half
                            pos["partial_exit_done"] = True
                            pos["partial_stage"] = 1
                            remaining = pos["remaining_shares"]

                    # Weekly EMA50 exit: daily close < weekly EMA50 (with Nifty shield)
                    if not exited and remaining > 0:
                        weekly_raw = ind["weekly_raw"]
                        w_dates = weekly_raw.index
                        day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                        w_before = w_dates[w_dates < day_ts]
                        if len(w_before) >= 1:
                            w_idx = len(w_before) - 1
                            wt_ema50_val = ind["wt_w_ema50"].iloc[w_idx]
                            if not pd.isna(wt_ema50_val) and price < float(wt_ema50_val):
                                if not wt_nifty_shields:
                                    trades.append(self._make_portfolio_trade(
                                        pos, remaining, day, price, "WT_BELOW_EMA50"))
                                    exited = True

                    # 12% trailing stop (with Nifty shield)
                    if not exited and remaining > 0:
                        highest = pos.get("wt_highest", pos["entry_price"])
                        highest = max(highest, float(highs.iloc[day_idx]) if day_idx < len(highs) else highest)
                        pos["wt_highest"] = highest
                        if price <= highest * 0.88:
                            if not wt_nifty_shields:
                                trades.append(self._make_portfolio_trade(
                                    pos, remaining, day, price, "WT_TRAIL_STOP"))
                                exited = True

                elif pos["strategy"] == "RS":
                    # RS Rotation exits — no partials, full position out
                    rs_shares = pos["shares"]

                    # Nifty crash shield: skip SL if Nifty fell same or more since entry
                    rs_nifty_shields = False
                    rs_nifty_entry = pos.get("nifty_at_entry", 0.0)
                    if rs_nifty_entry > 0 and nifty_today_close > 0:
                        nifty_pct = (nifty_today_close - rs_nifty_entry) / rs_nifty_entry
                        stock_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        if nifty_pct <= stock_pct and nifty_pct < 0:
                            rs_nifty_shields = True

                    # 1. Hard SL from entry (default 10%, configurable) — with Nifty shield
                    if not exited and not rs_nifty_shields and price <= pos["entry_price"] * rs_hard_sl:
                        sl_pct = int(round((1 - rs_hard_sl) * 100))
                        trades.append(self._make_portfolio_trade(
                            pos, rs_shares, day, price, f"RS_HARD_SL_{sl_pct}PCT"))
                        rs_cooldown[pos["symbol"]] = day_idx + rs_sl_cooldown
                        exited = True

                    # 2. Price < 30-week EMA (trend break)
                    if not exited:
                        rs_ema30w = ind.get("rs_ema30w_daily")
                        if rs_ema30w is not None and i < len(rs_ema30w) and not pd.isna(rs_ema30w.iloc[i]):
                            if price < float(rs_ema30w.iloc[i]):
                                trades.append(self._make_portfolio_trade(
                                    pos, rs_shares, day, price, "RS_TREND_BREAK"))
                                rs_cooldown[pos["symbol"]] = day_idx + 20
                                exited = True

                    # 3. 21d RS < threshold for 10 consecutive trading days (2 weeks)
                    if not exited:
                        rs_21d_series = ind.get("rs_21d")
                        if rs_21d_series is not None and i < len(rs_21d_series) and not pd.isna(rs_21d_series.iloc[i]):
                            rs_21d_val = float(rs_21d_series.iloc[i])
                            if rs_21d_val < rs_underperform_thresh:
                                pos["rs_neg_streak_days"] = pos.get("rs_neg_streak_days", 0) + 1
                            else:
                                pos["rs_neg_streak_days"] = 0
                            if pos.get("rs_neg_streak_days", 0) >= 10:
                                trades.append(self._make_portfolio_trade(
                                    pos, rs_shares, day, price, "RS_UNDERPERFORM"))
                                rs_cooldown[pos["symbol"]] = day_idx + 20
                                exited = True


                # Underwater exit — if held >= N trading days and still underwater, cut it
                # Strategy-dependent: MW 25d, RW 50d, WT 30d, others use configured value (default 10d)
                uw_days_map = {"RW": 50, "MW": 25, "WT": 30, "RS": rs_uw_days}
                uw_days = uw_days_map.get(pos["strategy"], underwater_exit_days)
                if not exited and uw_days:
                    trading_days_held = day_idx - pos.get("entry_day_idx", day_idx)
                    if trading_days_held >= uw_days and price < pos["entry_price"]:
                        sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
                        trades.append(self._make_portfolio_trade(
                            pos, sz, day, price, "UNDERWATER_EXIT"))
                        exited = True

                if not exited:
                    still_open.append(pos)

            positions = still_open
            running_pnl = sum(t["pnl"] for t in trades)

            # === 2. Collect all entry signals ===
            signals = []
            rw_signals = []  # RW collected separately, fills in only when no TR signals
            held_symbols = {p["symbol"] for p in positions}

            for ticker, ind in indicators.items():
                if ticker in held_symbols:
                    continue
                if day not in date_to_idx.get(ticker, {}):
                    continue

                i = date_to_idx[ticker][day]
                price = float(ind["closes"].iloc[i])
                open_price = float(ind["opens"].iloc[i])
                low = float(ind["lows"].iloc[i])
                high = float(ind["highs"].iloc[i])
                rsi2 = float(ind["rsi2"].iloc[i])
                rsi3 = float(ind["rsi3"].iloc[i])
                ibs = float(ind["ibs"].iloc[i])
                is_green = price > open_price
                sig_atr14 = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0

                if pd.isna(rsi2):
                    continue

                # Gap-down filter: skip if today's open < yesterday's close
                if no_gap_down and i > 0:
                    prev_close = float(ind["closes"].iloc[i - 1])
                    if open_price < prev_close:
                        continue

                # Strategy J entry: close within 0-3% above weekly support, IBS > 0.5, green
                if "J" in strategies:
                    w_support = ind["weekly_support"]
                    w_low_stop = ind["weekly_low_stop"]
                    if w_support is not None and not pd.isna(w_support.iloc[i]):
                        ws = float(w_support.iloc[i])
                        wls = float(w_low_stop.iloc[i]) if w_low_stop is not None and not pd.isna(w_low_stop.iloc[i]) else ws
                        if ws > 0:
                            close_near = ((price - ws) / ws) * 100
                            cci_val = float(ind["cci20"].iloc[i]) if not pd.isna(ind["cci20"].iloc[i]) else 0.0
                            if (close_near >= 0 and close_near <= 3.0
                                    and ibs > 0.5 and is_green
                                    and cci_val > -100):
                                j_stop_pct = (price - wls) / price * 100 if price > 0 else 99.0
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "J",
                                    "price": price,
                                    "ibs": ibs,
                                    "entry_support_j": ws,
                                    "entry_stop_j": wls,
                                    "atr14": sig_atr14,
                                    "stop_pct": j_stop_pct,
                                    "atr_norm": sig_atr14 / price if price > 0 else 99.0,
                                })

                # Strategy T entry: Price near EMA(20) (within 1%) AND was at upper Keltner in last 10 bars AND green
                if "T" in strategies:
                    ema20_val = float(ind["ema20"].iloc[i])
                    atr14_val = float(ind["atr14"].iloc[i]) if not pd.isna(ind["atr14"].iloc[i]) else 0.0
                    if atr14_val > 0:
                        near_ema20 = abs(price - ema20_val) / ema20_val <= 0.01
                        was_at_upper = False
                        for lb_j in range(max(0, i - 10), i):
                            past_high = float(ind["highs"].iloc[lb_j])
                            past_ema20 = float(ind["ema20"].iloc[lb_j])
                            past_atr14 = float(ind["atr14"].iloc[lb_j]) if not pd.isna(ind["atr14"].iloc[lb_j]) else 0.0
                            if past_high >= past_ema20 + 2 * past_atr14:
                                was_at_upper = True
                                break
                        if near_ema20 and was_at_upper and is_green and float(ind["ibs"].iloc[i]) > 0.5:
                            already = any(s["symbol"] == ticker for s in signals)
                            if not already:
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "T",
                                    "price": price,
                                    "atr14": sig_atr14,
                                    "stop_pct": 5.0,
                                    "atr_norm": sig_atr14 / price if price > 0 else 99.0,
                                })

                # Strategy R entry: Bullish RSI divergence + green + no gap-down + IBS > 0.5
                if "R" in strategies:
                    if is_green and ibs > 0.5:
                        rsi14_vals = ind["rsi14"].values
                        lows_vals = ind["lows"].values
                        divergence, swing_low_val = self._detect_bullish_divergence(
                            lows_vals, rsi14_vals, i, ind["swing_lows"],
                            rsi_threshold=35)
                        r_div_type = "regular"
                        if not divergence:
                            # Try hidden bullish divergence if price > EMA50 (uptrend)
                            ema50_val = float(ind["ema50"].iloc[i])
                            if price > ema50_val:
                                divergence, swing_low_val = self._detect_hidden_bullish_divergence(
                                    lows_vals, rsi14_vals, i, ind["swing_lows"])
                                if divergence:
                                    r_div_type = "hidden"
                        if divergence and swing_low_val is not None:
                            r_struct_stop = swing_low_val * 0.99
                            r_stop_pct = (price - r_struct_stop) / price * 100 if price > 0 else 99.0
                            r_min_stop = 2.0 if r_div_type == "hidden" else 0.0
                            if r_min_stop < r_stop_pct <= 6.0:
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "R",
                                    "price": price,
                                    "atr14": sig_atr14,
                                    "stop_pct": r_stop_pct,
                                    "atr_norm": sig_atr14 / price if price > 0 else 99.0,
                                    "r_swing_low_stop": r_struct_stop,
                                    "div_type": r_div_type,
                                })

                # Strategy MW entry: Weekly ADX crosses above 25 with DI+ > DI-
                if "MW" in strategies:
                    already_any = any(s["symbol"] == ticker for s in signals)
                    if not already_any and is_green and ibs > 0.5:
                        weekly_raw = ind["weekly_raw"]
                        w_dates = weekly_raw.index
                        day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                        w_before = w_dates[w_dates < day_ts]
                        if len(w_before) >= 2:
                            w_idx = len(w_before) - 1
                            curr_adx = ind["mw_weekly_adx_vals"][w_idx]
                            prev_adx = ind["mw_weekly_adx_vals"][w_idx - 1]
                            plus_di = ind["mw_weekly_plus_di_vals"][w_idx]
                            minus_di = ind["mw_weekly_minus_di_vals"][w_idx]
                            if (not np.isnan(curr_adx) and not np.isnan(prev_adx)
                                    and curr_adx >= mw_adx_threshold and curr_adx > prev_adx
                                    and plus_di > minus_di):
                                    signals.append({
                                        "symbol": ticker,
                                        "strategy": "MW",
                                        "price": price,
                                        "atr14": sig_atr14,
                                        "stop_pct": 5.0,
                                        "atr_norm": sig_atr14 / price if price > 0 else 99.0,
                                    })

                # Strategy WT entry: Weekly trend breakout (confirmed prev week candle)
                if "WT" in strategies:
                    already_any = any(s["symbol"] == ticker for s in signals)
                    if not already_any and is_green and ibs > 0.5:
                        weekly_raw = ind["weekly_raw"]
                        w_dates = weekly_raw.index
                        day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                        w_before = w_dates[w_dates < day_ts]
                        if len(w_before) >= 2:
                            w_idx = len(w_before) - 1
                            def _wt_val(series, idx):
                                v = series.iloc[idx] if idx < len(series) else np.nan
                                return float(v) if not pd.isna(v) else 0.0
                            wt_cl = _wt_val(ind["wt_w_close_prev"], w_idx)
                            wt_hn = _wt_val(ind["wt_w_high20_prev"], w_idx)
                            wt_e20 = _wt_val(ind["wt_w_ema20_prev"], w_idx)
                            wt_e50 = _wt_val(ind["wt_w_ema50_prev"], w_idx)
                            wt_slope = _wt_val(ind["wt_w_slope_prev"], w_idx)
                            wt_gap = _wt_val(ind["wt_w_gap_prev"], w_idx)
                            wt_gap2 = _wt_val(ind["wt_w_gap_prev2"], w_idx)
                            wt_gpct = _wt_val(ind["wt_w_gap_pct_prev"], w_idx)
                            wt_vol = _wt_val(ind["wt_w_vol_prev"], w_idx)
                            wt_vavg = _wt_val(ind["wt_w_vol_avg_prev"], w_idx)
                            wt_ret = _wt_val(ind["wt_w_return_prev"], w_idx)

                            if (wt_e20 > 0 and wt_e50 > 0 and wt_hn > 0
                                    and wt_cl >= wt_hn  # 20-week breakout
                                    and wt_e20 > wt_e50  # EMA trend
                                    and wt_gap > wt_gap2 > 0  # gap widening
                                    and wt_slope >= 0.4  # slope steep
                                    and wt_gpct >= 2.0  # gap >= 2%
                                    and wt_vavg > 0 and wt_vol >= wt_vavg * 1.3  # volume
                                    and wt_ret <= 15.0):  # no parabolic spike
                                signals.append({
                                    "symbol": ticker,
                                    "strategy": "WT",
                                    "price": price,
                                    "atr14": sig_atr14,
                                    "stop_pct": 12.0,
                                    "atr_norm": sig_atr14 / price if price > 0 else 99.0,
                                })

                # Strategy RW: collect weekly divergence signals separately (>=3pt RSI div)
                if "RW" in strategies:
                    rw_open = sum(1 for p in positions if p["strategy"] == "RW")
                    if rw_open < 4:
                        already_any = any(s["symbol"] == ticker for s in signals) or any(s["symbol"] == ticker for s in rw_signals)
                        if not already_any and is_green:
                            weekly_raw = ind["weekly_raw"]
                            w_dates = weekly_raw.index
                            day_ts = pd.Timestamp(day).tz_localize(w_dates.tz) if w_dates.tz else pd.Timestamp(day)
                            w_before = w_dates[w_dates < day_ts]
                            if len(w_before) >= 2:
                                w_idx = len(w_before) - 1
                                w_lows = ind["weekly_lows_vals"]
                                w_rsi14 = ind["weekly_rsi14_vals"]
                                w_swing_lows = ind["weekly_swing_lows"]

                                divergence, swing_low_val = self._detect_bullish_divergence(
                                    w_lows, w_rsi14, w_idx, w_swing_lows,
                                    max_lookback=13, min_sep=2,
                                    rsi_threshold=100, min_rsi_divergence=3)
                                rw_div_type = "regular"
                                if not divergence:
                                    divergence, swing_low_val = self._detect_hidden_bullish_divergence(
                                        w_lows, w_rsi14, w_idx, w_swing_lows,
                                        max_lookback=13, min_sep=2,
                                        rsi_threshold=100, min_rsi_divergence=3)
                                    if divergence:
                                        rw_div_type = "hidden"

                                if divergence and swing_low_val is not None:
                                    rw_struct_stop = swing_low_val * 0.99
                                    rw_stop_pct = (price - rw_struct_stop) / price * 100 if price > 0 else 99.0
                                    if 2.0 <= rw_stop_pct <= 8.0:
                                        rw_signals.append({
                                            "symbol": ticker,
                                            "strategy": "RW",
                                            "price": price,
                                            "atr14": sig_atr14,
                                            "stop_pct": rw_stop_pct,
                                            "atr_norm": sig_atr14 / price if price > 0 else 99.0,
                                            "r_swing_low_stop": rw_struct_stop,
                                            "div_type": rw_div_type,
                                        })

            # RW fills in only when no TR signals available that day
            if not signals and rw_signals:
                signals = rw_signals

            # RS Rotation: rank all stocks by 123d RS, fill open slots with top candidates
            if "RS" in strategies:
                # Regime filter: off when Nifty < 20w EMA, resume when 10w > 20w
                nifty_regime_ok = nifty_rs_regime_ok.get(day, True)
                rs_open = sum(1 for p in positions if p["strategy"] == "RS")
                rs_slots = MAX_POSITIONS - rs_open - sum(1 for s in signals if s["strategy"] != "RS")
                # For IBD mode, regime is optional (controlled by rs_ibd_filters)
                ibd_filters = rs_ibd_filters if rs_ibd_filters is not None else ["regime"]
                regime_gate = nifty_regime_ok if (rs_entry_mode != "rs_rating" or "regime" in ibd_filters) else True
                if rs_slots > 0 and regime_gate:
                    rs_candidates = []
                    for ticker, ind in indicators.items():
                        if ticker in held_symbols:
                            continue
                        if rs_cooldown.get(ticker, 0) > day_idx:
                            continue
                        if day not in date_to_idx.get(ticker, {}):
                            continue
                        ci = date_to_idx[ticker][day]
                        rs_123d_s = ind.get("rs_123d")
                        rs_ema30w = ind.get("rs_ema30w_daily")
                        if rs_123d_s is None or ci >= len(rs_123d_s) or pd.isna(rs_123d_s.iloc[ci]):
                            continue
                        rs_val = float(rs_123d_s.iloc[ci])
                        cp = float(ind["closes"].iloc[ci])
                        co = float(ind["opens"].iloc[ci])

                        if rs_entry_mode == "rs_rating":
                            # IBD-style: RS Rating >= 80 (top 20%), use weighted percentile
                            day_ratings = rs_rating_by_date.get(day, {})
                            rating = day_ratings.get(ticker, 0)
                            if rating < rs_ibd_min_rating or rating > rs_ibd_max_rating:
                                continue
                            # Consecutive days filter: rating must be >= min for N consecutive days
                            if rs_ibd_consec_days > 0 and day_idx >= rs_ibd_consec_days:
                                consec_ok = True
                                for lookback_d in range(1, rs_ibd_consec_days):
                                    prev_day = all_dates[day_idx - lookback_d]
                                    prev_rating = rs_rating_by_date.get(prev_day, {}).get(ticker, 0)
                                    if prev_rating < rs_ibd_min_rating:
                                        consec_ok = False
                                        break
                                if not consec_ok:
                                    continue
                            # Price > 30-week EMA (always required)
                            if rs_ema30w is None or ci >= len(rs_ema30w) or pd.isna(rs_ema30w.iloc[ci]):
                                continue
                            if cp <= float(rs_ema30w.iloc[ci]):
                                continue
                            # Optional filters controlled by rs_ibd_filters
                            if "green" in ibd_filters and cp <= co:
                                continue
                            if "ibs" in ibd_filters:
                                rs_high = float(ind["highs"].iloc[ci])
                                rs_low = float(ind["lows"].iloc[ci])
                                rs_hl = rs_high - rs_low
                                rs_ibs = (cp - rs_low) / rs_hl if rs_hl > 0 else 0.5
                                if rs_ibs <= 0.5:
                                    continue
                            if "no_gap_down" in ibd_filters and ci > 0:
                                prev_cl = float(ind["closes"].iloc[ci - 1])
                                if co < prev_cl:
                                    continue
                            # Check RS-123d > 0 (stock must outperform benchmark over 6 months)
                            actual_rs_123d = float(rs_123d_s.iloc[ci])
                            if "rs_positive" in ibd_filters and actual_rs_123d <= 0:
                                continue
                            # Smoothed RS-123d > 0 (5-day avg both ends, removes spike entries)
                            if "rs_smooth_positive" in ibd_filters:
                                rs_123d_smooth_s = ind.get("rs_123d_smooth")
                                if rs_123d_smooth_s is not None and ci < len(rs_123d_smooth_s) and not pd.isna(rs_123d_smooth_s.iloc[ci]):
                                    if float(rs_123d_smooth_s.iloc[ci]) <= 0:
                                        continue
                            rs_val = rating  # use rating for ranking
                        else:
                            # Default frozen: RS > 5% and rising
                            if rs_val <= 5.0:
                                continue
                            # RS must be rising vs 28 trading days ago
                            if ci >= 28 and not pd.isna(rs_123d_s.iloc[ci - 28]):
                                rs_28ago = float(rs_123d_s.iloc[ci - 28])
                                if rs_val <= rs_28ago:
                                    continue
                            else:
                                continue
                            # Price > 30-week EMA
                            if rs_ema30w is None or ci >= len(rs_ema30w) or pd.isna(rs_ema30w.iloc[ci]):
                                continue
                            if cp <= float(rs_ema30w.iloc[ci]):
                                continue
                            # Green candle + no gap-down + IBS > 0.5
                            if cp <= co:
                                continue
                            rs_high = float(ind["highs"].iloc[ci])
                            rs_low = float(ind["lows"].iloc[ci])
                            rs_hl = rs_high - rs_low
                            rs_ibs = (cp - rs_low) / rs_hl if rs_hl > 0 else 0.5
                            if rs_ibs <= 0.5:
                                continue
                            if ci > 0:
                                prev_cl = float(ind["closes"].iloc[ci - 1])
                                if co < prev_cl:
                                    continue
                        sig_atr = float(ind["atr14"].iloc[ci]) if not pd.isna(ind["atr14"].iloc[ci]) else 0.0
                        # --- Optional entry filters (controlled by rs_entry_filters) ---
                        rs_filters = rs_entry_filters or []
                        skip = False
                        # Filter 1: Distance from 20d high (not at blow-off top)
                        if "dist_high" in rs_filters and ci >= 20:
                            high_20d = max(float(ind["highs"].iloc[j]) for j in range(ci - 20, ci))
                            if high_20d > 0 and (high_20d - cp) / high_20d < rs_dist_high_pct:
                                skip = True  # within 3% of 20d high — too extended
                        # Filter 2: RSI(14) < 70 (not overbought)
                        if "rsi70" in rs_filters:
                            rsi_val = ind["rsi14"].iloc[ci]
                            if not pd.isna(rsi_val) and float(rsi_val) >= 70:
                                skip = True
                        # Filter 3: Volume climax (high vol + red candle in last 5 bars)
                        if "vol_climax" in rs_filters and ci >= 5:
                            for vj in range(ci - 5, ci):
                                v = float(ind["volume"].iloc[vj]) if not pd.isna(ind["volume"].iloc[vj]) else 0
                                va = float(ind["vol_avg20"].iloc[vj]) if not pd.isna(ind["vol_avg20"].iloc[vj]) else 1
                                c_vj = float(ind["closes"].iloc[vj])
                                o_vj = float(ind["opens"].iloc[vj])
                                if va > 0 and v > 2 * va and c_vj < o_vj:
                                    skip = True
                                    break
                        # Filter 4: ATR expansion (ATR > 1.5x its 20-period avg)
                        if "atr_expand" in rs_filters and ci >= 20 and sig_atr > 0:
                            atr_vals = [float(ind["atr14"].iloc[j]) for j in range(ci - 20, ci)
                                        if not pd.isna(ind["atr14"].iloc[j])]
                            if atr_vals:
                                atr_avg20 = sum(atr_vals) / len(atr_vals)
                                if atr_avg20 > 0 and sig_atr > 1.5 * atr_avg20:
                                    skip = True
                        # Filter 5: RS must be above its 100-day MA
                        if "rs_above_ma100" in rs_filters:
                            rs_ma100_s = ind.get("rs_123d_ma100")
                            if rs_ma100_s is not None and ci < len(rs_ma100_s) and not pd.isna(rs_ma100_s.iloc[ci]):
                                if rs_val < float(rs_ma100_s.iloc[ci]):
                                    skip = True
                        if skip:
                            continue
                        rs_candidates.append({
                            "symbol": ticker,
                            "strategy": "RS",
                            "price": cp,
                            "atr14": sig_atr,
                            "stop_pct": round((1 - rs_hard_sl) * 100, 1),
                            "atr_norm": sig_atr / cp if cp > 0 else 99.0,
                            "rs_123d": rs_val,
                            "ibd_rating": day_ratings.get(ticker, 0) if rs_entry_mode == "rs_rating" else 0,
                            "actual_rs": float(rs_123d_s.iloc[ci]),
                        })
                    # Rank by sector momentum (primary) then RS (secondary) when enabled
                    rank_key = "ibd_rating" if rs_ibd_rank_by_rating else "actual_rs"
                    if rank_by_sector_momentum:
                        _day_sec_mom_rs = sector_momentum_by_date.get(day, {})
                        from sector_mapping import STOCK_SECTOR_MAP as _SM_RS
                        for _rc in rs_candidates:
                            _sec = _SM_RS.get(_rc["symbol"], "OTHER")
                            _rc["sector_mom"] = _day_sec_mom_rs.get(_sec, 0.0)
                        rs_candidates.sort(key=lambda c: (-c["sector_mom"], -c[rank_key]))
                    else:
                        rs_candidates.sort(key=lambda c: -c[rank_key])
                    # Skip top N candidates if rs_ibd_skip_top > 0
                    rs_pool = rs_candidates[rs_ibd_skip_top:] if rs_ibd_skip_top > 0 else rs_candidates
                    rs_take = min(len(rs_pool), rs_slots)
                    signals.extend(rs_pool[:rs_take])

            total_signals += len(signals)

            # Filter out signals in sectors with negative momentum (skip RS — it uses its own ranking)
            if rank_by_sector_momentum:
                day_sec_mom_filt = sector_momentum_by_date.get(day, {})
                from sector_mapping import STOCK_SECTOR_MAP as _SM_FILT
                before_filt = len(signals)
                signals = [s for s in signals
                           if s.get("strategy") == "RS" or day_sec_mom_filt.get(_SM_FILT.get(s["symbol"], "OTHER"), 0.0) >= 0]
                filtered_out = before_filt - len(signals)
                missed_signals += filtered_out

            # === 3. Allocate capital (max entries_per_day) ===
            available_slots = MAX_POSITIONS - len(positions)
            # All strategies respect entries_per_day (frozen: 3/day)
            max_today = min(entries_per_day, available_slots)
            if signals and max_today > 0:
                if rank_by_risk:
                    rng = random.Random(seed)
                    strat_priority = {"R": 0, "MW": 0, "RS": 0, "WT": 1, "T": 2, "J": 3, "RW": 1}
                    if rank_by_sector_momentum:
                        # Sector momentum (descending), then lowest ATR — equal strategy priority
                        day_sec_mom = sector_momentum_by_date.get(day, {})
                        from sector_mapping import STOCK_SECTOR_MAP as _SM
                        def _sec_mom_key(s):
                            sec = _SM.get(s["symbol"], "OTHER")
                            mom = day_sec_mom.get(sec, 0.0)
                            return (strat_priority.get(s.get("strategy"), 9), -mom, s.get("atr_norm", 99.0), rng.random())
                        signals.sort(key=_sec_mom_key)
                    else:
                        # Lowest ATR first — no strategy priority
                        signals.sort(key=lambda s: (strat_priority.get(s.get("strategy"), 9), s.get("atr_norm", 99.0), rng.random()))
                else:
                    random.Random(seed).shuffle(signals)
                taken = min(len(signals), max_today)
                missed_signals += len(signals) - taken
                signals = signals[:taken]

                for sig in signals:
                    shares = int(PER_STOCK // sig["price"])
                    if shares > 0:
                        capital_used = round(sig["price"] * shares, 2)
                        deployed = sum(p["entry_price"] * p["shares"] for p in positions) + capital_used
                        free_capital = TOTAL_CAPITAL + running_pnl - deployed
                        if free_capital < 0:
                            missed_signals += 1
                            continue  # Not enough capital
                        # Enforce max 4 RW/TW positions each
                        if sig["strategy"] == "RW":
                            w_open = sum(1 for p in positions if p["strategy"] == "RW")
                            if w_open >= 4:
                                missed_signals += 1
                                continue
                        pos = {
                            "symbol": sig["symbol"],
                            "strategy": sig["strategy"],
                            "entry_date": day,
                            "entry_day_idx": day_idx,
                            "entry_price": sig["price"],
                            "shares": shares,
                            "remaining_shares": shares,
                            "partial_exit_done": False,
                            "capital_used": capital_used,
                            "capital_deployed": round(deployed, 0),
                            "capital_available": round(TOTAL_CAPITAL + running_pnl - deployed, 0),
                        }
                        pos["nifty_at_entry"] = nifty_close_by_date.get(day, 0.0)
                        if sig["strategy"] == "J":
                            pos["entry_support_j"] = sig["entry_support_j"]
                            pos["entry_stop_j"] = sig["entry_stop_j"]
                        elif sig["strategy"] in ("R", "RW"):
                            pos["r_swing_low_stop"] = sig["r_swing_low_stop"]
                            pos["div_type"] = sig.get("div_type", "regular")
                        elif sig["strategy"] == "WT":
                            pos["wt_highest"] = sig["price"]
                        elif sig["strategy"] == "RS":
                            pos["rs_neg_streak_days"] = 0
                            pos["rs_123d_at_entry"] = sig.get("rs_123d", 0)
                        # MW uses same hard SL as T — no extra pos fields needed
                        positions.append(pos)
            elif signals and available_slots <= 0:
                missed_signals += len(signals)

            if len(positions) > max_positions_used:
                max_positions_used = len(positions)
            positions_over_time.append({
                "date": day.isoformat(),
                "positions": len(positions)
            })
            # Liquid fund income on idle capital (empty slots)
            empty_slots = MAX_POSITIONS - len(positions)
            if empty_slots > 0:
                liquid_fund_income += empty_slots * PER_STOCK * LIQUID_FUND_DAILY_RATE

        # --- Close remaining positions at end ---
        last_day = all_dates[-1]
        for pos in positions:
            ticker = pos["symbol"]
            if last_day in date_to_idx.get(ticker, {}):
                i = date_to_idx[ticker][last_day]
                price = float(indicators[ticker]["closes"].iloc[i])
            else:
                # Fallback: use the most recent available close price
                ticker_dates = date_to_idx.get(ticker, {})
                recent = [d for d in ticker_dates if d <= last_day]
                if recent:
                    i = ticker_dates[max(recent)]
                    price = float(indicators[ticker]["closes"].iloc[i])
                else:
                    price = pos["entry_price"]
            sz = pos["remaining_shares"] if pos["partial_exit_done"] else pos["shares"]
            trades.append(self._make_portfolio_trade(
                pos, sz, last_day, price, "BACKTEST_END"))

        # Sort trades by exit_date (chronological order of realization)
        trades.sort(key=lambda t: (t["exit_date"], t["entry_date"]))

        # --- Build summary ---
        # Prorate return based on max capital actually deployed
        effective_capital = max_positions_used * PER_STOCK if max_positions_used > 0 else TOTAL_CAPITAL
        summary = self._calculate_summary(trades, effective_capital)

        strat_labels = {"J": "J(Weekly Support)", "T": "T(Keltner)", "R": "R(RSI Divergence)", "MW": "MW(Weekly ADX)"}
        strat_label = " + ".join(strat_labels.get(s, s) for s in strategies)

        return {
            "strategy": "Portfolio",
            "strategies": strategies,
            "strategies_label": strat_label,
            "exit_target": None,
            "start_date": all_dates[0].isoformat(),
            "end_date": all_dates[-1].isoformat(),
            "trading_days": len(all_dates),
            "capital": TOTAL_CAPITAL,
            "capital_lakhs": capital_lakhs,
            "effective_capital_lakhs": round(effective_capital / 100000, 1),
            "max_positions": MAX_POSITIONS,
            "trades": trades,
            "summary": summary,
            "total_signals": total_signals,
            "missed_signals": missed_signals,
            "max_positions_used": max_positions_used,
            "universe": universe,
            "liquid_fund_income": round(liquid_fund_income, 2),
        }

    def _make_portfolio_trade(self, pos, shares, exit_date, exit_price, reason):
        """Make a trade dict for portfolio backtest (includes symbol & strategy)."""
        trade = self._make_trade(
            pos["entry_date"], pos["entry_price"], shares,
            exit_date, exit_price, reason)
        trade["symbol"] = pos["symbol"]
        trade["strategy"] = pos["strategy"]
        trade["capital_used"] = round(shares * pos["entry_price"], 2)
        trade["capital_available"] = pos.get("capital_available", 0)
        if "div_type" in pos:
            trade["div_type"] = pos["div_type"]
        return trade

    def run_all_stocks(self, period_days, capital=100000, progress_callback=None, universe=50, end_date=None):
        """Run all strategy variants across stocks (Nifty 50 or 100)."""
        from data.momentum_engine import NIFTY_50_TICKERS

        if universe <= 50:
            tickers = NIFTY_50_TICKERS
        elif universe <= 100:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS
        elif universe <= 200:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS
        else:
            tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS + NIFTY_500_BEYOND200_TICKERS

        results = []
        total = len(tickers)

        for idx, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(idx + 1, total, ticker)

            # Fetch data ONCE per stock
            nse_symbol = f"{ticker}.NS"
            _end = end_date or datetime.now()
            daily_start = _end - timedelta(days=period_days + 500)
            try:
                daily = yf.Ticker(nse_symbol).history(start=daily_start, end=_end)
            except Exception:
                daily = pd.DataFrame()

            row = {"symbol": ticker, "results": {}}

            for strat, exit_tgt, label in BATCH_VARIANTS:
                result = self.run(ticker, period_days, strategy=strat,
                                  capital=capital, exit_target=exit_tgt,
                                  _daily_data=daily, end_date=_end)
                if "error" not in result:
                    row["results"][label] = {
                        "total_pnl": result["summary"]["total_pnl"],
                        "return_pct": result["summary"]["total_return_pct"],
                        "win_rate": result["summary"]["win_rate"],
                        "trades": result["summary"]["total_trades"],
                        "profit_factor": result["summary"]["profit_factor"],
                    }
                else:
                    row["results"][label] = None

            results.append(row)

        columns = [v[2] for v in BATCH_VARIANTS]
        return {"columns": columns, "rows": results}

