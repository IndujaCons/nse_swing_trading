"""Price alert evaluation engine.

Pure functions — no Flask imports. Called by the alert checker
background thread in ui/app.py every 5 minutes during market hours.

Conditions:
  price   — latest 5-min bar close from yfinance
  1hrRSI  — RSI(14) on 1-hour bars
  50DEMA  — 50-day exponential moving average

Operators:
  gt  — current value > threshold
  lt  — current value < threshold
  eq  — within 0.5% of threshold
"""

import json
import os
import uuid
import warnings
from datetime import datetime, timezone, timedelta

import yfinance as yf

warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))


# ── Ticker helpers ─────────────────────────────────────────────────────────────

def normalize_ticker(ticker):
    """Strip spaces and uppercase — NSE/US tickers never contain spaces."""
    return ticker.upper().replace(" ", "").strip()


def _yf_symbol(ticker, exchange):
    """Return yfinance symbol: append .NS for NSE, bare for US.
    Index symbols (^NSEI, ^NSEBANK, etc.) are exchange-agnostic — no suffix.
    """
    t = normalize_ticker(ticker)
    if t.startswith("^"):
        return t
    return t + ".NS" if exchange == "NSE" else t


def validate_ticker(ticker, exchange):
    """Return (valid: bool, error_msg: str|None).
    Does a lightweight yfinance lookup — fails fast if symbol unknown.
    """
    sym = _yf_symbol(ticker, exchange)
    try:
        df = yf.Ticker(sym).history(period="5d", interval="1d")
        if df.empty:
            return False, f"Ticker '{sym}' not found on yfinance — check the symbol"
        return True, None
    except Exception as e:
        return False, f"Could not validate '{sym}': {e}"


# ── Data fetchers ──────────────────────────────────────────────────────────────

def fetch_current_price(ticker, exchange):
    """Latest price from 5-min bars (today's session). Returns float or None."""
    try:
        sym = _yf_symbol(ticker, exchange)
        df = yf.Ticker(sym).history(period="1d", interval="5m")
        if df.empty:
            return None
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None


def fetch_1h_rsi(ticker, exchange, period=14):
    """RSI(14) on 1-hour bars over past 5 days. Returns float or None."""
    try:
        sym = _yf_symbol(ticker, exchange)
        df = yf.Ticker(sym).history(period="5d", interval="1h")
        closes = df["Close"].dropna()
        if len(closes) < period + 1:
            return None
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1])
    except Exception:
        return None


def fetch_50dema(ticker, exchange):
    """50-day EMA on daily closes. Returns float or None."""
    try:
        sym = _yf_symbol(ticker, exchange)
        df = yf.Ticker(sym).history(period="120d")
        closes = df["Close"].dropna()
        if len(closes) < 50:
            return None
        ema = closes.ewm(span=50, adjust=False).mean()
        return float(ema.iloc[-1])
    except Exception:
        return None


def fetch_all_values(ticker, exchange):
    """Fetch price, 1hrRSI, and 50DEMA concurrently. Returns dict with all three."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_price = ex.submit(fetch_current_price, ticker, exchange)
        f_rsi   = ex.submit(fetch_1h_rsi,        ticker, exchange)
        f_ema   = ex.submit(fetch_50dema,         ticker, exchange)
        price = f_price.result()
        rsi   = f_rsi.result()
        ema   = f_ema.result()
    return {
        "price":  round(price, 2) if price is not None else None,
        "rsi_1h": round(rsi,   2) if rsi   is not None else None,
        "ema50":  round(ema,   2) if ema   is not None else None,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_alert(alert):
    """Evaluate one alert. Returns (triggered: bool, current_value: float|None)."""
    cond      = alert["condition"]
    op        = alert["operator"]
    threshold = float(alert["value"])

    if cond == "price":
        val = fetch_current_price(alert["ticker"], alert["exchange"])
    elif cond == "1hrRSI":
        val = fetch_1h_rsi(alert["ticker"], alert["exchange"])
    elif cond == "50DEMA":
        val = fetch_50dema(alert["ticker"], alert["exchange"])
    else:
        return False, None

    if val is None:
        return False, None

    if op == "gt":
        triggered = val > threshold
    elif op == "lt":
        triggered = val < threshold
    elif op == "eq":
        triggered = abs(val - threshold) / max(abs(threshold), 1) < 0.005
    else:
        return False, None

    return triggered, round(val, 2)


# ── Store helpers ──────────────────────────────────────────────────────────────

def load_alerts(alerts_path):
    """Load alerts.json; return dict with 'alerts' list. Creates file if absent."""
    os.makedirs(os.path.dirname(alerts_path), exist_ok=True)
    if not os.path.exists(alerts_path):
        _save_raw(alerts_path, {"alerts": []})
    try:
        with open(alerts_path) as f:
            data = json.load(f)
        if "alerts" not in data:
            data["alerts"] = []
        return data
    except Exception:
        return {"alerts": []}


def _save_raw(alerts_path, data):
    """Atomic write."""
    tmp = alerts_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, alerts_path)


def save_alerts(alerts_path, data):
    _save_raw(alerts_path, data)


def create_alert(alerts_path, ticker, exchange, condition, operator, value):
    """Add a new alert. Returns the created alert dict."""
    data = load_alerts(alerts_path)
    alert = {
        "id":           uuid.uuid4().hex[:12],
        "ticker":       normalize_ticker(ticker),
        "exchange":     exchange,
        "condition":    condition,
        "operator":     operator,
        "value":        float(value),
        "active":       True,
        "triggered_at": None,
        "created_at":   datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S"),
        "last_value":   None,
        "last_checked": None,
    }
    data["alerts"].append(alert)
    save_alerts(alerts_path, data)
    return alert


def delete_alert(alerts_path, alert_id):
    """Remove alert by id. Returns True if found."""
    data = load_alerts(alerts_path)
    before = len(data["alerts"])
    data["alerts"] = [a for a in data["alerts"] if a["id"] != alert_id]
    save_alerts(alerts_path, data)
    return len(data["alerts"]) < before


def rearm_alert(alerts_path, alert_id):
    """Reset a triggered alert back to active. Returns True if found."""
    data = load_alerts(alerts_path)
    for a in data["alerts"]:
        if a["id"] == alert_id:
            a["active"]       = True
            a["triggered_at"] = None
            a["last_value"]   = None
            save_alerts(alerts_path, data)
            return True
    return False


# ── Batch check ────────────────────────────────────────────────────────────────

def check_all_alerts(alerts_path, on_triggered=None):
    """Evaluate all active alerts. Mark triggered ones inactive.

    on_triggered(alert, current_value) is called for each newly fired alert
    (e.g., to send email or push to a buffer).

    Returns list of newly triggered alert dicts.
    """
    data = load_alerts(alerts_path)
    now_str = datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S")
    newly_triggered = []
    changed = False

    for alert in data["alerts"]:
        if not alert.get("active"):
            continue
        try:
            triggered, val = evaluate_alert(alert)
        except Exception:
            continue

        alert["last_value"]   = val
        alert["last_checked"] = now_str
        changed = True

        if triggered:
            alert["active"]       = False
            alert["triggered_at"] = now_str
            newly_triggered.append(dict(alert))
            if on_triggered:
                try:
                    on_triggered(alert, val)
                except Exception:
                    pass

    if changed:
        save_alerts(alerts_path, data)

    return newly_triggered
