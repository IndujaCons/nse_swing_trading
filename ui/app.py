"""
Flask Application for RS Dashboard
"""

import os
import sys
import json
from datetime import datetime
from functools import wraps
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from flask import Flask, render_template, jsonify, request, session

from config.settings import (
    FLASK_HOST, FLASK_PORT, DEBUG_MODE,
    PAPER_TRADING_ONLY, load_config, save_config, get_cache_ttl,
    get_kite_users, DATA_STORE_PATH,
    WATCHLIST_FILE,
)
from data.screener_engine import ScreenerEngine
from data.live_signals_engine import LiveSignalsEngine
from data.momentum_engine import MomentumEngine
from data.momentum_backtest import MomentumBacktester
from broker.kite_broker import KiteBroker

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global instances
screener_engine = ScreenerEngine()
momentum_engine = MomentumEngine()

# Shared scanner (no user_id — uses shared cache, no per-user positions)
live_signals_scanner = LiveSignalsEngine()

# Per-user brokers and position engines
kite_users = get_kite_users()
brokers = {}       # {user_id: KiteBroker}
user_engines = {}  # {user_id: LiveSignalsEngine}

from data.goldm_engine import GoldmEngine
goldm_engines = {}  # {user_id: GoldmEngine}

from data.etf_engine import ETFEngine, ETF_YF_OVERRIDES
etf_engine = ETFEngine()

for _u in kite_users:
    brokers[_u["id"]] = KiteBroker(
        user_id=_u["id"], name=_u["name"],
        api_key=_u["api_key"], api_secret=_u["api_secret"])
    user_engines[_u["id"]] = LiveSignalsEngine(user_id=_u["id"])
    goldm_engines[_u["id"]] = GoldmEngine(user_id=_u["id"])

# Track which user_id initiated the most recent login popup (for OAuth callback)
_pending_login_user_id = None


# Background refresh state
refresh_lock = threading.Lock()
refresh_in_progress = False
refresh_progress = {"current": 0, "total": 0, "ticker": ""}

# Live Signals refresh state
live_signals_refresh_lock = threading.Lock()
live_signals_refresh_in_progress = False
live_signals_refresh_progress = {"current": 0, "total": 0, "ticker": ""}

# Momentum refresh state
momentum_refresh_lock = threading.Lock()
momentum_refresh_in_progress = False
momentum_refresh_progress = {"current": 0, "total": 0, "ticker": ""}

# Batch backtest state
batch_backtest_lock = threading.Lock()
batch_backtest_in_progress = False
batch_backtest_progress = {"current": 0, "total": 0, "ticker": ""}

# Portfolio backtest state
portfolio_backtest_lock = threading.Lock()
portfolio_backtest_in_progress = False
portfolio_backtest_progress = {"current": 0, "total": 0, "ticker": ""}



def get_refresh_progress():
    """Get current refresh progress."""
    global refresh_progress
    return refresh_progress.copy()


def update_progress(current, total, ticker):
    """Update progress callback for screener."""
    global refresh_progress
    refresh_progress = {"current": current, "total": total, "ticker": ticker}


def get_live_signals_refresh_progress():
    """Get current Live Signals refresh progress."""
    global live_signals_refresh_progress
    return live_signals_refresh_progress.copy()


def update_live_signals_progress(current, total, ticker):
    """Update progress callback for Live Signals scanner."""
    global live_signals_refresh_progress
    live_signals_refresh_progress = {"current": current, "total": total, "ticker": ticker}


def get_momentum_refresh_progress():
    """Get current momentum refresh progress."""
    global momentum_refresh_progress
    return momentum_refresh_progress.copy()


def update_momentum_progress(current, total, ticker):
    """Update progress callback for momentum scanner."""
    global momentum_refresh_progress
    momentum_refresh_progress = {"current": current, "total": total, "ticker": ticker}


# ============ Page Routes ============

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


# ============ API Routes ============

@app.route("/api/screener/data", methods=["GET"])
def get_screener_data():
    """Get categorized stock data."""
    try:
        data = screener_engine.run_screener(force_refresh=False)
        cache_age = screener_engine.get_cache_age_minutes()

        return jsonify({
            "success": True,
            "data": data,
            "cache_age_minutes": round(cache_age, 1) if cache_age else None,
            "cache_ttl_minutes": get_cache_ttl()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/screener/refresh", methods=["POST"])
def refresh_screener():
    """Force refresh screener data."""
    global refresh_in_progress

    with refresh_lock:
        if refresh_in_progress:
            return jsonify({
                "success": False,
                "error": "Refresh already in progress"
            }), 409

        refresh_in_progress = True

    try:
        data = screener_engine.run_screener(
            force_refresh=True,
            progress_callback=update_progress
        )

        return jsonify({
            "success": True,
            "data": data,
            "message": "Data refreshed successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        with refresh_lock:
            refresh_in_progress = False


@app.route("/api/screener/progress", methods=["GET"])
def get_progress():
    """Get current refresh progress."""
    global refresh_in_progress
    progress = get_refresh_progress()
    progress["in_progress"] = refresh_in_progress
    return jsonify(progress)


# ============ Live Signals API Routes ============

@app.route("/api/live-signals/data", methods=["GET"])
def get_live_signals_data():
    """Get cached J+T scan results."""
    try:
        data = live_signals_scanner.scan_entry_signals(force_refresh=False)
        cache_age = live_signals_scanner.get_cache_age_minutes()

        return jsonify({
            "success": True,
            "data": data,
            "cache_age_minutes": round(cache_age, 1) if cache_age else None,
            "cache_ttl_minutes": get_cache_ttl()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def _get_connected_broker():
    """Return first connected Zerodha broker, or None."""
    for uid, broker in brokers.items():
        if broker.access_token:
            return broker
    return None


# NSE instrument token cache — refreshed once per day
_nse_inst_cache = {"tokens": {}, "date": None}


def _get_nse_tokens(broker):
    """Return {tradingsymbol: instrument_token} for NSE (cached daily)."""
    from datetime import date as _date
    today = _date.today().isoformat()
    if _nse_inst_cache["date"] == today and _nse_inst_cache["tokens"]:
        return _nse_inst_cache["tokens"]
    try:
        instruments = broker._kite.instruments("NSE")
        _nse_inst_cache["tokens"] = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
        _nse_inst_cache["date"] = today
    except Exception:
        pass
    return _nse_inst_cache["tokens"]


def _kite_daily_closes(broker, symbol, from_date, to_date):
    """Fetch daily close Series from Kite for a NSE symbol. Returns pd.Series or None."""
    import pandas as pd
    tokens = _get_nse_tokens(broker)
    token = tokens.get(symbol)
    if not token:
        return None
    try:
        candles = broker._kite.historical_data(token, from_date, to_date, "day")
    except Exception:
        return None
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    return df.set_index("date")["close"].rename(symbol)


def _fetch_zerodha_ltp(tickers):
    """Fetch real-time LTP from first connected Zerodha broker for given tickers."""
    for uid, broker in brokers.items():
        if broker.access_token:
            instruments = [f"NSE:{t}" for t in tickers]
            # Add index instruments
            instruments.append("NSE:NIFTY 50")
            instruments.append("NSE:NIFTY 200")
            # Kite API supports up to 1000 instruments per call
            ltp_data = broker.get_ltp(instruments)
            if ltp_data:
                ltp_map = {}
                for key, val in ltp_data.items():
                    # key is like "NSE:RELIANCE", val has {"last_price": ...}
                    symbol = key.split(":")[-1] if ":" in key else key
                    ltp = val.get("last_price", 0)
                    if ltp > 0:
                        if symbol == "NIFTY 50":
                            ltp_map["^NSEI"] = ltp
                        elif symbol == "NIFTY 200":
                            ltp_map["^CNX200"] = ltp
                        else:
                            ltp_map[symbol] = ltp
                return ltp_map
    return None


@app.route("/api/live-signals/refresh", methods=["POST"])
def refresh_live_signals():
    """Stale-while-revalidate refresh (Tier-1 perf C).

    Returns the current cached scan **immediately** (sub-100ms) and kicks
    off a background thread to do the fresh scan. The frontend polls
    /api/live-signals/progress until in_progress=False, then GETs
    /api/live-signals/data for the fresh result. This way the click never
    blocks the UI on a 30-60s yfinance fetch.
    """
    global live_signals_refresh_in_progress

    # Read whatever's in cache right now — return it as the immediate response
    # so the page can re-render instantly while the fresh scan runs.
    cached_data = None
    cache_age = None
    try:
        cached_data = live_signals_scanner.scan_entry_signals(force_refresh=False)
        cache_age = live_signals_scanner.get_cache_age_minutes()
    except Exception:
        pass

    with live_signals_refresh_lock:
        already_running = live_signals_refresh_in_progress
        if not already_running:
            live_signals_refresh_in_progress = True

    if not already_running:
        # Spawn the actual scan in a daemon thread; the request returns
        # without waiting for it. Frontend polls /api/live-signals/progress.
        def _bg_refresh():
            global live_signals_refresh_in_progress
            try:
                from data.live_signals_engine import get_scan_tickers
                from config.settings import load_config
                universe = load_config().get("live_signals_universe", 50)
                scan_tickers = get_scan_tickers(universe)
                ltp_map = _fetch_zerodha_ltp(scan_tickers)

                data = live_signals_scanner.scan_entry_signals(
                    force_refresh=True,
                    progress_callback=update_live_signals_progress,
                    ltp_map=ltp_map
                )
                _persist_mom20_ranks_to_users(data)
                print(f"[refresh-bg] scan done — "
                      f"{len(data.get('mom20_signals', []))} mom20 signals")
            except Exception as e:
                print(f"[refresh-bg] scan failed: {e}")
            finally:
                with live_signals_refresh_lock:
                    live_signals_refresh_in_progress = False

        threading.Thread(target=_bg_refresh, daemon=True,
                         name="live-signals-refresh").start()

    return jsonify({
        "success":           True,
        "data":              cached_data,
        "cache_age_minutes": round(cache_age, 1) if cache_age else None,
        "cached":            True,
        "in_progress":       True,
        "already_running":   already_running,
        "message":           ("Refresh in progress — showing cached data"
                              if already_running
                              else "Refresh started in background"),
    })


@app.route("/api/live-signals/scan-date", methods=["POST"])
def scan_live_signals_date():
    """Scan J+T entry signals for a specific historical date."""
    global live_signals_refresh_in_progress

    data = request.get_json() or {}
    scan_date = data.get("scan_date")

    if not scan_date:
        return jsonify({"success": False, "error": "scan_date required"}), 400

    # Validate format
    try:
        dt = datetime.strptime(scan_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"success": False, "error": "Invalid date format (YYYY-MM-DD)"}), 400

    # Validate range
    if dt.date() > datetime.now().date():
        return jsonify({"success": False, "error": "Cannot scan future dates"}), 400
    if dt.year < 2015:
        return jsonify({"success": False, "error": "Date must be 2015 or later"}), 400

    with live_signals_refresh_lock:
        if live_signals_refresh_in_progress:
            return jsonify({"success": False, "error": "Scan already in progress"}), 409
        live_signals_refresh_in_progress = True

    try:
        result = live_signals_scanner.scan_entry_signals(
            force_refresh=True,
            progress_callback=update_live_signals_progress,
            scan_date=scan_date,
        )
        return jsonify({
            "success": True,
            "data": result,
            "message": f"Signals scanned for {scan_date}",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        with live_signals_refresh_lock:
            live_signals_refresh_in_progress = False


@app.route("/api/live-signals/progress", methods=["GET"])
def get_live_signals_progress():
    """Poll scan progress."""
    global live_signals_refresh_in_progress
    progress = get_live_signals_refresh_progress()
    progress["in_progress"] = live_signals_refresh_in_progress
    return jsonify(progress)


@app.route("/api/live-signals/margins", methods=["GET"])
def get_live_margins():
    """Get available margin/funds from connected Zerodha accounts."""
    try:
        result = {}
        for uid, broker in brokers.items():
            if broker.is_connected():
                result[uid] = broker.get_margins()
        return jsonify({"success": True, "margins": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============ Momentum API Routes ============

@app.route("/api/momentum/data", methods=["GET"])
def get_momentum_data():
    """Get cached momentum scanner results."""
    try:
        data = momentum_engine.run_screener(force_refresh=False)
        cache_age = momentum_engine.get_cache_age_minutes()

        return jsonify({
            "success": True,
            "data": data,
            "cache_age_minutes": round(cache_age, 1) if cache_age else None,
            "cache_ttl_minutes": get_cache_ttl()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/momentum/refresh", methods=["POST"])
def refresh_momentum():
    """Force refresh momentum scanner data."""
    global momentum_refresh_in_progress

    with momentum_refresh_lock:
        if momentum_refresh_in_progress:
            return jsonify({
                "success": False,
                "error": "Refresh already in progress"
            }), 409

        momentum_refresh_in_progress = True

    try:
        data = momentum_engine.run_screener(
            force_refresh=True,
            progress_callback=update_momentum_progress
        )

        return jsonify({
            "success": True,
            "data": data,
            "message": "Momentum data refreshed successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        with momentum_refresh_lock:
            momentum_refresh_in_progress = False


@app.route("/api/momentum/progress", methods=["GET"])
def get_momentum_progress():
    """Get current momentum refresh progress."""
    global momentum_refresh_in_progress
    progress = get_momentum_refresh_progress()
    progress["in_progress"] = momentum_refresh_in_progress
    return jsonify(progress)


# ============ Momentum Backtest Routes ============

def _parse_date_range(data):
    """Parse from_date/to_date from request, return (period_days, end_date)."""
    from_date = data.get("from_date")
    to_date = data.get("to_date")
    if from_date and to_date:
        from datetime import datetime as dt
        fd = dt.strptime(from_date, "%Y-%m-%d")
        td = dt.strptime(to_date, "%Y-%m-%d")
        period_days = (td - fd).days
        if period_days < 1:
            period_days = 30
        return period_days, td
    # Fallback to period_days
    period_days = int(data.get("period_days", 365))
    return period_days, None


@app.route("/api/momentum/backtest", methods=["POST"])
def run_momentum_backtest():
    """Run momentum backtest for a single symbol."""
    data = request.get_json()
    symbol = data.get("symbol")
    period_days, end_date = _parse_date_range(data)
    strategy = data.get("strategy", "B")
    exit_ema = data.get("exit_ema")

    if not symbol:
        return jsonify({"success": False, "error": "Symbol required"}), 400

    if strategy not in ("J", "T", "R", "MW"):
        strategy = "J"

    # exit_ema can be "5","8","10","20" (EMA) or "pct5" (% target)
    valid_targets = ("5", "8", "10", "20", "pct5", "5pct10pct")
    if exit_ema is not None:
        exit_ema = str(exit_ema) if str(exit_ema) in valid_targets else None

    try:
        backtester = MomentumBacktester()
        result = backtester.run(symbol, period_days, strategy=strategy,
                                exit_target=exit_ema, end_date=end_date)

        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/momentum/explain-trade", methods=["POST"])
def explain_trade():
    """Explain a single trade: setup, entry, exit progression, P&L."""
    data = request.get_json()
    symbol = data.get("symbol")
    strategy = data.get("strategy")
    entry_date = data.get("entry_date")

    if not symbol or not strategy or not entry_date:
        return jsonify({"success": False, "error": "symbol, strategy, entry_date required"}), 400

    if strategy not in ("J", "T", "R", "MW", "RS", "RS63", "MOM15"):
        return jsonify({"success": False, "error": "strategy must be J, T, R, MW, RS, RS63, or MOM15"}), 400

    try:
        backtester = MomentumBacktester()
        result = backtester.explain_trade(symbol, strategy, entry_date)
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def update_batch_progress(current, total, ticker):
    """Update progress callback for batch backtest."""
    global batch_backtest_progress
    batch_backtest_progress = {"current": current, "total": total, "ticker": ticker}


@app.route("/api/momentum/backtest-all", methods=["POST"])
def run_batch_backtest():
    """Run batch backtest across all Nifty 50 stocks and all strategy variants."""
    global batch_backtest_in_progress

    with batch_backtest_lock:
        if batch_backtest_in_progress:
            return jsonify({
                "success": False,
                "error": "Batch backtest already in progress"
            }), 409
        batch_backtest_in_progress = True

    data = request.get_json() or {}
    period_days, end_date = _parse_date_range(data)
    universe = int(data.get("universe", 50))
    if universe not in (50, 100, 200, 500):
        universe = 50

    try:
        backtester = MomentumBacktester()
        result = backtester.run_all_stocks(
            period_days,
            progress_callback=update_batch_progress,
            universe=universe,
            end_date=end_date
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        with batch_backtest_lock:
            batch_backtest_in_progress = False


@app.route("/api/momentum/backtest-all/progress", methods=["GET"])
def get_batch_backtest_progress():
    """Get current batch backtest progress."""
    global batch_backtest_in_progress, batch_backtest_progress
    progress = batch_backtest_progress.copy()
    progress["in_progress"] = batch_backtest_in_progress
    return jsonify(progress)


# ============ Portfolio Backtest Routes ============

def update_portfolio_backtest_progress(current, total, ticker):
    """Update progress callback for portfolio backtest."""
    global portfolio_backtest_progress
    portfolio_backtest_progress = {"current": current, "total": total, "ticker": ticker}


def _wrap_rebalance_result(result, capital_lakhs, label):
    """Wrap rebalance backtest result into portfolio backtest format for UI."""
    trades = result.get("trades", [])
    capital = capital_lakhs * 100000
    # Normalize trade fields
    for t in trades:
        if hasattr(t.get("entry_date"), "strftime"):
            t["entry_date"] = t["entry_date"].strftime("%Y-%m-%d")
        if hasattr(t.get("exit_date"), "strftime"):
            t["exit_date"] = t["exit_date"].strftime("%Y-%m-%d")
        if "holding_days" not in t and "hold_days" in t:
            t["holding_days"] = t["hold_days"]
        if "holding_days" not in t:
            t["holding_days"] = 0
        if "symbol" not in t and "ticker" in t:
            t["symbol"] = t["ticker"]
        if "symbol" not in t:
            t["symbol"] = t.get("ticker", "???")
        if "pnl_pct" not in t:
            cost = t.get("entry_price", 0) * t.get("shares", 0)
            t["pnl_pct"] = round(t.get("pnl", 0) / cost * 100, 1) if cost > 0 else 0

    # Build summary
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    pf = round(min(gross_wins / gross_losses if gross_losses > 0 else 999.99, 999.99), 2)
    avg_hold = round(sum(t.get("holding_days", 0) for t in trades) / len(trades), 1) if trades else 0

    dates = sorted(set(
        [t["entry_date"] for t in trades] + [t["exit_date"] for t in trades]
    ))

    return {
        "strategy": "Portfolio",
        "strategies": [label.split()[0].upper()],
        "strategies_label": label,
        "start_date": dates[0] if dates else "",
        "end_date": dates[-1] if dates else "",
        "trading_days": len(dates),
        "capital": capital,
        "capital_lakhs": capital_lakhs,
        "effective_capital_lakhs": capital_lakhs,
        "max_positions": 20,
        "trades": trades,
        "summary": {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / capital * 100, 2) if capital else 0,
            "avg_win": round(gross_wins / len(wins), 2) if wins else 0,
            "avg_loss": round(-gross_losses / len(losses), 2) if losses else 0,
            "largest_win": round(max((t["pnl"] for t in trades), default=0), 2),
            "largest_loss": round(min((t["pnl"] for t in trades), default=0), 2),
            "profit_factor": pf,
            "avg_holding_days": avg_hold,
        },
        "total_signals": len(trades),
        "missed_signals": 0,
        "max_positions_used": 20,
        "universe": 200,
        "liquid_fund_income": 0,
    }


@app.route("/api/momentum/backtest-portfolio", methods=["POST"])
def run_portfolio_backtest():
    """Run portfolio-level backtest (10L capital, J+T strategies)."""
    global portfolio_backtest_in_progress

    with portfolio_backtest_lock:
        if portfolio_backtest_in_progress:
            return jsonify({
                "success": False,
                "error": "Portfolio backtest already in progress"
            }), 409
        portfolio_backtest_in_progress = True

    data = request.get_json() or {}
    period_days, end_date = _parse_date_range(data)
    universe = int(data.get("universe", 50))
    if universe not in (50, 100, 200, 500):
        universe = 50
    capital_lakhs = int(data.get("capital_lakhs", 10))
    if capital_lakhs not in (10, 20, 50, 80, 100):
        capital_lakhs = 10
    per_stock = int(data.get("per_stock", 50000))
    if per_stock not in (50000, 100000, 200000, 500000):
        per_stock = 50000
    strategies = data.get("strategies", ["R", "MW"])
    valid_strats = {"J", "T", "R", "MW", "RS", "MOM15", "MOM20", "ALPHA20"}
    strategies = [s for s in strategies if s in valid_strats]
    if not strategies:
        strategies = ["R", "MW"]
    entries_per_day = int(data.get("entries_per_day", 3))
    if entries_per_day not in (1, 2, 3):
        entries_per_day = 3
    three_stage_exit = bool(data.get("three_stage_exit", True))

    try:
        backtester = MomentumBacktester()

        # Mom15 uses a separate rebalancing backtest (frozen: 4m, top15, buffer 10/30, beta 1.0)
        if "MOM15" in strategies:
            result = backtester.run_momentum30_backtest(
                period_days=period_days,
                capital_lakhs=capital_lakhs,
                rebalance_months=4,
                top_n=15,
                buffer_in=10,
                buffer_out=30,
                beta_cap=1.0,
                pit_universe=True,
                end_date=end_date,
            )
            if "error" in result:
                return jsonify({"success": False, "error": result["error"]})
            data = _wrap_rebalance_result(result, capital_lakhs, "Mom15 (4-Monthly Rebal)")
            return jsonify({"success": True, "data": data})

        # Mom20 uses a separate rebalancing backtest
        if "MOM20" in strategies:
            result = backtester.run_momentum30_backtest(
                period_days=period_days,
                capital_lakhs=capital_lakhs,
                rebalance_months=3,
                top_n=20,
                buffer_in=15,
                buffer_out=40,
                end_date=end_date,
            )
            if "error" in result:
                return jsonify({"success": False, "error": result["error"]})
            data = _wrap_rebalance_result(result, capital_lakhs, "Mom20 (Quarterly Rebal)")
            return jsonify({"success": True, "data": data})

        # Alpha20 uses a separate rebalancing backtest
        if "ALPHA20" in strategies:
            result = backtester.run_alpha20_backtest(
                period_days=period_days,
                capital_lakhs=capital_lakhs,
                rebalance_months=3,
                top_n=20,
                buffer_in=20,
                buffer_out=40,
                beta_cap=1.2,
                end_date=end_date,
            )
            if "error" in result:
                return jsonify({"success": False, "error": result["error"]})
            data = _wrap_rebalance_result(result, capital_lakhs, "Alpha20 (CAPM Alpha)")
            return jsonify({"success": True, "data": data})

        # Frozen IBD RS params when RS strategy is selected
        rs_kwargs = {}
        if "RS" in strategies:
            rs_kwargs = {
                "rs_entry_filters": ["dist_high"],
                "rs_regime_mode": "early_10w_2",
                "rs_hard_sl": 0.92,
                "rs_uw_days": 0,
                "rs_entry_mode": "rs_rating",
                "rs_ibd_filters": ["regime", "rs_positive"],
                "rs_sl_cooldown": 40,
                "rs_ibd_skip_top": 2,
                "rs_ibd_consec_days": 5,
            }
        result = backtester.run_portfolio_backtest(
            period_days,
            universe=universe,
            capital_lakhs=capital_lakhs,
            per_stock=per_stock,
            strategies=strategies,
            entries_per_day=entries_per_day,
            progress_callback=update_portfolio_backtest_progress,
            end_date=end_date,
            three_stage_exit=three_stage_exit,
            no_gap_down=True,
            rank_by_risk=True,
            underwater_exit_days=10,
            t_tight_sl=0.03,
            rank_by_sector_momentum=True,
            **rs_kwargs
        )
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        with portfolio_backtest_lock:
            portfolio_backtest_in_progress = False


@app.route("/api/momentum/backtest-portfolio/progress", methods=["GET"])
def get_portfolio_backtest_progress():
    """Get current portfolio backtest progress."""
    global portfolio_backtest_in_progress, portfolio_backtest_progress
    progress = portfolio_backtest_progress.copy()
    progress["in_progress"] = portfolio_backtest_in_progress
    return jsonify(progress)


# ============ Watchlist Routes ============

_watchlist_price_cache = {"data": {}, "timestamp": 0}
_watchlist_news_cache = {"data": {}, "timestamp": 0}


def _load_watchlist_data():
    """Load full watchlist data (groups) from file."""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                data = json.load(f)
            # Migrate old format (plain list) to new format (groups)
            if isinstance(data, list):
                data = {"groups": [{"name": "General", "tickers": data}], "active": "General"}
                _save_watchlist_data(data)
            return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"groups": [{"name": "General", "tickers": []}], "active": "General"}


def _save_watchlist_data(data):
    """Save full watchlist data to file."""
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _get_active_group(data):
    """Get the active group's ticker list."""
    for g in data["groups"]:
        if g["name"] == data.get("active"):
            return g
    return data["groups"][0] if data["groups"] else {"name": "General", "tickers": []}


ETF_TICKERS = [
    # Indian Broad Market
    "JUNIORBEES", "MID150BEES", "HDFCSMALL", "SETFNN50",
    # Indian Sectoral
    "BANKBEES", "PSUBNKBEES", "ITBEES", "HEALTHIETF", "AUTOBEES",
    "METALIETF", "CONSUMBEES", "INFRABEES", "OILIETF", "MOREALTY",
    # Indian Factor / Thematic
    "CPSEETF", "SETFMOMET", "QUAL30IETF", "KOTAKLOWV", "ALPL30IETF",
    "MODEFENCE", "MOMOMENTUM", "MONIFTY500",
    # Indian Commodities
    "GOLDBEES", "SILVERBEES",
    # India-listed international trackers (NSE-listed, .NS suffix)
    "MON100",  # MASPTOP50 removed
    # Liquid / Cash
    "LIQUIDBEES",
]

# International ETFs (US-listed, no .NS suffix)
INTL_ETF_TICKERS = ["FRDM", "EMXC", "AVDV", "ILF", "XLE", "GDX", "XME", "VGK",
                    "SOXX", "BOTZ", "EWY", "XLK", "XLP", "TLT", "XLV", "ITA", "EWJ"]
# LSE-listed UCITS ETFs (estate-tax friendly; .L suffix)
LSE_ETF_TICKERS = ["GDGB", "VEUR", "ISF", "EMXU", "LTAM", "IUES", "COPX", "WSML", "CJPN", "CSPX"]

def _yf_symbol(ticker):
    """Return yfinance symbol — skip .NS for international ETFs and index tickers."""
    if ticker in INTL_ETF_TICKERS or ticker.startswith("^"):
        return ticker
    if ticker in LSE_ETF_TICKERS:
        return ticker + ".L"
    return f"{ticker}.NS"

@app.route("/api/watchlist/tickers", methods=["GET"])
def watchlist_tickers():
    """Return full Nifty 500 + ETF list for autocomplete."""
    from nifty500_tickers import NIFTY_500_TICKERS
    combined = sorted(set(NIFTY_500_TICKERS + ETF_TICKERS + INTL_ETF_TICKERS))
    return jsonify({"tickers": combined})


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    """Return active group's watchlist with price data + group metadata."""
    import time
    data = _load_watchlist_data()
    group = _get_active_group(data)
    watchlist = group["tickers"]
    groups_meta = [{"name": g["name"], "count": len(g["tickers"])} for g in data["groups"]]

    if not watchlist:
        return jsonify({"success": True, "stocks": [], "groups": groups_meta,
                        "active_group": data["active"]})

    now = time.time()
    cache = _watchlist_price_cache
    cached_tickers = set(cache["data"].keys())
    if cache["data"] and now - cache["timestamp"] < 300 and set(watchlist).issubset(cached_tickers):
        stocks = [cache["data"][t] for t in watchlist if t in cache["data"]]
        return jsonify({"success": True, "stocks": stocks, "groups": groups_meta,
                        "active_group": data["active"]})

    try:
        import yfinance as yf
        symbols = [_yf_symbol(t) for t in watchlist]
        df = yf.download(symbols, period="5d", progress=False, group_by="ticker", threads=True)

        stocks = []
        for ticker in watchlist:
            sym = _yf_symbol(ticker)
            try:
                if len(watchlist) == 1:
                    closes = df["Close"].dropna()
                else:
                    closes = df[sym]["Close"].dropna()
                if len(closes) >= 2:
                    ltp = round(float(closes.iloc[-1]), 2)
                    prev = round(float(closes.iloc[-2]), 2)
                    change = round(ltp - prev, 2)
                    change_pct = round(change / prev * 100, 2) if prev else 0
                else:
                    ltp = round(float(closes.iloc[-1]), 2) if len(closes) else 0
                    prev = ltp
                    change = 0
                    change_pct = 0
                entry = {"ticker": ticker, "ltp": ltp, "prev_close": prev,
                         "change": change, "change_pct": change_pct}
                stocks.append(entry)
                cache["data"][ticker] = entry
            except Exception:
                stocks.append({"ticker": ticker, "ltp": 0, "prev_close": 0,
                               "change": 0, "change_pct": 0})
        cache["timestamp"] = now
        return jsonify({"success": True, "stocks": stocks, "groups": groups_meta,
                        "active_group": data["active"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/watchlist/add", methods=["POST"])
def watchlist_add():
    """Add a stock to active group."""
    from nifty500_tickers import NIFTY_500_TICKERS
    ticker = (request.json or {}).get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"success": False, "error": "Ticker required"}), 400
    if ticker not in NIFTY_500_TICKERS and ticker not in ETF_TICKERS and ticker not in INTL_ETF_TICKERS:
        return jsonify({"success": False, "error": f"{ticker} not in Nifty 500 / ETF list"}), 400

    data = _load_watchlist_data()
    group = _get_active_group(data)
    if ticker in group["tickers"]:
        return jsonify({"success": True})
    if len(group["tickers"]) >= 20:
        return jsonify({"success": False, "error": "Max 20 stocks per group"}), 400

    group["tickers"].append(ticker)
    _save_watchlist_data(data)
    return jsonify({"success": True})


@app.route("/api/watchlist/remove", methods=["POST"])
def watchlist_remove():
    """Remove a stock from active group."""
    ticker = (request.json or {}).get("ticker", "").upper().strip()
    data = _load_watchlist_data()
    group = _get_active_group(data)
    if ticker in group["tickers"]:
        group["tickers"].remove(ticker)
        _save_watchlist_data(data)
        _watchlist_price_cache["data"].pop(ticker, None)
    return jsonify({"success": True})


@app.route("/api/watchlist/groups", methods=["POST"])
def watchlist_groups():
    """Manage watchlist groups: switch, add, rename, delete."""
    body = request.json or {}
    action = body.get("action", "")
    data = _load_watchlist_data()

    if action == "switch":
        group_name = body.get("name", "")
        if any(g["name"] == group_name for g in data["groups"]):
            data["active"] = group_name
            _save_watchlist_data(data)
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "Group not found"}), 404

    elif action == "add":
        group_name = body.get("name", "").strip()
        if not group_name:
            return jsonify({"success": False, "error": "Name required"}), 400
        if len(data["groups"]) >= 10:
            return jsonify({"success": False, "error": "Max 10 groups"}), 400
        if any(g["name"] == group_name for g in data["groups"]):
            return jsonify({"success": False, "error": "Group already exists"}), 400
        data["groups"].append({"name": group_name, "tickers": []})
        data["active"] = group_name
        _save_watchlist_data(data)
        return jsonify({"success": True})

    elif action == "rename":
        old_name = body.get("old_name", "")
        new_name = body.get("new_name", "").strip()
        if not new_name:
            return jsonify({"success": False, "error": "Name required"}), 400
        if any(g["name"] == new_name for g in data["groups"]):
            return jsonify({"success": False, "error": "Name already exists"}), 400
        for g in data["groups"]:
            if g["name"] == old_name:
                g["name"] = new_name
                if data["active"] == old_name:
                    data["active"] = new_name
                _save_watchlist_data(data)
                return jsonify({"success": True})
        return jsonify({"success": False, "error": "Group not found"}), 404

    elif action == "delete":
        group_name = body.get("name", "")
        if len(data["groups"]) <= 1:
            return jsonify({"success": False, "error": "Cannot delete last group"}), 400
        data["groups"] = [g for g in data["groups"] if g["name"] != group_name]
        if data["active"] == group_name:
            data["active"] = data["groups"][0]["name"]
        _save_watchlist_data(data)
        return jsonify({"success": True})

    return jsonify({"success": False, "error": "Unknown action"}), 400


@app.route("/api/watchlist/news", methods=["GET"])
def watchlist_news():
    """Fetch news for watchlist stocks from Google News RSS (last 1 month)."""
    import time
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    data = _load_watchlist_data()
    group = _get_active_group(data)
    watchlist = group["tickers"]
    if not watchlist:
        return jsonify({"success": True, "news": {}})

    now = time.time()
    cache = _watchlist_news_cache
    # Check if all watchlist tickers are already cached
    missing = [t for t in watchlist if t not in cache.get("data", {})]
    if not missing and cache["data"] and now - cache["timestamp"] < 1800:
        news = {t: cache["data"].get(t, []) for t in watchlist}
        return jsonify({"success": True, "news": news})

    # Fetch only missing tickers (or all if cache expired)
    tickers_to_fetch = missing if (now - cache.get("timestamp", 0) < 1800) else watchlist[:15]
    news = dict(cache.get("data", {}))
    for ticker in tickers_to_fetch[:15]:
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            cutoff = datetime.now() - __import__('datetime').timedelta(days=90)
            items = []
            for item in root.findall(".//item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date_raw = item.findtext("pubDate", "")
                display_date = ""
                if pub_date_raw:
                    try:
                        dt = parsedate_to_datetime(pub_date_raw)
                        if dt.replace(tzinfo=None) < cutoff:
                            continue
                        display_date = dt.strftime("%d %b %Y")
                    except Exception:
                        # If date parsing fails, still include the item
                        parts = pub_date_raw.split(",")
                        display_date = parts[1].strip()[:11].strip() if len(parts) > 1 else ""
                items.append({"title": title, "link": link, "date": display_date})
                if len(items) >= 10:
                    break
            news[ticker] = items
        except Exception:
            news[ticker] = []

    cache["data"] = news
    cache["timestamp"] = now
    # Only return news for current watchlist
    filtered = {t: news.get(t, []) for t in watchlist}
    return jsonify({"success": True, "news": filtered})


@app.route("/api/watchlist/chart", methods=["GET"])
def watchlist_chart():
    """Return daily close prices + RS for a ticker."""
    import pandas as pd
    ticker = request.args.get("ticker", "").upper().strip()
    period = request.args.get("period", "1y").strip()
    if not ticker:
        return jsonify({"success": False, "error": "Ticker required"}), 400
    if period not in ("1mo", "3mo", "6mo", "1y"):
        period = "1y"

    # RS lookback: 1M→21d, 3M→63d, 6M→123d, 1Y→123d
    rs_lookback_map = {"1mo": 21, "3mo": 63, "6mo": 123, "1y": 123}
    rs_lookback = rs_lookback_map[period]
    # Need extra history for RS calculation
    extra_map = {"1mo": "4mo", "3mo": "1y", "6mo": "2y", "1y": "2y"}
    fetch_period = extra_map[period]

    try:
        import yfinance as yf
        from datetime import date as _date, timedelta as _timedelta
        is_lse = ticker.endswith(".L")
        base_sym = ticker.replace(".NS", "").replace(".BO", "").replace(".L", "")

        # Period → fetch days (display + RS lookback buffer)
        period_days_extra = {"1mo": 100, "3mo": 160, "6mo": 250, "1y": 430}
        fetch_days = period_days_extra[period]
        from_date = _date.today() - _timedelta(days=fetch_days)
        to_date = _date.today()

        stock_closes = None
        stock_volumes = None
        broker = _get_connected_broker()
        if broker and not is_lse:
            stock_closes = _kite_daily_closes(broker, base_sym, from_date, to_date)

        # Fallback to yfinance
        if stock_closes is None or len(stock_closes) < 5:
            extra_map = {"1mo": "4mo", "3mo": "1y", "6mo": "2y", "1y": "2y"}
            fetch_period = extra_map[period]
            if is_lse:
                yf_ticker = ticker          # use full symbol e.g. CSKR.L as-is
            elif base_sym in ETF_YF_OVERRIDES:
                yf_ticker = ETF_YF_OVERRIDES[base_sym] + ".NS"
            else:
                yf_ticker = _yf_symbol(base_sym)
            stock_df = yf.download(yf_ticker, period=fetch_period, progress=False)
            if stock_df.empty:
                return jsonify({"success": False, "error": "No data found"}), 404
            stock_closes = stock_df["Close"].squeeze().dropna()
            if stock_closes.index.tz is not None:
                stock_closes.index = stock_closes.index.tz_localize(None)
            if "Volume" in stock_df.columns:
                stock_volumes = stock_df["Volume"].squeeze()
                if stock_volumes.index.tz is not None:
                    stock_volumes.index = stock_volumes.index.tz_localize(None)

        # Benchmark: try Kite first (NIFTY 200), fall back to yfinance
        bench_closes = None
        if broker:
            bench_closes = _kite_daily_closes(broker, "NIFTY 200", from_date, to_date)
        if bench_closes is None or len(bench_closes) < 5:
            extra_map = {"1mo": "4mo", "3mo": "1y", "6mo": "2y", "1y": "2y"}
            bench_df = yf.download("^CNX200", period=extra_map[period], progress=False)
            if not bench_df.empty:
                bench_closes = bench_df["Close"].squeeze().dropna()
                if bench_closes.index.tz is not None:
                    bench_closes.index = bench_closes.index.tz_localize(None)

        if stock_closes is None or len(stock_closes) < 5:
            return jsonify({"success": False, "error": "No data found"}), 404

        # Compute RS (raw) and RS smoothed (5-day avg on both ends)
        if bench_closes is not None and len(bench_closes) >= 5:
            # Raw RS
            bench_ret = (bench_closes / bench_closes.shift(rs_lookback) - 1) * 100
            stock_ret = (stock_closes / stock_closes.shift(rs_lookback) - 1) * 100
            rs_series = stock_ret - bench_ret.reindex(stock_closes.index, method="ffill")
            # Smoothed RS: 5-day avg on both ends
            sc5 = stock_closes.rolling(5, center=True, min_periods=3).mean()
            bc_aligned = bench_closes.reindex(stock_closes.index, method="ffill")
            bc5 = bc_aligned.rolling(5, center=True, min_periods=3).mean()
            s_ret_smooth = (sc5 / sc5.shift(rs_lookback) - 1) * 100
            b_ret_smooth = (bc5 / bc5.shift(rs_lookback) - 1) * 100
            rs_smooth_series = s_ret_smooth - b_ret_smooth
        else:
            rs_series = pd.Series(dtype=float)
            rs_smooth_series = pd.Series(dtype=float)

        # DMA 63, 126, 200 — compute on full history before trimming
        dma50_full = stock_closes.rolling(63, min_periods=63).mean()
        dma100_full = stock_closes.rolling(126, min_periods=126).mean()
        dma200_full = stock_closes.rolling(200, min_periods=200).mean()

        # Trim to requested period
        period_days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        cutoff_date = stock_closes.index[-1] - pd.Timedelta(days=period_days_map[period])
        mask = stock_closes.index >= cutoff_date
        stock_closes = stock_closes[mask]

        dates = [d.strftime("%Y-%m-%d") for d in stock_closes.index]
        prices = [round(float(p), 2) for p in stock_closes.values]

        rs_data = []
        rs_smooth_data = []
        for d in stock_closes.index:
            val = rs_series.get(d)
            rs_data.append(round(float(val), 2) if val is not None and not pd.isna(val) else None)
            sval = rs_smooth_series.get(d)
            rs_smooth_data.append(round(float(sval), 2) if sval is not None and not pd.isna(sval) else None)

        # Nifty 200 normalized to same scale as stock price for overlay
        bench_prices = []
        if bench_closes is not None and len(bench_closes) >= 5:
            bc = bench_closes.reindex(stock_closes.index, method="ffill").dropna()
            if len(bc) >= 2:
                # Normalize: scale bench to start at same price as stock
                scale = float(stock_closes.iloc[0]) / float(bc.iloc[0]) if float(bc.iloc[0]) != 0 else 1
                for d in stock_closes.index:
                    bv = bc.get(d)
                    if bv is not None and not pd.isna(bv):
                        bench_prices.append(round(float(bv) * scale, 2))
                    else:
                        bench_prices.append(None)

        # RSI(14)
        _delta = stock_closes.diff()
        _gain  = _delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        _loss  = (-_delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi_full = 100 - (100 / (1 + _gain / _loss.replace(0, 1e-10)))

        dma50_data = []
        dma100_data = []
        dma200_data = []
        rsi_data = []
        for d in stock_closes.index:
            v50 = dma50_full.get(d)
            dma50_data.append(round(float(v50), 2) if v50 is not None and not pd.isna(v50) else None)
            v100 = dma100_full.get(d)
            dma100_data.append(round(float(v100), 2) if v100 is not None and not pd.isna(v100) else None)
            v200 = dma200_full.get(d)
            dma200_data.append(round(float(v200), 2) if v200 is not None and not pd.isna(v200) else None)
            vr = rsi_full.get(d)
            rsi_data.append(round(float(vr), 1) if vr is not None and not pd.isna(vr) else None)

        volumes_data = []
        if stock_volumes is not None:
            for d in stock_closes.index:
                v = stock_volumes.get(d)
                volumes_data.append(int(v) if v is not None and not pd.isna(v) and v > 0 else None)

        rs_label_map = {"1mo": "RS-21d", "3mo": "RS-63d", "6mo": "RS-123d", "1y": "RS-123d"}
        is_intl = ticker in INTL_ETF_TICKERS
        return jsonify({
            "success": True, "ticker": ticker, "dates": dates,
            "prices": prices, "rs": rs_data, "rs_smooth": rs_smooth_data,
            "rs_label": rs_label_map[period],
            "bench": bench_prices, "bench_label": "Nifty 200",
            "currency": "$" if is_intl else "₹",
            "dma50": dma50_data, "dma100": dma100_data, "dma200": dma200_data,
            "volumes": volumes_data, "rsi": rsi_data,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============ Zerodha Login Routes ============

@app.route("/api/users", methods=["GET"])
def get_users():
    """Return all configured users with connection status."""
    result = []
    for uid, broker in brokers.items():
        status = broker.get_connection_status()
        result.append(status)
    return jsonify({"success": True, "users": result})


@app.route("/api/login/url", methods=["GET"])
def get_login_url():
    """Get Zerodha login URL for a specific user."""
    global _pending_login_user_id
    user_id = request.args.get("user_id")
    broker = brokers.get(user_id) if user_id else None

    if not broker:
        # Fallback to first broker
        if brokers:
            user_id = list(brokers.keys())[0]
            broker = brokers[user_id]
        else:
            return jsonify({"success": False, "error": "No users configured"})

    url = broker.get_login_url()
    if url:
        _pending_login_user_id = user_id
        return jsonify({"success": True, "url": url})
    else:
        return jsonify({
            "success": False,
            "error": f"API key not configured for {broker.config_name}."
        })


@app.route("/api/login/token", methods=["POST"])
def exchange_token():
    """Exchange request token for access token."""
    data = request.get_json()
    request_token = data.get("request_token")
    user_id = data.get("user_id")

    if not request_token:
        return jsonify({
            "success": False,
            "error": "Request token required"
        }), 400

    broker = brokers.get(user_id) if user_id else None
    if not broker:
        return jsonify({"success": False, "error": "Valid user_id required"}), 400

    success = broker.exchange_token(request_token)
    if success:
        return jsonify({
            "success": True,
            "message": f"Connected {broker.config_name} to Zerodha"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to exchange token. Check API credentials."
        }), 401


@app.route("/api/login/callback")
def login_callback():
    """Handle Zerodha OAuth callback."""
    global _pending_login_user_id
    request_token = request.args.get("request_token")
    user_id = _pending_login_user_id
    broker = brokers.get(user_id) if user_id else None

    if request_token and broker:
        success = broker.exchange_token(request_token)
        if success:
            return """
            <html>
            <body>
            <script>
                window.opener.postMessage({type: 'ZERODHA_LOGIN_SUCCESS'}, '*');
                window.close();
            </script>
            <p>Login successful! You can close this window.</p>
            </body>
            </html>
            """
    return """
    <html>
    <body>
    <script>
        window.opener.postMessage({type: 'ZERODHA_LOGIN_FAILED'}, '*');
        window.close();
    </script>
    <p>Login failed. Please try again.</p>
    </body>
    </html>
    """


@app.route("/api/connection/status", methods=["GET"])
def connection_status():
    """Get all users' connection statuses."""
    statuses = []
    for uid, broker in brokers.items():
        statuses.append(broker.get_connection_status())
    return jsonify({"success": True, "users": statuses})


@app.route("/api/connection/disconnect", methods=["POST"])
def disconnect():
    """Disconnect a specific user from Zerodha."""
    data = request.get_json() or {}
    user_id = data.get("user_id")
    broker = brokers.get(user_id) if user_id else None
    if not broker:
        return jsonify({"success": False, "error": "Valid user_id required"}), 400
    broker.disconnect()
    return jsonify({"success": True, "message": f"Disconnected {broker.config_name}"})


# ============ Settings Routes ============

@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Get current settings."""
    config = load_config()
    return jsonify({
        "success": True,
        "settings": {
            "rs_period": config.get("rs_period", 63),
            "ema_period": config.get("ema_period", 63),
            "cache_ttl_minutes": config.get("cache_ttl_minutes", 15),
            "paper_trading_only": PAPER_TRADING_ONLY,
            "support_lookback_days": config.get("support_lookback_days", 120),
            "support_proximity_pct": config.get("support_proximity_pct", 3.0),
            "momentum_rsi2_threshold": config.get("momentum_rsi2_threshold", 75),
            "live_signals_universe": config.get("live_signals_universe", 50),
        }
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update settings and save to config file."""
    data = request.get_json()

    config = load_config()
    settings_changed = False

    # Update allowed settings
    if "rs_period" in data:
        val = int(data["rs_period"])
        if val in [21, 63, 126, 252] and val != config.get("rs_period"):
            config["rs_period"] = val
            settings_changed = True

    if "ema_period" in data:
        val = int(data["ema_period"])
        if val in [21, 63, 126, 252] and val != config.get("ema_period"):
            config["ema_period"] = val
            settings_changed = True

    if "cache_ttl_minutes" in data:
        val = int(data["cache_ttl_minutes"])
        if 1 <= val <= 60:
            config["cache_ttl_minutes"] = val

    # Live Signals settings
    live_signals_changed = False

    if "live_signals_universe" in data:
        val = int(data["live_signals_universe"])
        if val in [50, 100, 150, 200] and val != config.get("live_signals_universe"):
            config["live_signals_universe"] = val
            live_signals_changed = True

    # Momentum settings
    momentum_changed = False

    if "momentum_rsi2_threshold" in data:
        val = int(data["momentum_rsi2_threshold"])
        if val in [60, 65, 70, 75, 80, 85, 90] and val != config.get("momentum_rsi2_threshold"):
            config["momentum_rsi2_threshold"] = val
            momentum_changed = True

    save_config(config)

    # Clear cache if settings changed so next refresh uses new values
    if settings_changed:
        screener_engine.clear_cache()

    if live_signals_changed:
        pass  # Cache will be refreshed on next scan

    if momentum_changed:
        momentum_engine.clear_cache()

    return jsonify({
        "success": True,
        "message": "Settings saved" + (" - cache cleared, refreshing..." if settings_changed or live_signals_changed or momentum_changed else ""),
        "settings": config,
        "needs_refresh": settings_changed,
        "needs_live_signals_refresh": live_signals_changed,
        "needs_momentum_refresh": momentum_changed
    })


# ============ Paper Trading Routes ============

@app.route("/api/trade/paper", methods=["POST"])
def execute_paper_trade():
    """Execute a paper trade."""
    data = request.get_json()

    ticker = data.get("ticker")
    action = data.get("action", "BUY")
    price = data.get("price")
    quantity = data.get("quantity", 1)

    if not ticker or not price:
        return jsonify({
            "success": False,
            "error": "Ticker and price required"
        }), 400

    try:
        _first_broker = next(iter(brokers.values())) if brokers else None
        if not _first_broker:
            return jsonify({"success": False, "error": "No brokers configured"}), 400
        trade = _first_broker.execute_paper_trade(ticker, action, price, quantity)
        return jsonify({
            "success": True,
            "trade": trade
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/trade/paper/history", methods=["GET"])
def get_paper_trades():
    """Get paper trade history."""
    limit = request.args.get("limit", 50, type=int)
    _first_broker = next(iter(brokers.values())) if brokers else None
    trades = _first_broker.get_paper_trades(limit) if _first_broker else []
    return jsonify({
        "success": True,
        "trades": trades
    })


@app.route("/api/trade/paper/portfolio", methods=["GET"])
def get_paper_portfolio():
    """Get paper portfolio."""
    _first_broker = next(iter(brokers.values())) if brokers else None
    portfolio = _first_broker.get_paper_portfolio() if _first_broker else {"positions": [], "total_trades": 0}
    return jsonify({
        "success": True,
        "portfolio": portfolio
    })


@app.route("/api/trade/paper/clear", methods=["POST"])
def clear_paper_trades():
    """Clear paper trades."""
    _first_broker = next(iter(brokers.values())) if brokers else None
    if _first_broker:
        _first_broker.clear_paper_trades()
    return jsonify({
        "success": True,
        "message": "Paper trades cleared"
    })


# ============ ETF Core — Momentum Rotation ============

@app.route("/etf-core")
def etf_core():
    """ETF Core strategy page."""
    return render_template("etf_core.html")


@app.route("/api/etf/scan", methods=["POST"])
def etf_scan():
    """ETF scan — Z-score ranking from score_live() (same engine the
    Telegram scheduler uses). Universe is Z-score's UNIVERSE (47 symbols);
    UCITS-only names are listed unranked at the bottom. Position/watchlist
    state and iNAV are layered on top for display."""
    try:
        from data.etf_core_zscore_backtest import score_live, UNIVERSE as Z_UNIVERSE
        from data.etf_engine import ETF_UNIVERSE as ETF_META

        # Authoritative ranking — fail loudly instead of falling back silently
        try:
            zscore_list = score_live()
        except Exception as e:
            return jsonify({"success": False,
                            "error": f"score_live() raised: {e}"})
        if not zscore_list:
            return jsonify({"success": False,
                            "error": "score_live() returned no data — yfinance fetch likely timed out. Retry in a moment."})

        # Display metadata: prefer etf_engine's name+category for the 40
        # overlapping symbols; fall back to Z-score UNIVERSE's name (no category)
        # for the 7 globals/UCITS that are Z-only.
        meta = {}
        for sym, name, _yf in Z_UNIVERSE:
            meta[sym] = {"name": name, "category": ""}
        for e in ETF_META:
            meta[e["symbol"]] = {"name": e["name"], "category": e["category"]}

        # State (positions + manual watchlist) from etf_engine — no scan needed
        try:
            state = etf_engine._load_state()
        except Exception:
            state = {}
        positions = state.get("positions", {}) or {}
        wl_state  = state.get("watchlist", {}) or {}

        # iNAV (Kite-only, optional)
        broker = _get_connected_broker()
        inav_map = {}
        if broker and broker._kite:
            try:
                inav_map = etf_engine._fetch_inav(broker._kite) or {}
            except Exception:
                inav_map = {}

        # VIX (display only)
        try:
            vix_val = etf_engine._fetch_vix()
        except Exception:
            vix_val = None

        # Build watchlist: scored items in Z-rank order, then UCITS unscored
        scored_syms = {z["symbol"] for z in zscore_list}
        watchlist = []

        for z in zscore_list:
            sym = z["symbol"]
            m = meta.get(sym, {})
            in_pos = sym in positions
            in_wl  = sym in wl_state
            rank = z["rank"]

            # Signal logic on Z-rank only (same buffer as Mom20 spirit: top-5
            # entry, top-9 hold-buffer, beyond = drop)
            if in_pos:
                sig = "HOLD" if rank <= 9 else "EXIT_X3"
            elif in_wl:
                sig = "WATCHLIST"
            elif rank <= 5:
                sig = "ENTRY"
            elif rank <= 9:
                sig = "BUFFER"
            else:
                sig = "RANKED"

            inav = inav_map.get(sym)
            inav_prem = (round((z["price"] - inav) / inav * 100, 2)
                         if inav and z["price"] else None)

            watchlist.append({
                "symbol":      sym,
                "name":        m.get("name", ""),
                "category":    m.get("category", ""),
                "rank":        rank,
                "price":       z["price"],
                "mom_score":   z["score"],
                "ret_12m":     z["ret_12m"],
                "ret_3m":      z["ret_3m"],
                "beta":        z["beta"],
                "inav":        round(inav, 4) if inav else None,
                "inav_prem":   inav_prem,
                "in_position": in_pos,
                "in_watchlist": in_wl,
                "signal":      sig,
                # legacy RS63 columns — null since Z-score doesn't use them
                "sma63":       None,
                "rs63":        None,
                "trend_ok":    None,
                "no_data":     False,
            })

        for sym, name, _yf in Z_UNIVERSE:
            if sym in scored_syms:
                continue
            m = meta.get(sym, {"name": name, "category": ""})
            watchlist.append({
                "symbol":      sym,
                "name":        m["name"],
                "category":    m["category"],
                "rank":        None,
                "price":       None,
                "mom_score":   None,
                "ret_12m":     None,
                "ret_3m":      None,
                "beta":        None,
                "inav":        None,
                "inav_prem":   None,
                "in_position": sym in positions,
                "in_watchlist": sym in wl_state,
                "signal":      "UCITS",
                "sma63":       None,
                "rs63":        None,
                "trend_ok":    None,
                "no_data":     True,
            })

        return jsonify({
            "success":         True,
            "watchlist":       watchlist,
            "scan_time":       datetime.now().strftime("%Y-%m-%d %H:%M"),
            "vix":             vix_val,
            "positions":       list(positions.values()),
            "entry_signals":   [],
            "exit_signals":    [],
            "reentry_signals": [],
            "ucits_prices":    {},
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/etf/enter", methods=["POST"])
def etf_enter():
    """Enter an ETF position."""
    data = request.get_json() or {}
    symbol = data.get("symbol", "").upper().strip()
    price = data.get("entry_price") or data.get("price")
    qty = data.get("qty", 0)
    reason = data.get("reason", "ENTRY")

    if not symbol or not price:
        return jsonify({"success": False, "error": "symbol and price required"})

    slot, err = etf_engine.enter_position(symbol, float(price), float(qty), reason)
    if err:
        return jsonify({"success": False, "error": err})
    return jsonify({"success": True, "slot": slot})


@app.route("/api/etf/exit", methods=["POST"])
def etf_exit():
    """Exit an ETF position."""
    data = request.get_json() or {}
    symbol = data.get("symbol", "").upper().strip()
    exit_price = data.get("exit_price") or data.get("price")
    reason = data.get("reason", "MANUAL")

    if not symbol or not exit_price:
        return jsonify({"success": False, "error": "symbol and exit_price required"})

    trade, err = etf_engine.exit_position(symbol, float(exit_price), reason)
    if err:
        return jsonify({"success": False, "error": err})
    return jsonify({"success": True, "trade": trade})


@app.route("/api/etf/modify", methods=["POST"])
def etf_modify():
    """Modify entry_price and/or qty of an active position."""
    data = request.get_json() or {}
    symbol = data.get("symbol", "").upper().strip()
    price = data.get("entry_price")
    qty = data.get("qty")
    if not symbol:
        return jsonify({"success": False, "error": "symbol required"})
    if price is None and qty is None:
        return jsonify({"success": False, "error": "entry_price or qty required"})
    slot, err = etf_engine.modify_position(symbol, float(price) if price is not None else None,
                                            float(qty) if qty is not None else None)
    if err:
        return jsonify({"success": False, "error": err})
    return jsonify({"success": True, "slot": slot})


@app.route("/api/etf/positions", methods=["GET"])
def etf_positions():
    """Return current positions and full trade history."""
    status = etf_engine.get_status()
    return jsonify({"success": True, **status})


@app.route("/api/etf/status", methods=["GET"])
def etf_status():
    """Return last cached scan result."""
    status = etf_engine.get_status()
    return jsonify({"success": True, **status})


# ============ GOLDM Paper Trading ============

@app.route("/api/goldm/scan", methods=["POST"])
def goldm_scan():
    """Scan GOLDM: fetch candles, compute OR, check signals + exits."""
    user_id = request.json.get("user_id") if request.is_json else None
    if not user_id:
        user_id = list(brokers.keys())[0] if brokers else None
    if not user_id or user_id not in brokers:
        return jsonify({"success": False, "error": "No user connected"})

    broker = brokers[user_id]
    if not broker.connected:
        return jsonify({"success": False, "error": "Kite not connected"})

    engine = goldm_engines.get(user_id)
    if not engine:
        return jsonify({"success": False, "error": "No GOLDM engine"})

    result = engine.scan(broker.kite)
    if "error" in result and result.get("or") is None:
        return jsonify({"success": False, "error": result["error"]})

    return jsonify({"success": True, **result})


@app.route("/api/goldm/buy", methods=["POST"])
def goldm_buy():
    """Paper buy GOLDM position."""
    data = request.json
    user_id = data.get("user_id", list(brokers.keys())[0] if brokers else None)
    signal = data.get("signal")
    lots = data.get("lots", 1)

    if not signal:
        return jsonify({"success": False, "error": "No signal provided"})

    engine = goldm_engines.get(user_id)
    if not engine:
        return jsonify({"success": False, "error": "No GOLDM engine"})

    pos, err = engine.add_position(signal, lots=lots)
    if err:
        return jsonify({"success": False, "error": err})

    return jsonify({"success": True, "position": pos})


@app.route("/api/goldm/exit", methods=["POST"])
def goldm_exit():
    """Exit GOLDM position."""
    data = request.json
    user_id = data.get("user_id", list(brokers.keys())[0] if brokers else None)
    exit_price = data.get("exit_price")
    reason = data.get("reason", "MANUAL")

    if not exit_price:
        return jsonify({"success": False, "error": "No exit_price"})

    engine = goldm_engines.get(user_id)
    if not engine:
        return jsonify({"success": False, "error": "No GOLDM engine"})

    trade, err = engine.close_position(exit_price, reason=reason)
    if err:
        return jsonify({"success": False, "error": err})

    return jsonify({"success": True, "trade": trade})


@app.route("/api/goldm/status", methods=["GET"])
def goldm_status():
    """Get GOLDM status: OR, position, closed trades."""
    user_id = request.args.get("user_id", list(brokers.keys())[0] if brokers else None)
    engine = goldm_engines.get(user_id)
    if not engine:
        return jsonify({"success": False, "error": "No GOLDM engine"})

    status = engine.get_status()
    return jsonify({"success": True, **status})


# ============ ETF Core Signal Scheduler ============

def _etf_signal_scheduler():
    """Background thread: scan ETF Core every hour during Indian OR US market hours.

    Indian market window : Mon–Fri  09:00–16:00 IST
    US market window     : Mon–Fri  19:00–02:30 IST (next day)
      → Mon 19:00 – Tue 02:30, ..., Fri 19:00 – Sat 02:30 IST

    Deduplicates: only alerts when signals change vs the previous notification.
    """
    import time
    from datetime import timezone, timedelta
    from data.notifier import send_message

    IST = timezone(timedelta(hours=5, minutes=30))
    SCAN_INTERVAL = 3600  # 1 hour

    # Indian window (IST hours, inclusive start / exclusive end)
    IN_OPEN, IN_CLOSE = 9, 16       # 09:00–16:00

    # US window expressed in IST hours
    US_OPEN  = 19                   # 19:00 IST → ~09:30 EDT / 10:00 EST
    US_CLOSE_NEXT = 2               # 02:00 IST next day → ~16:30 EDT / 20:30 UTC

    def _in_indian_window(dt):
        """Return True if dt is within Indian market hours (Mon-Fri 09:00-16:00 IST)."""
        return dt.weekday() < 5 and IN_OPEN <= dt.hour < IN_CLOSE

    def _in_window(dt):
        """Return True if dt falls inside Indian or US trading window."""
        wd = dt.weekday()   # 0=Mon … 6=Sun
        h  = dt.hour

        # Indian: Mon-Fri 09:00–16:00
        if wd < 5 and IN_OPEN <= h < IN_CLOSE:
            return True

        # US evening leg: Mon-Fri 19:00–23:59
        if wd < 5 and h >= US_OPEN:
            return True

        # US early-morning leg: Tue-Sat 00:00–02:00
        # (Sat 00:00–02:00 is still US Friday session)
        if 0 < wd <= 5 and h < US_CLOSE_NEXT:
            return True

        return False

    def _next_scan_time(dt):
        """Return next scan datetime.

        Indian hours: fires at :15 past each hour (09:15, 10:15, … 15:15 IST).
        Outside Indian hours (US window etc.): next top-of-hour inside window.
        """
        # If within Indian window, align to next :15 mark
        if _in_indian_window(dt):
            next_15 = dt.replace(minute=15, second=0, microsecond=0)
            if next_15 <= dt:
                next_15 += timedelta(hours=1)
            return next_15

        # Outside Indian hours — find next window open (top of hour)
        candidate = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        for _ in range(200):
            if _in_window(candidate):
                # If re-entering Indian window, align to :15
                if _in_indian_window(candidate):
                    return candidate.replace(minute=15, second=0, microsecond=0)
                return candidate
            candidate += timedelta(hours=1)
        return candidate

    def _next_window_open(dt):
        """Return the next datetime when a trading window opens."""
        candidate = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        for _ in range(200):  # safety cap
            if _in_window(candidate):
                return candidate
            candidate += timedelta(hours=1)
        return candidate  # fallback

    print("[ETF scheduler] Started — scans at :15 past each hour 09:15–15:15 IST")

    last_mom20_key        = None   # dedup for Mom20 ranking
    last_overflow_key     = None   # dedup for Mom20 overflow
    startup_scan_done     = False  # one-time ETF scan on startup regardless of window
    startup_scan_done_mom20 = False  # one-time Mom20 scan on startup regardless of window

    while True:
      try:
        now_ist = datetime.now(IST)

        if not _in_window(now_ist) and startup_scan_done:
            target = _next_window_open(now_ist)
            sleep_secs = (target - now_ist).total_seconds()
            print(f"[ETF scheduler] Outside market hours — sleeping until "
                  f"{target.strftime('%Y-%m-%d %H:%M IST')}")
            time.sleep(max(sleep_secs, 60))
            continue

        startup_scan_done = True


        print(f"[ETF scheduler] Scanning at {now_ist.strftime('%Y-%m-%d %H:%M IST')}")

        # ── ETF Z-Score scan ─────────────────────────────────────────────────
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
            from data.etf_core_zscore_backtest import score_live
            from data.notifier import format_etf_zscore_alert
            with ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(score_live)
                try:
                    etf_ranked = _fut.result(timeout=180)   # 3-min hard cap
                except _FutTimeout:
                    raise RuntimeError("score_live() timed out after 3 minutes")
            etf_msg = format_etf_zscore_alert(etf_ranked)
            if etf_msg:
                print(f"[ETF scheduler] ETF Z-Score — alerting")
                send_message(etf_msg)
            else:
                print(f"[ETF scheduler] ETF Z-Score no signals")
        except Exception as e:
            err = f"⚠️ ETF Z-Score scan error: {e}"
            print(f"[ETF scheduler] {err}")
            try:
                send_message(err)
            except Exception:
                pass

        # ── Mom20 + Overflow scan (Indian trading hours only, or startup) ─────────
        if not _in_indian_window(now_ist) and startup_scan_done_mom20:
            print(f"[ETF scheduler] Mom20 — skipped (outside Indian hours 09:00–16:00 IST)")
        else:
            try:
                from data.notifier import format_mom20_alert, format_mom20_overflow_alert
                # Shared scan lock: if a user-triggered background refresh is
                # already running (Tier-1 perf C), don't double-fetch yfinance.
                # Wait for it to finish, then read the now-fresh cache.
                global live_signals_refresh_in_progress
                with live_signals_refresh_lock:
                    user_scan_running = live_signals_refresh_in_progress
                    if not user_scan_running:
                        live_signals_refresh_in_progress = True

                if user_scan_running:
                    print("[ETF scheduler] Mom20 — user-triggered scan in flight, waiting…")
                    import time as _t_wait
                    waited = 0
                    while waited < 180:
                        _t_wait.sleep(2)
                        waited += 2
                        with live_signals_refresh_lock:
                            if not live_signals_refresh_in_progress:
                                break
                    # Read the freshly-updated cache instead of re-scanning
                    mom20_result = live_signals_scanner.scan_entry_signals(force_refresh=False)
                else:
                    try:
                        with ThreadPoolExecutor(max_workers=1) as _ex:
                            _fut = _ex.submit(live_signals_scanner.scan_entry_signals, True)
                            try:
                                mom20_result = _fut.result(timeout=180)
                            except _FutTimeout:
                                raise RuntimeError("scan_entry_signals() timed out after 3 minutes")
                    finally:
                        with live_signals_refresh_lock:
                            live_signals_refresh_in_progress = False

                _persist_mom20_ranks_to_users(mom20_result)

                # Mom20 top-40 — dedup by ticker+score (rounded 1dp)
                mom20_msg = format_mom20_alert(mom20_result)
                if mom20_msg:
                    mom20_key = tuple(
                        (s["ticker"], round(s.get("momentum_score", 0), 1))
                        for s in mom20_result.get("mom20_signals", [])
                    )
                    if mom20_key != last_mom20_key:
                        print(f"[ETF scheduler] Mom20 ranking changed — alerting")
                        send_message(mom20_msg)
                        last_mom20_key = mom20_key
                    else:
                        print(f"[ETF scheduler] Mom20 ranking unchanged — skipping")
                else:
                    print(f"[ETF scheduler] Mom20 no signals")
                    last_mom20_key = None

                # Mom20 overflow — dedup by ticker+score (rounded 1dp)
                overflow_msg = format_mom20_overflow_alert(mom20_result)
                if overflow_msg:
                    overflow_key = tuple(
                        (s["ticker"], round(s.get("momentum_score", 0), 1))
                        for s in mom20_result.get("mom20_overflow", [])
                    )
                    if overflow_key != last_overflow_key:
                        n_ov = len(mom20_result.get("mom20_overflow", []))
                        print(f"[ETF scheduler] Mom20 overflow changed — {n_ov} candidates — alerting")
                        send_message(overflow_msg)
                        last_overflow_key = overflow_key
                    else:
                        print(f"[ETF scheduler] Mom20 overflow unchanged — skipping")
                else:
                    print(f"[ETF scheduler] Mom20 overflow — none")
                    last_overflow_key = None

            except Exception as e:
                err = f"⚠️ Mom20 scan error: {e}"
                print(f"[ETF scheduler] {err}")
                try:
                    send_message(err)
                except Exception:
                    pass
            finally:
                startup_scan_done_mom20 = True

        # Sleep until next :15-past-the-hour scan (or next window open)
        now_ist = datetime.now(IST)
        next_scan = _next_scan_time(now_ist)
        sleep_secs = (next_scan - now_ist).total_seconds()
        print(f"[ETF scheduler] Next scan at {next_scan.strftime('%H:%M IST')} (sleep {int(sleep_secs)}s)")
        time.sleep(max(sleep_secs, 30))

      except Exception as _loop_err:
          print(f"[ETF scheduler] !! Loop exception (will retry in 60s): {_loop_err}")
          try:
              send_message(f"⚠️ Scheduler loop error: {_loop_err}")
          except Exception:
              pass
          time.sleep(60)


# ============ User Registry ============

from data.user_registry import (
    load_users, add_user, get_user, update_user, delete_user, ensure_all_dirs,
    mom20_portfolio_path, mom20_history_path, mom20_live_prices_path,
    techmo_portfolio_path, techmo_history_path, techmo_live_prices_path,
    etf_positions_path, etf_history_path,
    baskets_dir, trade_books_dir,
)
from data.mom20_basket import generate_basket, to_zerodha_csv, parse_trade_book, sync_portfolio_from_trades

ensure_all_dirs()


# ── Disk-backed caches (Tier-1 perf, plan item A) ────────────────────────────
# Three caches: sector ranking, ETF LTPs, sector map. All disk-persisted so a
# `systemctl restart` doesn't cold-start the user's first click. Caches are
# *also* pre-warmed at startup (see _prewarm_caches in run_server).
_CACHE_DIR_A = DATA_STORE_PATH
_SECTOR_RANK_CACHE_FILE = os.path.join(_CACHE_DIR_A, "sector_ranking_cache.json")
_ETF_LTP_CACHE_FILE     = os.path.join(_CACHE_DIR_A, "etf_ltp_cache.json")

# Sector ranking — 15-min TTL.
_SECTOR_RANK_CACHE = {"ranking": None, "timestamp": 0,
                      "prev_sector_ranks": {}, "prev_sector_rank_date": ""}

# ETF LTP — per-symbol 5-min TTL (LTPs change intraday but a few minutes of
# staleness is fine for our use cases).
_ETF_LTP_CACHE = {}        # {symbol: (price, fetched_at)}
_ETF_LTP_TTL   = 300

# Sector map — rarely changes (only on N200 reconstitution / our manual edits)
# so cache forever once loaded.
_SECTOR_MAP_CACHE = None


def _load_sector_rank_disk():
    """Load sector_ranking_cache.json into memory if present."""
    try:
        with open(_SECTOR_RANK_CACHE_FILE) as f:
            d = json.load(f)
        if d.get("ranking"):
            _SECTOR_RANK_CACHE["ranking"]               = d["ranking"]
            _SECTOR_RANK_CACHE["timestamp"]             = float(d.get("timestamp", 0))
            _SECTOR_RANK_CACHE["prev_sector_ranks"]     = d.get("prev_sector_ranks") or {}
            _SECTOR_RANK_CACHE["prev_sector_rank_date"] = d.get("prev_sector_rank_date") or ""
    except Exception:
        pass


def _save_sector_rank_disk():
    try:
        os.makedirs(_CACHE_DIR_A, exist_ok=True)
        today_str = datetime.now().strftime("%Y-%m-%d")
        prev_ranks = {}
        prev_rank_date = ""
        try:
            with open(_SECTOR_RANK_CACHE_FILE) as _f:
                _existing = json.load(_f)
            existing_ts = float(_existing.get("timestamp", 0))
            if existing_ts:
                existing_date = datetime.fromtimestamp(existing_ts).strftime("%Y-%m-%d")
                if existing_date < today_str:
                    # New day: promote existing on-disk ranking as prev_sector_ranks
                    prev_ranks = {r["symbol"]: r["rank"]
                                  for r in (_existing.get("ranking") or []) if r.get("rank")}
                    prev_rank_date = existing_date
                else:
                    prev_ranks     = _existing.get("prev_sector_ranks") or {}
                    prev_rank_date = _existing.get("prev_sector_rank_date") or ""
        except Exception:
            pass
        with open(_SECTOR_RANK_CACHE_FILE, "w") as f:
            json.dump({
                "ranking":               _SECTOR_RANK_CACHE["ranking"],
                "timestamp":             _SECTOR_RANK_CACHE["timestamp"],
                "prev_sector_ranks":     prev_ranks,
                "prev_sector_rank_date": prev_rank_date,
            }, f)
        _SECTOR_RANK_CACHE["prev_sector_ranks"]     = prev_ranks
        _SECTOR_RANK_CACHE["prev_sector_rank_date"] = prev_rank_date
    except Exception as e:
        print(f"[cache] sector rank disk save failed: {e}")


def _load_etf_ltp_disk():
    """Load etf_ltp_cache.json into memory if present."""
    try:
        with open(_ETF_LTP_CACHE_FILE) as f:
            d = json.load(f)
        for sym, val in d.items():
            _ETF_LTP_CACHE[sym] = (float(val.get("price", 0)),
                                   float(val.get("ts", 0)))
    except Exception:
        pass


def _save_etf_ltp_disk():
    try:
        os.makedirs(_CACHE_DIR_A, exist_ok=True)
        out = {sym: {"price": p, "ts": ts}
               for sym, (p, ts) in _ETF_LTP_CACHE.items()}
        with open(_ETF_LTP_CACHE_FILE, "w") as f:
            json.dump(out, f)
    except Exception as e:
        print(f"[cache] ETF LTP disk save failed: {e}")


def _fetch_etf_ltp(symbols):
    """Fetch live close for a list of NSE ETF symbols (no .NS suffix needed
    in input). Returns {symbol: price}, missing symbols simply absent.
    Cached per-symbol for 5 min in-memory + disk-persisted."""
    if not symbols:
        return {}
    import time as _t
    now = _t.time()
    out, missing = {}, []
    for s in symbols:
        cached = _ETF_LTP_CACHE.get(s)
        if cached and (now - cached[1]) < _ETF_LTP_TTL and cached[0] > 0:
            out[s] = cached[0]
        else:
            missing.append(s)

    if not missing:
        return out

    import yfinance as yf
    yf_syms = [s if "." in s else f"{s}.NS" for s in missing]
    try:
        df = yf.download(yf_syms, period="5d", progress=False,
                         auto_adjust=True, group_by="ticker",
                         threads=True, timeout=30)
    except Exception:
        return out  # serve whatever we already have cached

    fetched_any = False
    for orig, ysym in zip(missing, yf_syms):
        try:
            s = (df[ysym]["Close"] if len(yf_syms) > 1 else df["Close"]).dropna()
            if len(s):
                price = float(s.iloc[-1])
                out[orig] = price
                _ETF_LTP_CACHE[orig] = (price, now)
                fetched_any = True
        except Exception:
            pass

    if fetched_any:
        _save_etf_ltp_disk()
    return out


def _get_sector_ranking(force_refresh=False):
    """Return today's full Phase 1 sector Z-score ranking (list of 19 dicts).
    15-min TTL; backed by disk cache so a restart doesn't cold-start the
    first user click."""
    import time as _t
    if (not force_refresh and _SECTOR_RANK_CACHE["ranking"]
            and (_t.time() - _SECTOR_RANK_CACHE["timestamp"]) < 900):
        return _SECTOR_RANK_CACHE["ranking"]
    try:
        from data.score_live_sectors import score_live_sectors
        ranking = score_live_sectors()
        if ranking:
            _SECTOR_RANK_CACHE["ranking"]   = ranking
            _SECTOR_RANK_CACHE["timestamp"] = _t.time()
            _save_sector_rank_disk()
        return ranking
    except Exception as e:
        print(f"[mom20] sector ranking failed: {e}")
        # If compute failed but we have stale disk data, serve it.
        return _SECTOR_RANK_CACHE["ranking"] or []


def _get_top5_sectors(force_refresh=False):
    """Top-5 sector names (used by Mom20 ETF top-up). Slices full ranking."""
    return [r["symbol"] for r in _get_sector_ranking(force_refresh)[:5]]


def _load_sector_map():
    """ticker → primary_sector for current N200 (Phase 2 sector map CSV).
    Cached for the lifetime of the process — file is small and changes only
    on N200 reconstitution or manual sector-map updates."""
    global _SECTOR_MAP_CACHE
    if _SECTOR_MAP_CACHE is not None:
        return _SECTOR_MAP_CACHE
    import csv as _csv
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "nse_const", "nifty200_sector_map.csv")
    try:
        with open(path) as f:
            _SECTOR_MAP_CACHE = {row["ticker"]: row["primary_sector"]
                                 for row in _csv.DictReader(f)}
    except Exception:
        _SECTOR_MAP_CACHE = {}
    return _SECTOR_MAP_CACHE


def _prewarm_caches():
    """Background thread: hydrate from disk + refresh in-memory caches so the
    first user click after restart is fast. Started from run_server() as a
    daemon thread so it doesn't block Flask boot."""
    import time as _t
    _t.sleep(3)  # let Flask finish binding the port
    try:
        print("[prewarm] hydrating disk caches…")
        _load_sector_rank_disk()
        _load_etf_ltp_disk()
        _load_sector_map()
        # If disk cache is missing/old, do one fresh compute.
        ranking = _get_sector_ranking(force_refresh=False)
        from data.sector_etf_map import SECTOR_TO_ETF, KNOWN_ETF_SYMBOLS
        top5 = [r["symbol"] for r in ranking[:5]]
        etf_syms = list({SECTOR_TO_ETF[s][0] for s in top5
                         if SECTOR_TO_ETF.get(s)} | KNOWN_ETF_SYMBOLS)
        if etf_syms:
            _fetch_etf_ltp(etf_syms)
        print(f"[prewarm] done — {len(ranking)} sectors, "
              f"{len(_ETF_LTP_CACHE)} ETFs cached")
    except Exception as e:
        print(f"[prewarm] error: {e}")


def _load_user_live_prices(user_id):
    """Read prices map from a user's mom20_live_prices.json. Returns {} on miss."""
    try:
        with open(mom20_live_prices_path(user_id)) as f:
            return {k: float(v) for k, v in (json.load(f).get("prices") or {}).items()}
    except Exception:
        return {}


def _persist_mom20_ranks_to_users(scan_result):
    """After a Live Signals scan, write fresh mom20 ranks into each user's
    mom20_live_prices.json. Preserves prices and updated_at; only ranks change.
    """
    if not scan_result:
        return
    rank_map = {s["ticker"]: s.get("rank")
                for s in scan_result.get("mom20_signals", [])
                if s.get("rank") is not None}
    for t, r in (scan_result.get("mom20_unfiltered_ranks") or {}).items():
        rank_map.setdefault(t, r)
    if not rank_map:
        return
    for u in load_users():
        path = mom20_live_prices_path(u["id"])
        try:
            try:
                with open(path) as f:
                    cur = json.load(f)
            except Exception:
                cur = {}
            cur["ranks"] = rank_map
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(cur, f, indent=2)
        except Exception:
            pass


@app.route("/api/portfolio-users", methods=["GET"])
def api_get_users():
    return jsonify({"success": True, "users": load_users()})


@app.route("/api/portfolio-users", methods=["POST"])
def api_add_user():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "name required"})
    mom20_cap = int(data.get("mom20_capital") or 0)
    etf_cap   = int(data.get("etf_capital")   or 0)
    user = add_user(name, mom20_capital=mom20_cap, etf_capital=etf_cap)
    return jsonify({"success": True, "user": user})


@app.route("/api/portfolio-users/<user_id>", methods=["PATCH"])
def api_update_user(user_id):
    data = request.get_json() or {}
    mom20_cap = data.get("mom20_capital")
    etf_cap   = data.get("etf_capital")
    user = update_user(user_id,
                       mom20_capital=int(mom20_cap) if mom20_cap is not None else None,
                       etf_capital=int(etf_cap)     if etf_cap   is not None else None)
    if not user:
        return jsonify({"success": False, "error": "user not found"})
    return jsonify({"success": True, "user": user})


@app.route("/api/portfolio-users/<user_id>", methods=["DELETE"])
def api_delete_user(user_id):
    ok = delete_user(user_id)
    if not ok:
        return jsonify({"success": False, "error": "user not found"})
    return jsonify({"success": True})


# ── Mom20 sector ranking (shared, no user) ────────────────────────────────────

@app.route("/api/sector-ranking", methods=["GET"])
def api_sector_ranking():
    """Today's Phase 1 sector Z-score ranking (19 sectors). 15-min TTL."""
    from datetime import date as _date
    force = request.args.get("refresh", "").lower() in ("1", "true", "yes")
    ranking = _get_sector_ranking(force_refresh=force)
    if not ranking:
        return jsonify({"success": False, "error": "ranking unavailable"})
    prev_ranks = _SECTOR_RANK_CACHE.get("prev_sector_ranks") or {}
    ranked_with_delta = []
    for r in ranking:
        entry = dict(r)
        prev_r = prev_ranks.get(r["symbol"])
        curr_r = r.get("rank")
        entry["rank_delta"] = (prev_r - curr_r
                               if prev_r is not None and curr_r is not None else None)
        ranked_with_delta.append(entry)
    return jsonify({
        "success":        True,
        "as_of":          _date.today().isoformat(),
        "ranking":        ranked_with_delta,
        "top5":           [r["symbol"] for r in ranking[:5]],
        "cached_at":      _SECTOR_RANK_CACHE["timestamp"],
        "prev_rank_date": _SECTOR_RANK_CACHE.get("prev_sector_rank_date") or "",
    })


# ── Sector map (ticker → primary sector, current N200) ───────────────────────

@app.route("/api/sector-map", methods=["GET"])
def api_sector_map():
    """Return {ticker: primary_sector} for current Nifty 200, cached."""
    return jsonify({"success": True, "map": _load_sector_map()})


# ── Mom20 basket (per user) ────────────────────────────────────────────────────

@app.route("/api/portfolio-users/<user_id>/mom20-basket", methods=["GET"])
def api_mom20_basket_preview(user_id):
    """Preview basket: exits, entries, quantities for this user's capital."""
    user = get_user(user_id)
    if not user:
        return jsonify({"success": False, "error": "user not found"})

    # Get current Mom20 signals
    try:
        result = live_signals_scanner.scan_entry_signals(force_refresh=False)
        signals = result.get("mom20_signals", [])
        unfiltered_ranks = result.get("mom20_unfiltered_ranks", {})
        all_prices = dict(result.get("mom20_all_prices", {}))
        mom20_overflow = result.get("mom20_overflow", [])
    except Exception as e:
        return jsonify({"success": False, "error": f"signals unavailable: {e}"})

    # User's live prices (from "⚡ Live Prices" click) are fresher than the
    # scanner cache — override scanner prices so exits/entries show current ₹.
    # Also covers held tickers no longer in N200 (e.g. post-reconstitution).
    for _t, _p in _load_user_live_prices(user_id).items():
        if _p:
            all_prices[_t] = _p

    # Load user's current portfolio
    pf_path = mom20_portfolio_path(user_id)
    try:
        with open(pf_path) as f:
            portfolio = json.load(f)
    except Exception:
        portfolio = {"status": "empty", "basket": []}

    # ETF top-up inputs (Q2): today's top-5 sectors + sector classifier + ETF prices
    _top5     = _get_top5_sectors()
    _sec_map  = _load_sector_map()
    from data.sector_etf_map import SECTOR_TO_ETF, KNOWN_ETF_SYMBOLS
    _need_etfs = [SECTOR_TO_ETF[s][0] for s in _top5
                  if SECTOR_TO_ETF.get(s)]
    # Also fetch LTPs for any *already-held* ETFs so the Tracker / Exit row
    # shows a real price even after the ETF's sector drops out of top-5.
    _held_etfs = [item["ticker"] for item in portfolio.get("basket", [])
                  if item.get("ticker") in KNOWN_ETF_SYMBOLS]
    # Top-3 overflow sectors → also need their ETF prices for the new
    # overflow→ETF top-up path inside generate_basket.
    _overflow_sec_count = {}
    for o in mom20_overflow:
        _sec = _sec_map.get(o.get("ticker"))
        if _sec:
            _overflow_sec_count[_sec] = _overflow_sec_count.get(_sec, 0) + 1
    _top_overflow_secs = [s for s, _ in
                          sorted(_overflow_sec_count.items(), key=lambda x: -x[1])[:3]]
    _overflow_etfs = [SECTOR_TO_ETF[s][0] for s in _top_overflow_secs
                      if SECTOR_TO_ETF.get(s)]
    _need_etfs = list(set(_need_etfs + _held_etfs + _overflow_etfs))
    _etf_prices = _fetch_etf_ltp(_need_etfs) if _need_etfs else {}
    basket_data = generate_basket(user, signals, portfolio,
                                  unfiltered_ranks=unfiltered_ranks,
                                  all_prices=all_prices,
                                  sector_map=_sec_map,
                                  top5_sectors=_top5,
                                  etf_prices=_etf_prices,
                                  mom20_overflow=mom20_overflow)

    # Append overflow candidates (combined unfiltered rank ≤ 15 = buffer_in, β > 1.2) as optional entries
    capital    = user.get("strategies", {}).get("mom20", {}).get("capital", 0) or 0
    per_slot   = capital / 20 if capital else 0
    held_set   = {h["ticker"] for h in portfolio.get("basket", [])}
    overflow_entries = []
    for ov in mom20_overflow:
        combined_rank = unfiltered_ranks.get(ov["ticker"], 999)
        if combined_rank > 15:   # same buffer_in as main basket
            continue
        if ov["ticker"] in held_set:
            continue
        px  = all_prices.get(ov["ticker"], ov.get("price", 0)) or 0
        qty = int(per_slot / px) if px > 0 else 0
        overflow_entries.append({
            "ticker":           ov["ticker"],
            "rank":             combined_rank,   # combined unfiltered rank
            "ov_rank":          ov["rank"],       # rank within overflow list
            "price":            round(px, 2),
            "qty":              qty,
            "capital_allocated": round(qty * px, 0),
            "score":            ov.get("momentum_score", 0),
            "beta":             ov.get("beta", 0),
            "ret_12m":          ov.get("ret_12m", 0),
            "ret_3m":           ov.get("ret_3m", 0),
            "is_overflow":      True,
        })
    basket_data["overflow_entries"] = overflow_entries
    return jsonify({"success": True, **basket_data})


@app.route("/api/portfolio-users/<user_id>/mom20-basket/download", methods=["GET"])
def api_mom20_basket_download(user_id):
    """Generate and download Zerodha basket CSV for this user."""
    import flask
    user = get_user(user_id)
    if not user:
        return jsonify({"success": False, "error": "user not found"})

    # Allow UI to override capital without saving to profile
    capital_override = request.args.get("capital", type=int)
    if capital_override and capital_override > 0:
        import copy
        user = copy.deepcopy(user)
        user.setdefault("strategies", {}).setdefault("mom20", {})["capital"] = capital_override

    try:
        result = live_signals_scanner.scan_entry_signals(force_refresh=False)
        signals = result.get("mom20_signals", [])
        unfiltered_ranks = result.get("mom20_unfiltered_ranks", {})
        all_prices = dict(result.get("mom20_all_prices", {}))
        mom20_overflow = result.get("mom20_overflow", [])
    except Exception as e:
        return jsonify({"success": False, "error": f"signals unavailable: {e}"})

    # User's live prices (from "⚡ Live Prices" click) are fresher than the
    # scanner cache — override scanner prices so exits/entries show current ₹.
    # Also covers held tickers no longer in N200 (e.g. post-reconstitution).
    for _t, _p in _load_user_live_prices(user_id).items():
        if _p:
            all_prices[_t] = _p

    pf_path = mom20_portfolio_path(user_id)
    try:
        with open(pf_path) as f:
            portfolio = json.load(f)
    except Exception:
        portfolio = {"status": "empty", "basket": []}

    # ETF top-up inputs (Q2): today's top-5 sectors + sector classifier + ETF prices
    _top5     = _get_top5_sectors()
    _sec_map  = _load_sector_map()
    from data.sector_etf_map import SECTOR_TO_ETF, KNOWN_ETF_SYMBOLS
    _need_etfs = [SECTOR_TO_ETF[s][0] for s in _top5
                  if SECTOR_TO_ETF.get(s)]
    # Also fetch LTPs for any *already-held* ETFs so the Tracker / Exit row
    # shows a real price even after the ETF's sector drops out of top-5.
    _held_etfs = [item["ticker"] for item in portfolio.get("basket", [])
                  if item.get("ticker") in KNOWN_ETF_SYMBOLS]
    # Top-3 overflow sectors → also need their ETF prices.
    _overflow_sec_count = {}
    for o in mom20_overflow:
        _sec = _sec_map.get(o.get("ticker"))
        if _sec:
            _overflow_sec_count[_sec] = _overflow_sec_count.get(_sec, 0) + 1
    _top_overflow_secs = [s for s, _ in
                          sorted(_overflow_sec_count.items(), key=lambda x: -x[1])[:3]]
    _overflow_etfs = [SECTOR_TO_ETF[s][0] for s in _top_overflow_secs
                      if SECTOR_TO_ETF.get(s)]
    _need_etfs = list(set(_need_etfs + _held_etfs + _overflow_etfs))
    _etf_prices = _fetch_etf_ltp(_need_etfs) if _need_etfs else {}
    basket_data = generate_basket(user, signals, portfolio,
                                  unfiltered_ranks=unfiltered_ranks,
                                  all_prices=all_prices,
                                  sector_map=_sec_map,
                                  top5_sectors=_top5,
                                  etf_prices=_etf_prices,
                                  mom20_overflow=mom20_overflow)
    from data.mom20_basket import to_zerodha_json
    json_content = to_zerodha_json(basket_data)

    # Save a copy locally
    import datetime
    first_name = user.get("name", "investor").split()[0].lower()
    fname = f"mom20_{datetime.date.today().isoformat()}_{first_name}.json"
    save_path = os.path.join(baskets_dir(user_id), fname)
    os.makedirs(baskets_dir(user_id), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(json_content)

    return flask.Response(
        json_content,
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )


@app.route("/api/portfolio-users/<user_id>/mom20-tradebook", methods=["POST"])
def api_mom20_tradebook_upload(user_id):
    """Upload Zerodha trade book CSV → sync user's Mom20 portfolio."""
    user = get_user(user_id)
    if not user:
        return jsonify({"success": False, "error": "user not found"})

    if "file" not in request.files:
        return jsonify({"success": False, "error": "no file uploaded"})

    csv_content = request.files["file"].read().decode("utf-8", errors="replace")
    trades = parse_trade_book(csv_content)
    if not trades:
        return jsonify({"success": False, "error": "no valid trades found in CSV"})

    # Load last basket data for entry context
    import glob as _glob
    bdir = baskets_dir(user_id)
    basket_files = sorted(_glob.glob(os.path.join(bdir, "mom20_*.csv")))
    basket_data = {}  # fallback if no basket saved

    pf_path = mom20_portfolio_path(user_id)
    try:
        with open(pf_path) as f:
            portfolio = json.load(f)
    except Exception:
        portfolio = {"status": "empty", "basket": []}

    # Enrich SELL trades with entry price + realized P&L from current portfolio
    # (must happen before sync_portfolio_from_trades removes the positions).
    basket_lookup = {item["ticker"]: item for item in portfolio.get("basket", [])}
    for t in trades:
        if t["action"] == "SELL" and t["ticker"] in basket_lookup:
            holding    = basket_lookup[t["ticker"]]
            entry_p    = holding.get("entry_price", 0)
            t["entry_price"] = entry_p
            t["entry_date"]  = holding.get("entry_date", "")
            if entry_p:
                t["pnl"]     = round((t["price"] - entry_p) * t["qty"], 2)
                t["pnl_pct"] = round((t["price"] / entry_p - 1) * 100, 2)

    updated = sync_portfolio_from_trades(portfolio, trades, basket_data)

    # Compute realized P&L summary for this rebalance
    sells_with_pnl = [t for t in trades if t["action"] == "SELL" and "pnl" in t]
    realized_pnl   = round(sum(t["pnl"] for t in sells_with_pnl), 2)

    # Save updated portfolio
    import datetime
    tmp = pf_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(updated, f, indent=2)
    os.replace(tmp, pf_path)

    # Save trade book file
    tb_fname = f"zerodha_{datetime.date.today().isoformat()}.csv"
    tb_path = os.path.join(trade_books_dir(user_id), tb_fname)
    os.makedirs(trade_books_dir(user_id), exist_ok=True)
    with open(tb_path, "w") as f:
        f.write(csv_content)

    # Append to history
    hist_path = mom20_history_path(user_id)
    try:
        with open(hist_path) as f:
            history = json.load(f)
    except Exception:
        history = []

    history.append({
        "rebalance_date":      datetime.date.today().isoformat(),
        "trade_book_uploaded": datetime.datetime.now().isoformat(),
        "trades_parsed":       len(trades),
        "buys":                [t for t in trades if t["action"] == "BUY"],
        "sells":               [t for t in trades if t["action"] == "SELL"],
        "realized_pnl":        realized_pnl,
        "status":              "synced",
    })
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return jsonify({"success": True, "trades_synced": len(trades),
                    "portfolio_size": len(updated.get("basket", []))})


# ── Mom20 seed portfolio (manual entry) ───────────────────────────────────────

@app.route("/api/portfolio-users/<user_id>/mom20-seed", methods=["POST"])
def api_mom20_seed(user_id):
    """Manually seed Mom20 portfolio (existing holdings before first trade book upload)."""
    user = get_user(user_id)
    if not user:
        return jsonify({"success": False, "error": "user not found"})
    data = request.get_json() or {}
    holdings = data.get("holdings", [])

    basket = []
    for h in holdings:
        ticker = (h.get("ticker") or "").strip().upper()
        qty    = int(h.get("qty") or 0)
        price  = float(h.get("entry_price") or 0)
        if not ticker or qty <= 0 or price <= 0:
            continue
        basket.append({
            "ticker":      ticker,
            "qty":         qty,
            "entry_price": round(price, 2),
            "entry_date":  h.get("entry_date") or datetime.date.today().isoformat(),
            "weight":      round(100 / 20, 2),
            "source":      "manual_seed",
        })

    pf_path = mom20_portfolio_path(user_id)
    portfolio = {
        "status":      "seeded" if basket else "empty",
        "basket":      basket,
        "last_synced": datetime.date.today().isoformat(),
    }
    tmp = pf_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(portfolio, f, indent=2)
    os.replace(tmp, pf_path)

    return jsonify({"success": True, "seeded": len(basket)})


def _retrospective_realized_pnl(history: list) -> float:
    """Walk history chronologically to compute realized P&L for sells that
    predate the enrichment feature (missing 'pnl' field).  Returns total."""
    pos_book = {}  # ticker → entry_price (weighted avg across rebalances)
    total = 0.0
    for rb in history:
        for t in rb.get("buys", []):
            ticker = t.get("ticker")
            qty    = t.get("qty", 0)
            price  = t.get("price", 0)
            if not ticker or not qty or not price:
                continue
            if ticker in pos_book:
                old = pos_book[ticker]
                new_qty = old["qty"] + qty
                pos_book[ticker] = {
                    "entry_price": round((old["qty"] * old["entry_price"] + qty * price) / new_qty, 2),
                    "qty": new_qty,
                }
            else:
                pos_book[ticker] = {"entry_price": price, "qty": qty}

        for t in rb.get("sells", []):
            ticker = t.get("ticker")
            pnl    = t.get("pnl")
            if pnl is None and ticker in pos_book:
                ep  = pos_book[ticker]["entry_price"]
                pnl = round((t.get("price", 0) - ep) * t.get("qty", 0), 2)
                t["entry_price"] = ep
                t["pnl"]         = pnl
                t["pnl_pct"]     = round((t.get("price", 0) / ep - 1) * 100, 2) if ep else 0
            total += pnl or 0
            pos_book.pop(ticker, None)
    return round(total, 2)


@app.route("/api/portfolio-users/<user_id>/mom20-rebuild", methods=["POST"])
def api_mom20_rebuild_portfolio(user_id):
    """Replay all rebalance history records to reconstruct correct portfolio qtys."""
    import datetime as _dt_mod
    user = get_user(user_id)
    if not user:
        return jsonify({"success": False, "error": "user not found"})
    hist_path = mom20_history_path(user_id)
    try:
        with open(hist_path) as f:
            history = json.load(f)
    except Exception:
        return jsonify({"success": False, "error": "no history found"})

    basket = {}
    for rb in history:
        for sell in rb.get("sells", []):
            basket.pop(sell.get("ticker"), None)
        for buy in rb.get("buys", []):
            ticker = buy.get("ticker")
            qty    = buy.get("qty", 0)
            price  = buy.get("price", 0)
            date   = buy.get("trade_date", rb.get("rebalance_date", ""))
            if not ticker or qty <= 0:
                continue
            if ticker in basket:
                old_qty   = basket[ticker]["qty"]
                old_price = basket[ticker]["entry_price"]
                new_qty   = old_qty + qty
                wavg      = (old_qty * old_price + qty * price) / new_qty if new_qty else price
                basket[ticker] = {**basket[ticker],
                                  "qty": new_qty,
                                  "entry_price": round(wavg, 2),
                                  "last_added_date": date,
                                  "last_added_price": round(price, 2)}
            else:
                basket[ticker] = {"ticker": ticker, "qty": qty,
                                  "entry_price": round(price, 2),
                                  "entry_date": date,
                                  "weight": round(100 / 20, 2)}

    pf_path = mom20_portfolio_path(user_id)
    portfolio = {"status": "invested" if basket else "empty",
                 "basket": list(basket.values()),
                 "last_synced": _dt_mod.date.today().isoformat(),
                 "rebuilt_from_history": _dt_mod.datetime.now().isoformat()}
    tmp = pf_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(portfolio, f, indent=2)
    os.replace(tmp, pf_path)
    return jsonify({"success": True, "positions": len(basket),
                    "basket": list(basket.values())})


@app.route("/api/portfolio-users/<user_id>/mom20-history/<int:idx>", methods=["DELETE"])
def api_mom20_delete_history(user_id, idx):
    """Delete one rebalance history record by its 0-based storage index, then rebuild portfolio."""
    import datetime as _dt_mod
    user = get_user(user_id)
    if not user:
        return jsonify({"success": False, "error": "user not found"})
    hist_path = mom20_history_path(user_id)
    try:
        with open(hist_path) as f:
            history = json.load(f)
    except Exception:
        return jsonify({"success": False, "error": "no history found"})
    if idx < 0 or idx >= len(history):
        return jsonify({"success": False, "error": f"index {idx} out of range (0–{len(history)-1})"})

    removed = history.pop(idx)
    tmp = hist_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, hist_path)

    # Rebuild portfolio from remaining history
    basket = {}
    for rb in history:
        for sell in rb.get("sells", []):
            basket.pop(sell.get("ticker"), None)
        for buy in rb.get("buys", []):
            ticker = buy.get("ticker")
            qty    = buy.get("qty", 0)
            price  = buy.get("price", 0)
            date   = buy.get("trade_date", rb.get("rebalance_date", ""))
            if not ticker or qty <= 0:
                continue
            if ticker in basket:
                old_qty   = basket[ticker]["qty"]
                old_price = basket[ticker]["entry_price"]
                new_qty   = old_qty + qty
                wavg      = (old_qty * old_price + qty * price) / new_qty if new_qty else price
                basket[ticker] = {**basket[ticker], "qty": new_qty,
                                  "entry_price": round(wavg, 2),
                                  "last_added_date": date,
                                  "last_added_price": round(price, 2)}
            else:
                basket[ticker] = {"ticker": ticker, "qty": qty,
                                  "entry_price": round(price, 2),
                                  "entry_date": date,
                                  "weight": round(100 / 20, 2)}

    pf_path = mom20_portfolio_path(user_id)
    portfolio = {"status": "invested" if basket else "empty",
                 "basket": list(basket.values()),
                 "last_synced": _dt_mod.date.today().isoformat(),
                 "rebuilt_from_history": _dt_mod.datetime.now().isoformat()}
    tmp = pf_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(portfolio, f, indent=2)
    os.replace(tmp, pf_path)

    return jsonify({"success": True,
                    "deleted": removed.get("rebalance_date"),
                    "remaining": len(history),
                    "positions": len(basket)})


@app.route("/api/portfolio-users/<user_id>/mom20-history", methods=["GET"])
def api_mom20_history(user_id):
    """Return rebalance history for a user (most recent first) with P&L."""
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(mom20_history_path(user_id)) as f:
            history = json.load(f)
    except Exception:
        history = []

    # Enrich sells missing P&L (pre-enrichment uploads) and patch rebal totals
    _retrospective_realized_pnl(history)
    for rb in history:
        if rb.get("realized_pnl") is None:
            rb["realized_pnl"] = round(sum(t.get("pnl", 0) for t in rb.get("sells", [])), 2)

    cumulative_realized = round(sum(rb.get("realized_pnl", 0) for rb in history), 2)
    total_invested = round(sum(t.get("qty", 0) * t.get("price", 0)
                               for rb in history for t in rb.get("buys", [])), 2)
    return jsonify({
        "success": True,
        "history": list(reversed(history)),
        "cumulative_realized_pnl": cumulative_realized,
        "total_invested": total_invested,
    })


@app.route("/api/portfolio-users/<user_id>/mom20-portfolio", methods=["GET"])
def api_mom20_portfolio_get(user_id):
    """Return current portfolio basket for seed editor pre-fill."""
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(mom20_portfolio_path(user_id)) as f:
            pf = json.load(f)
    except Exception:
        pf = {"status": "empty", "basket": []}
    return jsonify({"success": True, "portfolio": pf})


@app.route("/api/portfolio-users/<user_id>/mom20-performance", methods=["GET"])
def api_mom20_performance(user_id):
    """Return holdings with live prices and P&L for portfolio tracker.
    ?live=1  → fetch fresh yfinance prices, persist to mom20_live_prices.json, return them
    default  → read persisted prices+ranks from mom20_live_prices.json (instant, no network)
    """
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(mom20_portfolio_path(user_id)) as f:
            pf = json.load(f)
    except Exception:
        pf = {"status": "empty", "basket": []}

    basket = pf.get("basket", [])
    want_live = request.args.get("live") == "1"
    lp_path = mom20_live_prices_path(user_id)

    # Load persisted prices + ranks
    price_map = {}
    rank_map = {}
    prices_updated_at = None
    try:
        with open(lp_path) as f:
            lp = json.load(f)
        price_map = {k: float(v) for k, v in (lp.get("prices") or {}).items()}
        rank_map  = dict(lp.get("ranks") or {})
        prices_updated_at = lp.get("updated_at")
    except Exception:
        pass

    # Overlay ranks from live signals cache (kept fresh by scheduler) so the
    # tracker always shows the same rank as the Live Signals tab — no stale data.
    prev_ranks = {}
    try:
        from config.settings import LIVE_SIGNALS_CACHE_FILE
        with open(LIVE_SIGNALS_CACHE_FILE) as f:
            ls = json.load(f)
        live_ranks = ls.get("mom20_unfiltered_ranks", {})
        if live_ranks:
            rank_map.update({k: v for k, v in live_ranks.items() if v is not None})
        prev_ranks = ls.get("prev_ranks") or {}
    except Exception:
        pass

    if not basket:
        return jsonify({"success": True, "holdings": [], "initial_capital": 0,
                        "total_entry_value": 0, "total_current_value": 0,
                        "unrealized_pnl": 0, "realized_pnl": 0, "total_pnl": 0,
                        "total_return_pct": 0, "tracking_since": None,
                        "prices_updated_at": prices_updated_at})

    tickers = [h["ticker"] for h in basket]

    # When Live Prices is clicked from the basket page, the JS passes the
    # union of entry+exit tickers via ?extra_tickers=A,B,C so a single bulk
    # fetch refreshes prices for the tracker AND the basket panel.
    extra = [t.strip() for t in request.args.get("extra_tickers", "").split(",")
             if t.strip()]
    fetch_tickers = list(dict.fromkeys(tickers + extra))   # dedup, preserve order
    fresh_prices = {}

    if want_live and fetch_tickers:
        import yfinance as yf
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        yf_syms = [f"{t}.NS" for t in fetch_tickers]
        try:
            df = yf.download(yf_syms, period="1d", interval="5m", progress=False,
                             group_by="ticker", threads=True, auto_adjust=True)
            for t, sym in zip(fetch_tickers, yf_syms):
                try:
                    val = float(df["Close"].dropna().iloc[-1]) if len(fetch_tickers) == 1 \
                          else float(df[sym]["Close"].dropna().iloc[-1])
                    if val > 0:
                        price_map[t] = val
                        fresh_prices[t] = round(val, 2)
                except Exception:
                    pass
        except Exception:
            pass

        # Live Prices touches only prices — ranks require a full Nifty 200 scan
        # which lives behind the Live Signals refresh path, not this endpoint.
        IST = _tz(_td(hours=5, minutes=30))
        prices_updated_at = _dt.now(IST).isoformat(timespec="seconds")
        try:
            os.makedirs(os.path.dirname(lp_path), exist_ok=True)
            with open(lp_path, "w") as f:
                json.dump({"updated_at": prices_updated_at,
                           "prices": {k: round(v, 4) for k, v in price_map.items()},
                           "ranks":  {k: v for k, v in rank_map.items() if v is not None}},
                          f, indent=2)
        except Exception:
            pass

    holdings = []
    total_entry = 0.0
    total_current = 0.0
    earliest_date = None

    # Resolve sector rank for any held ETFs so the Tracker's Rank column
    # can render as 'ETF#<n>' (matching the basket Holds/Entries display).
    from data.sector_etf_map import KNOWN_ETF_SYMBOLS, ETF_TO_SECTOR
    has_etfs = any(item.get("ticker") in KNOWN_ETF_SYMBOLS for item in basket)
    sector_rank_by_name = {}
    if has_etfs:
        try:
            for r in (_get_sector_ranking() or []):
                sector_rank_by_name[r["symbol"]] = r["rank"]
        except Exception:
            pass

    for h in basket:
        t = h["ticker"]
        ep = h.get("entry_price", 0)
        qty = h.get("qty", 0)
        cp = price_map.get(t, ep)
        entry_val = round(ep * qty, 2)
        curr_val  = round(cp * qty, 2)
        ret_pct   = round((cp - ep) / ep * 100, 2) if ep > 0 else 0
        total_entry   += entry_val
        total_current += curr_val

        edate = h.get("entry_date", "")[:10]
        if edate and (earliest_date is None or edate < earliest_date):
            earliest_date = edate

        if t in KNOWN_ETF_SYMBOLS:
            sec = ETF_TO_SECTOR.get(t)
            sec_rank = sector_rank_by_name.get(sec)
            row_rank = f"ETF#{sec_rank}" if sec_rank else "ETF"
        else:
            row_rank = rank_map.get(t)

        # rank_delta: positive = improved (rank number fell), negative = worsened
        curr_rank_num = row_rank if isinstance(row_rank, (int, float)) else None
        prev_rank_num = prev_ranks.get(t)
        rank_delta = (prev_rank_num - curr_rank_num
                      if prev_rank_num is not None and curr_rank_num is not None
                      else None)
        holdings.append({
            "ticker":        t,
            "rank":          row_rank,
            "rank_delta":    rank_delta,
            "qty":           qty,
            "entry_price":   round(ep, 2),
            "current_price": round(cp, 2),
            "entry_date":    edate,
            "entry_value":   entry_val,
            "current_value": curr_val,
            "return_pct":    ret_pct,
            "note":          h.get("note", ""),
        })

    holdings.sort(key=lambda x: x["return_pct"], reverse=True)
    unrealized_pnl = round(total_current - total_entry, 2)

    # Realized P&L + total invested capital — single history read.
    # total_invested = sum of net capital injected across all rebalances:
    #   each rebalance contributes max(0, buy_total − sell_total).
    #   Pure rebalances contribute 0 (sells fund buys).
    #   Capital additions (e.g. top-up buys with no sells) contribute their full buy amount.
    realized_pnl    = 0.0
    initial_capital = 0.0
    try:
        with open(mom20_history_path(user_id)) as _hf:
            _hist = json.load(_hf)
        realized_pnl = _retrospective_realized_pnl(_hist)
        for rb in _hist:
            buy_tot  = sum(t.get("qty", 0) * t.get("price", 0) for t in rb.get("buys",  []))
            sell_tot = sum(t.get("qty", 0) * t.get("price", 0) for t in rb.get("sells", []))
            net = buy_tot - sell_tot
            if net > 0:
                initial_capital += net
    except Exception:
        pass
    realized_pnl    = round(realized_pnl, 2)
    initial_capital = round(initial_capital, 2) or round(total_entry, 2)

    total_pnl        = round(unrealized_pnl + realized_pnl, 2)
    total_return_pct = round(total_pnl / initial_capital * 100, 2) if initial_capital else 0

    return jsonify({
        "success":            True,
        "holdings":           holdings,
        "fresh_prices":       fresh_prices,
        "initial_capital":    initial_capital,       # Rebalance #01 total deployed
        "total_entry_value":  round(total_entry, 2), # current cost basis (internal)
        "total_current_value": round(total_current, 2),
        "unrealized_pnl":     unrealized_pnl,
        "realized_pnl":       realized_pnl,
        "total_pnl":          total_pnl,
        "total_return_pct":   total_return_pct,      # total P&L / initial capital
        "tracking_since":     earliest_date,
        "prices_updated_at":  prices_updated_at,
    })


# ── Mom20 per-position note ───────────────────────────────────────────────────

@app.route("/api/portfolio-users/<user_id>/mom20-position-note", methods=["POST"])
def api_mom20_position_note(user_id):
    """Save (or clear) a free-text note on a Mom20 basket holding."""
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    note   = (body.get("note") or "").strip()
    if not ticker:
        return jsonify({"success": False, "error": "ticker required"})
    pf_path = mom20_portfolio_path(user_id)
    try:
        with open(pf_path) as f:
            pf = json.load(f)
    except Exception:
        return jsonify({"success": False, "error": "portfolio not found"})
    matched = False
    for h in pf.get("basket", []):
        if h.get("ticker") == ticker:
            if note:
                h["note"] = note
            else:
                h.pop("note", None)
            matched = True
            break
    if not matched:
        return jsonify({"success": False, "error": f"{ticker} not in basket"})
    with open(pf_path, "w") as f:
        json.dump(pf, f, indent=2)
    return jsonify({"success": True})




# ── AI Portfolio Briefing (Claude) ────────────────────────────────────────────

_AI_BRIEFING_CACHE: dict = {}          # cache_key → {ts, text}
_AI_BRIEFING_TTL   = 86400             # 24h — only manual Regenerate busts this
_AI_BRIEFING_DISK  = os.path.join(DATA_STORE_PATH, "ai_briefing_cache.json")

# Load persisted cache from disk so it survives server restarts
try:
    with open(_AI_BRIEFING_DISK) as _bf:
        _AI_BRIEFING_CACHE.update(json.load(_bf))
except Exception:
    pass

def _build_briefing_prompt(holdings, sector_ranking, today_str,
                           live_signals=None, sector_momentum=None,
                           mom20_regime='?', nifty_regime='?', sector_map=None,
                           section=None):
    held_tickers = {h['ticker'] for h in holdings}
    held_sectors  = {h.get('sector', '') for h in holdings if h.get('sector')}
    _smap = sector_map or {}

    # Build beta lookup from live signals for portfolio holdings
    _beta_lookup = {}
    for s in (live_signals or {}).get('mom20_signals', []):
        _beta_lookup[s.get('ticker', '')] = s.get('beta', 0)
    for s in (live_signals or {}).get('mom20_overflow', []):
        _beta_lookup[s.get('ticker', '')] = s.get('beta', 0)

    lines = [
        "You are a senior Indian equity analyst providing a daily market intelligence briefing.",
        f"Today: {today_str}. Mom20 Regime: {mom20_regime}. Nifty Regime: {nifty_regime}.",
        "Regime flags are INFORMATIONAL. Entries remain permitted unless explicitly stated otherwise.",
        "",
        "## STRATEGY CONTEXT",
        "Mom20 = Nifty200 momentum strategy: top 20 by Z-score (50% 12m + 50% 3m).",
        "Beta cap ≤ 1.2 (no minimum). Beta bands:",
        "  β > 1.2   → OVERFLOW: excluded from main basket; ETF proxy may be used as substitute",
        "  β 1.0–1.2 → High: amplifies sector moves; caution if sector RS weakening",
        "  β 0.6–1.0 → Moderate: preferred profile",
        "  β < 0.6   → Defensive: may lag bull markets",
        "  β < 0.5   → Ultra-low: positive quality trait, NOT a risk flag",
        "",
        "## SECTOR → ETF PROXY TABLE",
        "Use this when overflow stocks or momentum sectors have no eligible direct stock.",
        "Sector | ETF Symbol | ETF Name",
        "NIFTY INDIA DEFENCE     | MODEFENCE  | Motilal Oswal Nifty India Defence ETF",
        "NIFTY METAL             | METALIETF  | Nippon Nifty Metal ETF",
        "NIFTY HEALTHCARE        | HEALTHIETF | Nippon Nifty Healthcare ETF",
        "NIFTY IT                | ITBEES     | Nippon Nifty IT ETF",
        "NIFTY AUTO              | AUTOBEES   | Nippon Nifty Auto ETF",
        "NIFTY FMCG              | FMCGIETF   | ICICI Pru Nifty FMCG ETF",
        "NIFTY PSU BANK          | PSUBNKBEES | Nippon Nifty PSU Bank ETF",
        "NIFTY BANK              | BANKBEES   | Nippon Nifty Bank ETF",
        "NIFTY INFRA             | INFRABEES  | Nippon Nifty Infra ETF",
        "NIFTY OIL & GAS         | OILIETF    | ICICI Pru Nifty Oil & Gas ETF",
        "NIFTY REALTY            | MOREALTY   | Motilal Oswal Realty ETF",
        "NIFTY INDIA MFG         | MAKEINDIA  | Mirae Asset Make-in-India ETF",
        "NIFTY ENERGY            | —          | No tradeable NSE ETF available",
        "",
    ]

    # 1. Portfolio Holdings (with beta)
    lines += ["## PORTFOLIO HOLDINGS (all users)"]
    if holdings:
        lines.append("User | Ticker | Sector | Rank | Entry₹ | Now₹ | P&L% | β")
        for h in sorted(holdings, key=lambda x: (x.get('rank') or 9999) if isinstance(x.get('rank'), (int, float)) else 9999):
            tk   = h['ticker']
            beta = _beta_lookup.get(tk, 0)
            beta_flag = ' ⚡' if beta > 1.0 else ''
            lines.append(
                f"{h.get('user','?')} | {tk} | {h.get('sector','?')} | "
                f"{h.get('rank','?')} | "
                f"₹{h.get('entry_price',0):.0f} | ₹{h.get('current_price',0):.0f} | "
                f"{h.get('return_pct',0):+.1f}% | β{beta:.2f}{beta_flag}"
            )
    else:
        lines.append("(no holdings)")

    # 2. Mom20 Live Ranking
    signals = (live_signals or {}).get('mom20_signals', [])
    lines += ["", "## MOM20 LIVE RANKING (top 40; rank ≤ 15 = buffer-in entry zone)"]
    if signals:
        lines.append("Rank | Ticker | Sector | Status | 12m% | 3m% | Score | Beta")
        for s in signals[:40]:
            tk   = s.get('ticker', '')
            sec  = _smap.get(tk, s.get('sector', '?'))
            held = 'HELD' if tk in held_tickers else 'NEW'
            lines.append(
                f"#{s.get('rank','?')} | {tk} | {sec} | {held} | "
                f"{s.get('ret_12m',0):+.1f}% | {s.get('ret_3m',0):+.1f}% | "
                f"{s.get('momentum_score',0):.3f} | β{s.get('beta',0):.2f}"
            )
    else:
        lines.append("(signals not available)")

    # 3. Overflow basket (with sector)
    overflow = (live_signals or {}).get('mom20_overflow', [])
    if overflow:
        lines += ["", "## OVERFLOW BASKET (β > 1.2 — excluded from main basket; ETF proxy may substitute)"]
        lines.append("Rank | Ticker | Sector | 12m% | 3m% | Beta")
        for s in overflow[:15]:
            tk  = s.get('ticker', '')
            sec = _smap.get(tk, s.get('sector', '?'))
            lines.append(
                f"#{s.get('rank','?')} | {tk} | {sec} | "
                f"{s.get('ret_12m',0):+.1f}% | {s.get('ret_3m',0):+.1f}% | β{s.get('beta',0):.2f}"
            )
    else:
        lines += ["", "## OVERFLOW BASKET", "(empty — no β > 1.2 stocks in top rankings)"]

    # 4. Sector Z-Score Ranking
    scores_valid = any((r.get('score') or 0) != 0 for r in (sector_ranking or []))
    lines += ["", "## SECTOR Z-SCORE RANKING (top 15)"]
    if not scores_valid:
        lines.append("⚠️ Live sector scores unavailable this cycle.")
    else:
        lines.append("Rank | Sector | Score | 12m% | 3m%")
        for r in (sector_ranking or [])[:15]:
            star = ' ★' if r.get('symbol', '') in held_sectors else ''
            lines.append(
                f"#{r.get('rank','?')} | {r.get('symbol','?')}{star} | "
                f"{r.get('score',0):.3f} | {r.get('ret_12m',0):+.1f}% | {r.get('ret_3m',0):+.1f}%"
            )

    # 5. Sector RS Momentum
    sm = sector_momentum or {}
    lines += ["", "## SECTOR RS MOMENTUM (vs Nifty200)"]
    if sm:
        lines.append("Sector | Composite | 5d Δ | 10d Δ")
        items = sorted(sm.items(), key=lambda x: x[1].get('momentum', 0), reverse=True)
        for sec, v in items:
            star = ' ★' if sec in held_sectors else ''
            lines.append(
                f"{sec}{star} | {v.get('momentum',0):+.1f} | "
                f"{v.get('delta_5d',0):+.2f} | {v.get('delta_10d',0):+.2f}"
            )
    else:
        lines.append("(not available)")

    # ── Section task definitions ───────────────────────────────────────────────
    _TASKS = {
        'risk': [
            "### ⚠️ Portfolio Warnings & Risk Flags",
            "Go through each holding. Flag: rank > 40 (exit signal), rank 21–40 (buffer zone — watch), "
            "P&L < −10%, stocks in sectors with negative RS momentum. "
            "ALSO compute holdings per sector as % of total positions; flag any sector > 25% as "
            "concentration risk, > 40% as severe; escalate if the dominant sector's RS 5d Δ is negative. "
            "For each flag: user, ticker, rank, what specifically to watch. "
            "HIGH-BETA HOLDINGS (β > 1.0, marked ⚡ in holdings table): for each, state whether the stock's "
            "sector RS is rising, stalling, or falling — high-beta + falling sector RS is an elevated risk.",
        ],
        'entries': [
            "### 🆕 New Entry Candidates",
            "Top 5 non-held stocks at rank ≤ 20. For each candidate, address in prose: "
            "(i) sector direction in words (rising/stalling/falling) and its Z-score rank — no raw delta numbers, "
            "(ii) impact on portfolio sector concentration — state current sector weight %, "
            "(iii) beta-regime fit, "
            "(iv) 3m-vs-12m momentum balance (sustained or decelerating?), "
            "(v) any pending news or corporate event from the feed. "
            "End each candidate with a bolded verdict: **STRONG ADD / MODERATE ADD / WAIT / SKIP** "
            "+ one-line rationale.",
        ],
        'overflow': [
            "### 🔀 Overflow & High-Beta Signals",
            "Analyse the overflow basket (β > 1.2 stocks excluded from main basket). "
            "Group overflow stocks by sector. For each sector group:",
            "  (i)  Name the overflow stocks, their ranks and 12m/3m momentum.",
            "  (ii) State the sector's RS direction (rising/stalling/falling from the momentum table).",
            "  (iii) Check the SECTOR → ETF PROXY TABLE. If an ETF exists for the sector AND the sector "
            "        is in the Rising group: recommend adding the ETF as a proxy, naming the ETF symbol "
            "        and rationale (captures sector momentum without high-beta single-stock risk). "
            "        If no ETF exists: state this explicitly — do not invent a proxy.",
            "  (iv)  If the sector is Neutral or Falling: state that the ETF proxy is NOT recommended "
            "        despite overflow stock momentum — sector tailwind is absent.",
            "For already-held ETF proxies (e.g. MODEFENCE, HEALTHIETF in portfolio): "
            "confirm whether the underlying sector is still Rising; flag if it has deteriorated. "
            "Close with a one-line summary of which overflow sectors are actionable vs. which to ignore.",
        ],
        'sectors': [
            "### 📈 Sectors — Deep Dive",
            "Output a single markdown table with these exact columns (no prose, no bullets, no extra sections): "
            "| Sector | Signal | Entry | Z# | Ret 12m | Ret 3m | Holdings (★ = rank ≤15, entry zone) |",
            "",
            "SIGNAL RULES (derive from composite RS and deltas — do NOT print the raw numbers): "
            "🟢 Accel → composite>0, 5dΔ>0, 5dΔ>10dΔ; "
            "🟡 Steady → composite>0, 5dΔ>0, 5dΔ≤10dΔ; "
            "🔵 Recovering → composite>0, 5dΔ<0, 5dΔ>10dΔ (decline slowing); "
            "🟠 Fading → composite>0, 5dΔ<0, 5dΔ≤10dΔ; "
            "🔴 Falling → composite≤0.",
            "",
            "ENTRY COLUMN: only fill when the sector has at least one stock at rank ≤15 in the Mom40 holdings column. "
            "Rules: sector 🟢/🟡 + stock★ → ✅ Buy; sector 🔵 + stock★ → 🔵 Watch; "
            "sector 🟠 + stock★ → ⚠️ Caution; sector 🔴 + stock★ → ❌ Avoid. Leave blank otherwise.",
            "",
            "HOLDINGS COLUMN: use the MOM20 LIVE RANKING table above — it has a Sector column. "
            "Group ALL 40 stocks by their sector; list every stock for that sector as TICKER#rank. "
            "Append ★ to any stock with rank ≤ 15. Do NOT skip any stock. Leave — only if truly no stocks from that sector appear in the top 40.",
            "",
            "SECTOR COLUMN: use short names (drop 'NIFTY '). Sort rows by Z# ascending (sectors without Z# go at bottom). "
            "Include ALL sectors — both those with RS data and those with Z-score only.",
            "",
            "After the table, add a 3-line 'Key Reads' block: one line each for the most important ✅ Buy, "
            "⚠️ Caution, and ❌ Avoid from the table. "
            "STRICT: output ONLY the markdown table + those 3 Key Reads lines. "
            "NO derivation notes. NO signal explanations. NO revised rows. NO extra prose. "
            "Do NOT show classification reasoning. Do NOT repeat or revise any row. Stop after the 3rd Key Read.",
        ],
        'watchlist': [
            "### 💡 Strategic Watch List",
            "What to add/exit at next rebalance. Any emerging theme (defence rally, IT selloff, FMCG recovery). "
            "Close with a mandatory **Don't-Do** line: 1–3 candidates that look attractive on raw rank but "
            "should be passed over, naming the failed dimension "
            "(concentration / extension / event risk / sector RS deterioration).",
        ],
    }
    _SECTION_ORDER = ['risk', 'entries', 'overflow', 'sectors', 'watchlist']
    _FORMAT_RULES  = [
        "",
        "---",
        "## FORMATTING RULES — follow exactly:",
        "",
        "1. **Header hierarchy**",
        "   - Top-level sections use `## ` with emoji (e.g. `## ⚠️ Portfolio Warnings & Risk Flags`)",
        "   - Sub-sections use `### ` (e.g. `### Tier 1 — Exit Candidates`, `### Buffer Zone`)",
        "   - Inline labels within paragraphs use **bold** (e.g. **Rank > 40 (Exit Signal):**)",
        "",
        "2. **New entry candidate cards**",
        "   - Each candidate gets its own `### N. TICKER (Rank #X | β X.XX | SECTOR)` header",
        "   - Verdict on its own line, bolded: `**Verdict: STRONG ADD** — one-line rationale.`",
        "",
        "3. **Numerical data in inline code**",
        "   - Wrap all metrics in backticks: RS composite, 5d Δ, 10d Δ, P&L%, ranks, β values, scores",
        "",
        "4. **Emojis reserved for top-level section headers and ⚠️ risk flags only**",
        "   - No emojis inside body prose, bullets, sub-bullets, or theme labels",
    ]

    lines += ["", "---"]
    if section and section in _TASKS:
        lines += ["## YOUR TASK — write ONLY this section (no preamble, no other sections):", ""]
        lines += _TASKS[section]
        lines += ["", "Write in professional analyst style. Every claim must reference a data point from the tables above."]
    else:
        lines += ["## YOUR TASK — write a market intelligence report with these sections:", ""]
        for key in _SECTION_ORDER:
            lines += _TASKS[key]
            lines.append("")
        lines += [
            "Write in professional analyst style. Use bullet points within sections. "
            "Aim for ~1000–1100 words. Every claim must reference a data point from above.",
        ]
    lines += _FORMAT_RULES
    return "\n".join(lines)


@app.route("/api/ai-briefing/stream")
def api_ai_briefing_stream():
    """Stream a Claude-generated portfolio briefing via SSE."""
    from flask import Response, stream_with_context
    from datetime import datetime as _dtm2, timezone as _tz2, timedelta

    force     = request.args.get('refresh') == '1'
    section   = request.args.get('section') or None   # e.g. 'risk', 'entries', 'sectors'…
    cache_key = section or 'all'

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        def _err():
            yield "data: ANTHROPIC_API_KEY not set in environment.\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(_err()), mimetype='text/event-stream')

    # ── Load live signals cache FIRST — freshest rank source ────────────────
    # Ranks come from here (updated by Live Signals refresh), NOT from the
    # per-user live_prices file (which only updates when "Live Prices" is clicked).
    live_signals    = {}
    sector_momentum = {}
    mom20_regime    = '?'
    nifty_regime    = '?'
    ls_updated_at   = 0.0
    fresh_ranks     = {}   # ticker → rank, from live signals (freshest source)
    try:
        from config.settings import LIVE_SIGNALS_CACHE_FILE
        with open(LIVE_SIGNALS_CACHE_FILE) as f:
            ls_cache = json.load(f)
        live_signals    = ls_cache
        sector_momentum = ls_cache.get('sector_momentum', {})
        mom20_regime    = ls_cache.get('mom20_regime', '?') or '?'
        nifty_regime    = ls_cache.get('nifty_regime', '?') or '?'
        fresh_ranks     = ls_cache.get('mom20_unfiltered_ranks', {}) or {}
        # Parse updated_at to a UNIX timestamp for cache-busting comparison
        _ua = ls_cache.get('updated_at') or ls_cache.get('timestamp') or ''
        if _ua:
            try:
                ls_updated_at = _dtm2.fromisoformat(_ua.replace('Z', '+00:00')).timestamp()
            except Exception:
                pass
    except Exception:
        pass

    # Serve from cache; only force=True (Regenerate button) bypasses it
    cached = _AI_BRIEFING_CACHE.get(cache_key)
    if not force and cached:
        age = (_dtm2.now().timestamp() - cached['ts'])
        if age < _AI_BRIEFING_TTL:
            def _cached_stream():
                import time as _t
                ist = _tz2(timedelta(hours=5, minutes=30))
                gen_time = _dtm2.fromtimestamp(cached['ts'], tz=ist).strftime('%d %b %H:%M')
                yield f"data: [CACHED:{gen_time}]\n\n"
                for chunk in (cached['text'][i:i+80] for i in range(0, len(cached['text']), 80)):
                    yield f"data: {json.dumps(chunk)}\n\n"
                    _t.sleep(0.005)
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(_cached_stream()), mimetype='text/event-stream')

    # ── Gather context — ALL users' portfolios ──────────────────────────────
    from data.sector_etf_map import ETF_TO_SECTOR
    sector_map = {}
    try:
        sector_map = _load_sector_map() or {}
    except Exception:
        pass

    all_users   = load_users()
    holdings    = []

    for u in all_users:
        uid   = u.get('id') or u.get('config_user_id', '')
        uname = u.get('name', uid)
        try:
            with open(mom20_portfolio_path(uid)) as f:
                pf = json.load(f)
            basket = pf.get('basket', [])
            if not basket:
                continue
            price_map = {}
            try:
                with open(mom20_live_prices_path(uid)) as f:
                    lp = json.load(f)
                price_map = {k: float(v) for k, v in (lp.get('prices') or {}).items()}
            except Exception:
                pass
            for h in basket:
                t  = h['ticker']
                ep = h.get('entry_price', 0)
                cp = price_map.get(t, ep)
                # Prefer fresh_ranks (from live signals, just refreshed by user)
                # over rank_map from live_prices file (only updated on "Live Prices" click)
                rank = fresh_ranks.get(t) or '?'
                holdings.append({
                    'user':          uname,
                    'ticker':        t,
                    'sector':        sector_map.get(t) or ETF_TO_SECTOR.get(t, ''),
                    'rank':          rank,
                    'entry_price':   ep,
                    'entry_date':    h.get('entry_date', '?'),
                    'qty':           h.get('qty', '?'),
                    'current_price': cp,
                    'return_pct':    round((cp - ep) / ep * 100, 1) if ep else 0,
                })
        except Exception as e:
            print(f"[ai-briefing] portfolio load error ({uid}): {e}")

    # Force-refresh sector ranking for the briefing (don't rely on stale cache)
    sector_ranking = []
    try:
        from data.score_live_sectors import score_live_sectors as _score_sectors
        sector_ranking = _score_sectors() or []
    except Exception:
        try:
            sector_ranking = _get_sector_ranking() or []
        except Exception:
            pass

    ist = _tz2(timedelta(hours=5, minutes=30))
    today_str = _dtm2.now(ist).strftime('%d %b %Y')

    user_msg = _build_briefing_prompt(
        holdings, sector_ranking, today_str,
        live_signals=live_signals, sector_momentum=sector_momentum,
        mom20_regime=mom20_regime, nifty_regime=nifty_regime,
        sector_map=sector_map,
        section=section,
    )

    # ── Stream Claude response ──────────────────────────────────────────────
    def generate():
        import anthropic as _ant
        client = _ant.Anthropic(api_key=api_key)
        full_text = []
        try:
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=8000,
                temperature=0.2,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_text.append(text_chunk)
                    yield f"data: {json.dumps(text_chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'Error: {e}')}\n\n"
        yield "data: [DONE]\n\n"
        # Cache the full response (memory + disk)
        _AI_BRIEFING_CACHE[cache_key] = {
            'ts':   _dtm2.now().timestamp(),
            'text': ''.join(full_text),
        }
        try:
            with open(_AI_BRIEFING_DISK, 'w') as _bf:
                json.dump(_AI_BRIEFING_CACHE, _bf)
        except Exception:
            pass

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Mom20 portfolio chart ──────────────────────────────────────────────────────

@app.route("/api/portfolio-users/<user_id>/mom20-chart", methods=["GET"])
def api_mom20_chart(user_id):
    """
    Daily cumulative return: portfolio vs Nifty 500, from first rebalance date.
    Uses rebalance history to reconstruct the actual holdings on each date so the
    chart's final point matches the RETURNS tile exactly.
    """
    import yfinance as yf
    import pandas as pd

    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})

    # ── Load history to build an accurate holdings timeline ──────────────────
    _hist = []
    try:
        with open(mom20_history_path(user_id)) as _hf:
            _hist = json.load(_hf)
        _retrospective_realized_pnl(_hist)
    except Exception:
        pass

    # Load current basket as fallback when there is no history
    basket = []
    try:
        with open(mom20_portfolio_path(user_id)) as f:
            basket = json.load(f).get("basket", [])
    except Exception:
        pass

    if not _hist and not basket:
        return jsonify({"success": False, "error": "no portfolio"})

    # Build a holdings timeline: list of (effective_from_date, {ticker: {qty, entry_price}})
    # Each entry represents post-rebalance holdings from that date onward.
    holdings_timeline = []
    initial_capital   = 0.0

    if _hist:
        current_h = {}
        for rb in _hist:
            rb_date = (rb.get("rebalance_date") or "")[:10]
            if not rb_date:
                continue
            # Apply sells (remove positions)
            for s in rb.get("sells", []):
                current_h.pop(s.get("ticker"), None)
            # Apply buys (add or accumulate)
            for b in rb.get("buys", []):
                t     = b.get("ticker")
                qty   = b.get("qty", 0)
                price = b.get("price", 0)
                if not t or qty <= 0:
                    continue
                if t in current_h:
                    old_qty   = current_h[t]["qty"]
                    old_price = current_h[t]["entry_price"]
                    new_qty   = old_qty + qty
                    wavg      = (old_qty * old_price + qty * price) / new_qty
                    current_h[t] = {"qty": new_qty, "entry_price": round(wavg, 2)}
                else:
                    current_h[t] = {"qty": qty, "entry_price": round(price, 2)}
            holdings_timeline.append((rb_date, dict(current_h)))

        # Match header formula: net new capital injected across ALL rebalances.
        # Pure rebalances (sells fund buys) contribute 0; top-ups contribute their net.
        initial_capital = 0.0
        for rb in _hist:
            buy_tot  = sum(t.get("qty", 0) * t.get("price", 0) for t in rb.get("buys",  []))
            sell_tot = sum(t.get("qty", 0) * t.get("price", 0) for t in rb.get("sells", []))
            net = buy_tot - sell_tot
            if net > 0:
                initial_capital += net

    # Fallback: seed the timeline from current basket
    if not holdings_timeline and basket:
        dates_list = [h.get("entry_date", "")[:10] for h in basket if h.get("entry_date")]
        if not dates_list:
            return jsonify({"success": False, "error": "no entry dates"})
        seed_date = min(dates_list)
        seed_h    = {h["ticker"]: {"qty": h.get("qty", 0),
                                    "entry_price": h.get("entry_price", 0)}
                     for h in basket}
        holdings_timeline = [(seed_date, seed_h)]
        initial_capital   = sum(h.get("entry_price", 0) * h.get("qty", 0) for h in basket)

    if not holdings_timeline or initial_capital <= 0:
        return jsonify({"success": False, "error": "insufficient history data"})

    start_date = holdings_timeline[0][0]

    # ── Collect all tickers ever held and download prices ────────────────────
    all_tickers = sorted({t for _, h in holdings_timeline for t in h})
    bench_syms  = ["^CRSLDX", "^CNX200"]
    yf_syms     = [f"{t}.NS" for t in all_tickers] + bench_syms

    try:
        raw = yf.download(yf_syms, start=start_date, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    def get_close(sym):
        try:
            if len(yf_syms) == 1:
                return raw["Close"].dropna()
            return raw[sym]["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)

    bench_n500 = get_close("^CRSLDX")
    bench_n200 = get_close("^CNX200")
    bench_raw  = bench_n500 if not bench_n500.empty else bench_n200
    if bench_raw.empty:
        return jsonify({"success": False, "error": "benchmark data unavailable"})

    ticker_closes = {}
    for t, sym in zip(all_tickers, yf_syms[:-2]):
        s = get_close(sym)
        if not s.empty:
            ticker_closes[t] = s

    # ── Compute portfolio value on each trading day ───────────────────────────
    all_dates = bench_raw.index

    def get_holdings_for_date(dt_str):
        """Return the active post-rebalance holdings for dt_str (sorted ascending by date)."""
        active = holdings_timeline[0][1]
        for rb_date, h in holdings_timeline:
            if dt_str >= rb_date:
                active = h
            else:
                break
        return active

    port_values = []
    for dt in all_dates:
        dt_str = dt.strftime("%Y-%m-%d")
        holdings = get_holdings_for_date(dt_str)
        val = 0.0
        for t, h in holdings.items():
            qty = h.get("qty", 0)
            ep  = h.get("entry_price", 0)
            if qty <= 0:
                continue
            s = ticker_closes.get(t)
            if s is not None and dt in s.index:
                val += qty * float(s.loc[dt])
            else:
                val += qty * ep     # fallback to cost when price unavailable
        port_values.append(val)

    if not port_values:
        return jsonify({"success": False, "error": "could not compute portfolio values"})

    # ── Today bar: replace with live prices so chart tip matches RETURNS tile ─
    import datetime as _dt
    today_str    = _dt.date.today().isoformat()
    last_yf_date = all_dates[-1].strftime("%Y-%m-%d") if len(all_dates) else ""
    try:
        live_prices = {}
        try:
            with open(mom20_live_prices_path(user_id)) as f:
                live_prices = {k: float(v)
                               for k, v in (json.load(f).get("prices") or {}).items()}
        except Exception:
            pass

        if live_prices:
            latest_h  = holdings_timeline[-1][1]
            today_val = sum(
                h.get("qty", 0) * live_prices.get(t, h.get("entry_price", 0))
                for t, h in latest_h.items()
                if h.get("qty", 0) > 0
            )
            if today_val > 0:
                if last_yf_date == today_str:
                    port_values[-1] = today_val
                elif last_yf_date < today_str:
                    port_values.append(today_val)
                    all_dates = list(all_dates) + [pd.Timestamp(today_str)]
    except Exception:
        pass

    # ── Build realized P&L and per-period cost-basis helpers ─────────────────
    # Correct formula: return = (unrealized + cumulative_realized) / initial_capital
    # where unrealized = portfolio_value - cost_basis_of_active_holdings
    hist_realized_by_date = {}
    if _hist:
        running = 0.0
        for rb in _hist:
            running += sum(s.get("pnl", 0) for s in rb.get("sells", []))
            hist_realized_by_date[(rb.get("rebalance_date") or "")[:10]] = round(running, 2)

    sorted_realized = sorted(hist_realized_by_date.items())

    def _realized_offset(dt_str):
        offset = 0.0
        for rd, pnl in sorted_realized:
            if rd <= dt_str:
                offset = pnl
            else:
                break
        return offset

    # Cost basis is constant within each rebalance period
    period_cost_bases = [
        (rb_date, sum(h["qty"] * h["entry_price"] for h in snapshot.values()))
        for rb_date, snapshot in holdings_timeline
    ]

    def _cost_basis_for_date(dt_str):
        cb = period_cost_bases[0][1]
        for rb_date, cost in period_cost_bases:
            if dt_str >= rb_date:
                cb = cost
            else:
                break
        return cb

    # ── Compute return series ─────────────────────────────────────────────────
    base_port = initial_capital
    base_n500 = float(bench_n500.iloc[0]) if not bench_n500.empty else None
    base_n200 = float(bench_n200.iloc[0]) if not bench_n200.empty else None
    dates_out = [d.strftime("%Y-%m-%d") for d in all_dates]

    port_pct = [
        round(
            (v - _cost_basis_for_date(d.strftime("%Y-%m-%d"))
             + _realized_offset(d.strftime("%Y-%m-%d"))) / base_port * 100,
            3,
        )
        for v, d in zip(port_values, all_dates)
    ]

    # Force last point to exactly match the RETURNS tile:
    # (basket × live_prices − cost_basis + total_realized) / initial_capital
    if basket and port_pct:
        try:
            cost_basis_now  = sum(h.get("qty", 0) * h.get("entry_price", 0) for h in basket)
            lp              = live_prices if live_prices else {}
            current_val_now = sum(
                h.get("qty", 0) * lp.get(h["ticker"], h.get("entry_price", 0))
                for h in basket if h.get("qty", 0) > 0
            )
            realized_now  = sum(s.get("pnl", 0) for rb in _hist for s in rb.get("sells", []))
            port_pct[-1]  = round(
                (current_val_now - cost_basis_now + realized_now) / base_port * 100, 3
            )
        except Exception:
            pass

    def bench_series(raw, base):
        if raw.empty or base is None:
            return [None] * len(all_dates)
        out = []
        last_val = None
        for d in all_dates:
            try:
                v = round((float(raw.loc[d]) / base - 1) * 100, 3)
                last_val = v
                out.append(v)
            except Exception:
                out.append(last_val)
        return out

    n500_pct = bench_series(bench_n500, base_n500)
    n200_pct = bench_series(bench_n200, base_n200)

    return jsonify({
        "success":      True,
        "dates":        dates_out,
        "portfolio":    port_pct,
        "n500":         n500_pct,
        "n200":         n200_pct,
        "start_date":   start_date,
        "total_return": port_pct[-1] if port_pct else None,
        "n500_return":  n500_pct[-1] if n500_pct else None,
        "n200_return":  n200_pct[-1] if n200_pct else None,
    })


# ── TechMo universe + scanner ─────────────────────────────────────────────────

TECHMO_FILE = os.path.join(DATA_STORE_PATH, "techmo_portfolio.json")

# Combined Nasdaq-100 + AI Universe, cluster-mapped
TECHMO_UNIVERSE = {
    # Compute Silicon
    "NVDA":"Compute","AMD":"Compute","AVGO":"Compute","INTC":"Compute",
    "ARM":"Compute",
    # Memory & Storage
    "MU":"Memory","WDC":"Memory","SNDK":"Memory","STX":"Memory","NTAP":"Memory",
    # Semicon Equipment (incl. EDA)
    "ASML":"Semicon Equip","LRCX":"Semicon Equip","KLAC":"Semicon Equip",
    "KEYS":"Semicon Equip","AMAT":"Semicon Equip","SNPS":"Semicon Equip",
    # Packaging (OSAT)
    "TSM":"Foundry","ASX":"Packaging","AMKR":"Packaging","CAMT":"Packaging",
    # Photonics / Optical components
    "COHR":"Photonics","LITE":"Photonics","GLW":"Photonics","FN":"Photonics","AAOI":"Photonics",
    # Networking & Connectivity
    "ANET":"Networking","CSCO":"Networking","CRDO":"Networking","APH":"Networking",
    "MRVL":"Networking","NOK":"Networking","CIEN":"Networking",
    # Server OEMs
    "SMCI":"Server OEM","DELL":"Server OEM","HPE":"Server OEM",
    "JBL":"Server OEM","FLEX":"Server OEM",
    # AI Neoclouds
    "CRWV":"Neocloud","NBIS":"Neocloud","IREN":"Neocloud","APLD":"Neocloud",
    "WULF":"Neocloud","CORZ":"Neocloud","CIFR":"Neocloud",
    # Power & Cooling
    "VRT":"Power/Cool","ETN":"Power/Cool",
    "PWR":"Power/Cool","HUBB":"Power/Cool","MOD":"Power/Cool",
    # Energy / AI Power
    "CEG":"AI Energy","VST":"AI Energy","NEE":"AI Energy","SMR":"AI Energy",
    "OKLO":"AI Energy","EOSE":"AI Energy","EQT":"AI Energy","GEV":"AI Energy","BE":"AI Energy",
    # Power Electronics
    "STM":"Power Elec","ADI":"Power Elec","MPWR":"Power Elec",
    "NVTS":"Power Elec","ON":"Power Elec","TXN":"Power Elec",
    # Robotics & Autonomy
    "TSLA":"Robotics","PATH":"Robotics","SYM":"Robotics",
    "SERV":"Robotics","TER":"Robotics","ISRG":"Robotics",
    # Defense & Drones
    "KTOS":"Defense","AVAV":"Defense","ONDS":"Defense","RCAT":"Defense",
    "OSIS":"Defense","LMT":"Defense","NOC":"Defense",
    # Space & Satellites
    "ASTS":"Space","RKLB":"Space","LUNR":"Space","PL":"Space",
    "BKSY":"Space","IRDM":"Space",
    # Materials
    "MP":"Materials","UUUU":"Materials","FCX":"Materials",
    "AA":"Materials","TECK":"Materials",
    # Frontier AI Models
    "MSFT":"Frontier AI","GOOGL":"Frontier AI","META":"Frontier AI","AMZN":"Frontier AI",
    # QQQ-only additions
    "AAPL":"Consumer Tech","NFLX":"Consumer Tech","QCOM":"Wireless",
    "COST":"Retail","ORLY":"Retail","ADP":"Enterprise SW",
}

_TECHMO_SCAN_CACHE = {"data": None, "ts": 0}

@app.route("/api/techmo/scan", methods=["GET"])
def api_techmo_scan():
    """Live momentum ranking of the TechMo universe. 15-min cache."""
    import yfinance as yf
    import pandas as pd
    import time as _t
    force = request.args.get("refresh", "").lower() in ("1", "true", "yes")
    if (not force and _TECHMO_SCAN_CACHE["data"]
            and (_t.time() - _TECHMO_SCAN_CACHE["ts"]) < 900):
        return jsonify(_TECHMO_SCAN_CACHE["data"])

    tickers = list(TECHMO_UNIVERSE.keys())
    try:
        raw = yf.download(tickers, period="14mo", progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    def get_close(t):
        try:
            s = (raw[t]["Close"] if len(tickers) > 1 else raw["Close"]).dropna()
            return s
        except Exception:
            return pd.Series(dtype=float)

    rows = []
    for t in tickers:
        s = get_close(t)
        if len(s) < 60:
            continue
        p_now = float(s.iloc[-1])
        p_12m = float(s.iloc[-253]) if len(s) >= 253 else float(s.iloc[0])
        p_3m  = float(s.iloc[-63])  if len(s) >= 63  else float(s.iloc[0])
        rows.append({
            "ticker":  t,
            "cluster": TECHMO_UNIVERSE[t],
            "price":   round(p_now, 2),
            "ret12m":  round((p_now / p_12m - 1) * 100, 1),
            "ret3m":   round((p_now / p_3m  - 1) * 100, 1),
        })

    if not rows:
        return jsonify({"success": False, "error": "no price data"})

    df = pd.DataFrame(rows)
    df["z12"]  = (df["ret12m"] - df["ret12m"].mean()) / df["ret12m"].std()
    df["z3"]   = (df["ret3m"]  - df["ret3m"].mean())  / df["ret3m"].std()
    df["score"] = (0.5 * df["z12"] + 0.5 * df["z3"]).round(3)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = (df.index + 1).astype(int)

    result = {
        "success":    True,
        "universe":   len(df),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "signals":    df.head(40).to_dict(orient="records"),
    }
    _TECHMO_SCAN_CACHE["data"] = result
    _TECHMO_SCAN_CACHE["ts"]   = _t.time()
    return jsonify(result)

@app.route("/api/techmo/performance", methods=["GET"])
def api_techmo_performance():
    """Live P&L for TechMo portfolio — fetches USD prices via yfinance."""
    import yfinance as yf
    try:
        with open(TECHMO_FILE) as f:
            port = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    positions = port.get("positions", [])
    tickers   = [p["ticker"] for p in positions]
    yf_syms   = tickers  # US stocks, no suffix

    prices = {}
    try:
        raw = yf.download(yf_syms, period="2d", progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
        for t in tickers:
            try:
                s = (raw[t]["Close"] if len(yf_syms) > 1 else raw["Close"]).dropna()
                if not s.empty:
                    prices[t] = round(float(s.iloc[-1]), 2)
            except Exception:
                pass
    except Exception:
        pass

    holdings = []
    total_cost    = 0.0
    total_current = 0.0
    for p in positions:
        t      = p["ticker"]
        shares = p["shares"]
        ep     = p["entry_price"]
        cost   = round(shares * ep, 2)
        now_px = prices.get(t, ep)
        value  = round(shares * now_px, 2)
        pnl    = round(value - cost, 2)
        pnl_pct = round((now_px / ep - 1) * 100, 2) if ep else 0
        total_cost    += cost
        total_current += value
        holdings.append({
            "ticker":      t,
            "cluster":     p.get("cluster", ""),
            "rank":        p.get("rank"),
            "shares":      shares,
            "entry_price": ep,
            "now_price":   now_px,
            "cost":        cost,
            "value":       value,
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "has_price":   t in prices,
        })

    holdings.sort(key=lambda h: h["rank"] or 99)
    total_pnl     = round(total_current - total_cost, 2)
    total_pnl_pct = round(total_pnl / total_cost * 100, 2) if total_cost else 0

    return jsonify({
        "success":       True,
        "universe":      port.get("universe", "TechMo"),
        "entry_date":    port.get("entry_date", ""),
        "next_rebalance": port.get("next_rebalance", ""),
        "holdings":      holdings,
        "total_cost":    round(total_cost, 2),
        "total_current": round(total_current, 2),
        "total_pnl":     total_pnl,
        "total_pnl_pct": total_pnl_pct,
    })


@app.route("/api/techmo/chart", methods=["GET"])
def api_techmo_chart():
    """Daily cumulative return: TechMo portfolio vs QQQ from entry date."""
    import yfinance as yf
    import pandas as pd
    try:
        with open(TECHMO_FILE) as f:
            port = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    positions  = port.get("positions", [])
    start_date = port.get("entry_date", "2026-05-21")
    tickers    = [p["ticker"] for p in positions]
    yf_syms    = tickers + ["QQQ"]
    total_cost = sum(p["shares"] * p["entry_price"] for p in positions)

    try:
        raw = yf.download(yf_syms, start=start_date, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    def get_close(sym):
        try:
            s = (raw[sym]["Close"] if len(yf_syms) > 1 else raw["Close"]).dropna()
            return s
        except Exception:
            return pd.Series(dtype=float)

    qqq_s = get_close("QQQ")
    ticker_closes = {t: get_close(t) for t in tickers}

    all_dates = qqq_s.index if not qqq_s.empty else pd.DatetimeIndex([])
    if all_dates.empty:
        return jsonify({"success": False, "error": "no price data"})

    port_pct, qqq_pct = [], []
    base_qqq = float(qqq_s.iloc[0]) if not qqq_s.empty else None

    for dt in all_dates:
        val = 0.0
        for p in positions:
            t  = p["ticker"]
            s  = ticker_closes.get(t)
            ep = p["entry_price"]
            if s is not None and dt in s.index:
                val += p["shares"] * float(s.loc[dt])
            else:
                val += p["shares"] * ep
        port_pct.append(round((val - total_cost) / total_cost * 100, 3))
        if base_qqq and dt in qqq_s.index:
            qqq_pct.append(round((float(qqq_s.loc[dt]) / base_qqq - 1) * 100, 3))
        else:
            qqq_pct.append(qqq_pct[-1] if qqq_pct else 0)

    dates_out = [d.strftime("%Y-%m-%d") for d in all_dates]
    return jsonify({
        "success":      True,
        "dates":        dates_out,
        "portfolio":    port_pct,
        "qqq":          qqq_pct,
        "start_date":   start_date,
        "total_return": port_pct[-1] if port_pct else None,
        "qqq_return":   qqq_pct[-1] if qqq_pct else None,
    })


# ── TechMo multi-user portfolio ───────────────────────────────────────────────

@app.route("/api/techmo-users/<user_id>/portfolio", methods=["GET"])
def api_techmo_portfolio_get(user_id):
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(techmo_portfolio_path(user_id)) as f:
            pf = json.load(f)
    except Exception:
        pf = {"status": "empty", "basket": [], "capital": 0}
    return jsonify({"success": True, "portfolio": pf})


@app.route("/api/techmo-users/<user_id>/portfolio", methods=["POST"])
def api_techmo_portfolio_save(user_id):
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    data = request.get_json() or {}
    path = techmo_portfolio_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"success": True})


@app.route("/api/techmo-users/<user_id>/performance", methods=["GET"])
def api_techmo_performance_user(user_id):
    """Live P&L for a specific user's TechMo portfolio."""
    import yfinance as yf
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(techmo_portfolio_path(user_id)) as f:
            pf = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    basket = pf.get("basket", [])
    if not basket:
        return jsonify({"success": True, "holdings": [], "total_cost": 0,
                        "total_current": 0, "total_pnl": 0, "total_pnl_pct": 0,
                        "realized_pnl": 0, "initial_capital": pf.get("capital", 0),
                        "tracking_since": None})

    live = request.args.get("live", "").lower() in ("1", "true")
    tickers = [h["ticker"] for h in basket if h.get("ticker")]

    # Load or fetch prices
    lp_path = techmo_live_prices_path(user_id)
    prices, prices_updated_at = {}, ""
    if live:
        try:
            raw = yf.download(tickers, period="2d", progress=False,
                              auto_adjust=True, group_by="ticker", threads=True)
            for t in tickers:
                try:
                    s = (raw[t]["Close"] if len(tickers) > 1 else raw["Close"]).dropna()
                    if not s.empty:
                        prices[t] = round(float(s.iloc[-1]), 2)
                except Exception:
                    pass
            prices_updated_at = datetime.now().strftime("%d %b %H:%M")
            os.makedirs(os.path.dirname(lp_path), exist_ok=True)
            with open(lp_path, "w") as f:
                json.dump({"prices": prices, "updated_at": prices_updated_at}, f)
        except Exception:
            pass
    else:
        try:
            with open(lp_path) as f:
                lp = json.load(f)
            prices = lp.get("prices", {})
            prices_updated_at = lp.get("updated_at", "")
        except Exception:
            pass

    # Realized P&L from history
    realized_pnl = 0.0
    initial_capital = pf.get("capital", 0) or 0.0
    try:
        with open(techmo_history_path(user_id)) as f:
            hist = json.load(f)
        for rb in hist:
            for s in rb.get("sells", []):
                realized_pnl += s.get("pnl", 0)
        if not initial_capital and hist:
            initial_capital = sum(
                b.get("qty", 0) * b.get("price", 0)
                for b in hist[0].get("buys", [])
            )
    except Exception:
        pass

    holdings, total_cost, total_current = [], 0.0, 0.0
    for h in basket:
        t   = h.get("ticker", "")
        qty = h.get("qty", 0) or h.get("shares", 0)
        ep  = h.get("entry_price", 0)
        cost = round(qty * ep, 2)
        now_px = prices.get(t, ep)
        value  = round(qty * now_px, 2)
        pnl    = round(value - cost, 2)
        pnl_pct = round((now_px / ep - 1) * 100, 2) if ep else 0
        total_cost    += cost
        total_current += value
        holdings.append({
            "ticker":      t,
            "cluster":     h.get("cluster", TECHMO_UNIVERSE.get(t, "")),
            "rank":        h.get("rank"),
            "shares":      qty,
            "entry_price": ep,
            "entry_date":  h.get("entry_date", ""),
            "now_price":   now_px,
            "cost":        cost,
            "value":       value,
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "has_price":   t in prices,
        })

    holdings.sort(key=lambda h: h["rank"] or 99)
    total_pnl     = round(total_current - total_cost, 2)
    total_pnl_pct = round(total_pnl / total_cost * 100, 2) if total_cost else 0
    initial_capital = initial_capital or total_cost

    return jsonify({
        "success":          True,
        "holdings":         holdings,
        "total_cost":       round(total_cost, 2),
        "total_current":    round(total_current, 2),
        "total_pnl":        total_pnl,
        "total_pnl_pct":    total_pnl_pct,
        "realized_pnl":     round(realized_pnl, 2),
        "initial_capital":  round(initial_capital, 2),
        "tracking_since":   pf.get("tracking_since", ""),
        "next_rebalance":   pf.get("next_rebalance", "2026-06-01"),
        "prices_updated_at": prices_updated_at,
    })


@app.route("/api/techmo-users/<user_id>/upload-tradebook", methods=["POST"])
def api_techmo_upload_tradebook(user_id):
    """Upload IBKR trade CSV — same format as Mom20 trade book."""
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    import csv, io
    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "error": "no file"})
    try:
        content = file.read().decode("utf-8-sig")
        reader  = csv.DictReader(io.StringIO(content))
        trades  = []
        for row in reader:
            ticker = (row.get("Symbol") or row.get("ticker") or "").strip().upper()
            action = (row.get("Action") or row.get("action") or "").strip().upper()
            qty    = float(row.get("Quantity") or row.get("qty") or 0)
            price  = float(row.get("Price") or row.get("price") or 0)
            date_s = (row.get("Date") or row.get("date") or "").strip()
            if ticker and action in ("BUY", "SELL") and qty > 0 and price > 0:
                trades.append({"ticker": ticker, "action": action,
                               "qty": qty, "price": price, "date": date_s})
        if not trades:
            return jsonify({"success": False, "error": "no valid trades found"})

        # Apply trades to portfolio
        pf_path = techmo_portfolio_path(user_id)
        try:
            with open(pf_path) as f:
                pf = json.load(f)
        except Exception:
            pf = {"status": "empty", "basket": [], "capital": 0, "next_rebalance": "2026-06-01"}

        basket = {h["ticker"]: h for h in pf.get("basket", [])}
        hist_path = techmo_history_path(user_id)
        try:
            with open(hist_path) as f:
                hist = json.load(f)
        except Exception:
            hist = []

        rb = {"rebalance_date": trades[0]["date"], "buys": [], "sells": []}
        for t in trades:
            ticker = t["ticker"]
            if t["action"] == "BUY":
                if ticker in basket:
                    old = basket[ticker]
                    new_qty   = old["qty"] + t["qty"]
                    new_price = (old["qty"] * old["entry_price"] + t["qty"] * t["price"]) / new_qty
                    basket[ticker]["qty"]         = new_qty
                    basket[ticker]["entry_price"] = round(new_price, 4)
                else:
                    basket[ticker] = {
                        "ticker":      ticker,
                        "qty":         t["qty"],
                        "entry_price": t["price"],
                        "entry_date":  t["date"],
                        "cluster":     TECHMO_UNIVERSE.get(ticker, ""),
                        "rank":        None,
                    }
                rb["buys"].append({"ticker": ticker, "qty": t["qty"], "price": t["price"]})
            elif t["action"] == "SELL" and ticker in basket:
                ep  = basket[ticker]["entry_price"]
                pnl = round((t["price"] - ep) * t["qty"], 2)
                rb["sells"].append({"ticker": ticker, "qty": t["qty"],
                                    "price": t["price"], "pnl": pnl})
                basket[ticker]["qty"] -= t["qty"]
                if basket[ticker]["qty"] <= 0:
                    del basket[ticker]

        hist.append(rb)
        pf["basket"] = list(basket.values())
        pf["status"] = "seeded"
        if not pf.get("tracking_since") and rb["buys"]:
            pf["tracking_since"] = trades[0]["date"]
        if not pf.get("capital"):
            buy_tot  = sum(b["qty"] * b["price"] for b in rb["buys"])
            sell_tot = sum(s["qty"] * s["price"] for s in rb["sells"])
            if buy_tot > sell_tot:
                pf["capital"] = round(pf.get("capital", 0) + buy_tot - sell_tot, 2)

        os.makedirs(os.path.dirname(pf_path), exist_ok=True)
        with open(pf_path, "w") as f:
            json.dump(pf, f, indent=2)
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=2)

        return jsonify({"success": True, "trades_processed": len(trades),
                        "holdings": len(pf["basket"])})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/techmo-users/<user_id>/history", methods=["GET"])
def api_techmo_history(user_id):
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(techmo_history_path(user_id)) as f:
            return jsonify({"success": True, "history": json.load(f)})
    except Exception:
        return jsonify({"success": True, "history": []})


@app.route("/api/techmo-users/<user_id>/chart", methods=["GET"])
def api_techmo_chart_user(user_id):
    """Daily portfolio return vs QQQ for a specific user."""
    import yfinance as yf
    import pandas as pd
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(techmo_portfolio_path(user_id)) as f:
            pf = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    basket = pf.get("basket", [])
    tracking_since = pf.get("tracking_since", "")
    if not basket or not tracking_since:
        return jsonify({"success": False, "error": "no portfolio data"})

    tickers    = [h["ticker"] for h in basket]
    total_cost = sum((h.get("qty") or h.get("shares", 0)) * h.get("entry_price", 0) for h in basket)
    yf_syms    = tickers + ["QQQ"]

    try:
        raw = yf.download(yf_syms, start=tracking_since, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    def get_close(sym):
        try:
            return (raw[sym]["Close"] if len(yf_syms) > 1 else raw["Close"]).dropna()
        except Exception:
            return pd.Series(dtype=float)

    qqq_s = get_close("QQQ")
    closes = {h["ticker"]: get_close(h["ticker"]) for h in basket}
    all_dates = qqq_s.index if not qqq_s.empty else pd.DatetimeIndex([])
    base_qqq  = float(qqq_s.iloc[0]) if not qqq_s.empty else None

    port_pct, qqq_pct = [], []
    for dt in all_dates:
        val = sum(
            (h.get("qty") or h.get("shares", 0)) *
            (float(closes[h["ticker"]].loc[dt]) if h["ticker"] in closes and dt in closes[h["ticker"]].index
             else h.get("entry_price", 0))
            for h in basket
        )
        port_pct.append(round((val - total_cost) / total_cost * 100, 3) if total_cost else 0)
        qqq_pct.append(round((float(qqq_s.loc[dt]) / base_qqq - 1) * 100, 3)
                       if base_qqq and dt in qqq_s.index else (qqq_pct[-1] if qqq_pct else 0))

    return jsonify({
        "success":      True,
        "dates":        [d.strftime("%Y-%m-%d") for d in all_dates],
        "portfolio":    port_pct,
        "qqq":          qqq_pct,
        "total_return": port_pct[-1] if port_pct else None,
        "qqq_return":   qqq_pct[-1] if qqq_pct else None,
    })


# ── ETF positions (per user, manual entry) ────────────────────────────────────

@app.route("/api/portfolio-users/<user_id>/etf-positions", methods=["GET"])
def api_etf_positions_get(user_id):
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    try:
        with open(etf_positions_path(user_id)) as f:
            positions = json.load(f)
    except Exception:
        positions = []
    try:
        with open(etf_history_path(user_id)) as f:
            history = json.load(f)
    except Exception:
        history = []
    return jsonify({"success": True, "positions": positions, "history": history})


@app.route("/api/portfolio-users/<user_id>/etf-positions", methods=["POST"])
def api_etf_position_add(user_id):
    """Manually record a new ETF position entry."""
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    data = request.get_json() or {}
    import uuid, datetime

    pos = {
        "id":          str(uuid.uuid4())[:8],
        "ticker":      (data.get("ticker") or "").upper().strip(),
        "name":        data.get("name", ""),
        "qty":         int(data.get("qty") or 0),
        "entry_price": float(data.get("entry_price") or 0),
        "entry_date":  data.get("entry_date") or datetime.date.today().isoformat(),
        "z_score_rank": data.get("z_score_rank"),
        "z_score":      data.get("z_score"),
        "status":      "open",
    }
    if not pos["ticker"] or pos["qty"] <= 0 or pos["entry_price"] <= 0:
        return jsonify({"success": False, "error": "ticker, qty, entry_price required"})

    path = etf_positions_path(user_id)
    try:
        with open(path) as f:
            positions = json.load(f)
    except Exception:
        positions = []
    positions.append(pos)
    with open(path, "w") as f:
        json.dump(positions, f, indent=2)
    return jsonify({"success": True, "position": pos})


@app.route("/api/portfolio-users/<user_id>/etf-positions/<pos_id>/exit", methods=["POST"])
def api_etf_position_exit(user_id, pos_id):
    """Record ETF exit, move to history."""
    if not get_user(user_id):
        return jsonify({"success": False, "error": "user not found"})
    data = request.get_json() or {}
    import datetime

    pos_path  = etf_positions_path(user_id)
    hist_path = etf_history_path(user_id)

    try:
        with open(pos_path) as f:
            positions = json.load(f)
    except Exception:
        positions = []

    pos = next((p for p in positions if p["id"] == pos_id), None)
    if not pos:
        return jsonify({"success": False, "error": "position not found"})

    exit_price = float(data.get("exit_price") or 0)
    exit_date  = data.get("exit_date") or datetime.date.today().isoformat()
    if exit_price <= 0:
        return jsonify({"success": False, "error": "exit_price required"})

    pnl_abs = round((exit_price - pos["entry_price"]) * pos["qty"], 2)
    pnl_pct = round((exit_price / pos["entry_price"] - 1) * 100, 2)
    from datetime import date as _date
    try:
        hold_days = (_date.fromisoformat(exit_date) - _date.fromisoformat(pos["entry_date"])).days
    except Exception:
        hold_days = None

    record = {**pos, "exit_price": exit_price, "exit_date": exit_date,
              "pnl_abs": pnl_abs, "pnl_pct": pnl_pct, "hold_days": hold_days,
              "status": "closed"}

    try:
        with open(hist_path) as f:
            history = json.load(f)
    except Exception:
        history = []
    history.append(record)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    positions = [p for p in positions if p["id"] != pos_id]
    with open(pos_path, "w") as f:
        json.dump(positions, f, indent=2)

    return jsonify({"success": True, "record": record})


# ============ Mom20 Portfolio Manager ============

from data.mom20_portfolio import (
    load_portfolio, save_portfolio, calc_performance,
    get_next_rebalance_date, days_to_rebalance,
    OVERFLOW_PORTFOLIO_FILE,
    load_history, save_history, MOM20_HISTORY_FILE,
)
import datetime as _dt


def _mom20_live_prices():
    """Returns (signals_list, regime_str, price_map)."""
    try:
        result = live_signals_scanner.scan_entry_signals()
        sigs   = result.get("mom20_signals", []) + result.get("mom20_overflow", [])
        regime = result.get("mom20_regime", "OFF")
        prices = {s["ticker"]: s["price"] for s in sigs}
        return sigs, regime, prices
    except Exception:
        return [], "OFF", {}


@app.route("/api/mom20-portfolio", methods=["GET"])
def mom20_portfolio_state():
    """Return portfolio state + live performance."""
    portfolio = load_portfolio()
    _, regime, prices = _mom20_live_prices()
    perf = calc_performance(portfolio, prices)
    return jsonify({
        **perf,
        "capital": portfolio.get("capital", 2000000),
        "next_rebalance": str(get_next_rebalance_date()),
        "days_to_rebalance": days_to_rebalance(),
        "regime": regime,
        "basket": portfolio.get("basket", []),
        "pending_orders": portfolio.get("pending_orders", []),
    })


@app.route("/api/mom20-portfolio/save", methods=["POST"])
def mom20_portfolio_save():
    """Snapshot basket + current prices → paper tracking begins."""
    data = request.get_json() or {}
    capital = int(data.get("capital", 2000000))
    basket_in = data.get("basket", [])  # [{ticker, weight, price}]
    if not basket_in:
        return jsonify({"success": False, "error": "Basket is empty"}), 400

    portfolio = load_portfolio()
    old_basket = {s["ticker"]: s for s in portfolio.get("basket", [])}
    new_tickers = {s["ticker"] for s in basket_in}

    today_str = _dt.date.today().isoformat()
    history = load_history()
    rebal_num = len(history) + 1

    # Exits: in old basket but not in new
    exits = []
    for t, s in old_basket.items():
        if t not in new_tickers:
            entry_price = s.get("invested_price") or s.get("saved_price") or 0
            # Use current price from incoming basket prices map (best proxy)
            price_map = {x["ticker"]: x.get("price", 0) for x in basket_in}
            exit_price = price_map.get(t, entry_price)
            entry_date = s.get("entry_date", portfolio.get("tracking_since", today_str))
            hold_days = ((_dt.date.fromisoformat(today_str) - _dt.date.fromisoformat(entry_date)).days
                         if entry_date else 0)
            pnl_pct = round((exit_price / entry_price - 1) * 100, 2) if entry_price else 0
            exits.append({"ticker": t, "entry_price": entry_price, "exit_price": exit_price,
                          "pnl_pct": pnl_pct, "hold_days": hold_days})

    # Entries: in new basket but not in old
    entries = []
    holds = []
    for s in basket_in:
        t = s["ticker"]
        slot = round(capital * s.get("weight", 5) / 100)
        base = {"ticker": t, "rank": s.get("rank"), "score": s.get("score"),
                "beta": s.get("beta"), "ret12m": s.get("ret12m"), "ret3m": s.get("ret3m"),
                "entry_price": s.get("price", 0), "capital": slot}
        if t in old_basket:
            holds.append({"ticker": t, "entry_price": old_basket[t].get("invested_price") or
                          old_basket[t].get("saved_price") or s.get("price", 0)})
        else:
            entries.append(base)

    slot_size = round(capital / max(len(basket_in), 1))
    pending_rebalance = {
        "rebalance_num": rebal_num,
        "date": today_str,
        "capital": capital,
        "slot": slot_size,
        "exits": exits,
        "entries": entries,
        "holds": holds,
        "finalized": False,
    }

    portfolio["capital"] = capital
    portfolio["status"] = "paper"
    portfolio["tracking_since"] = today_str
    portfolio["basket"] = [
        {
            "ticker": s["ticker"],
            "weight": s["weight"],
            "saved_price": s.get("price", 0),
            "invested_price": None,
            "entry_date": today_str,
            "rank": s.get("rank"),
            "score": s.get("score"),
            "beta": s.get("beta"),
            "ret12m": s.get("ret12m"),
            "ret3m": s.get("ret3m"),
        }
        for s in basket_in
    ]
    portfolio["pending_orders"] = []
    portfolio["pending_rebalance"] = pending_rebalance
    save_portfolio(portfolio)
    return jsonify({"success": True, "status": "paper", "tracking_since": portfolio["tracking_since"]})


@app.route("/api/mom20-portfolio/search", methods=["GET"])
def mom20_portfolio_search():
    """Search tickers. q=__IMPORT__ returns current Mom20 top-20."""
    q = request.args.get("q", "").upper().strip()
    if q == "__IMPORT__":
        try:
            result = live_signals_scanner.scan_entry_signals()
            signals = result.get("mom20_signals", [])[:20]
            return jsonify({"success": True, "signals": [
                {
                    "ticker": s["ticker"], "price": s["price"], "rank": s["rank"],
                    "score": s.get("momentum_score"), "beta": s.get("beta"),
                    "ret12m": s.get("ret_12m"), "ret3m": s.get("ret_3m"),
                } for s in signals
            ]})
        except Exception as e:
            return jsonify({"success": False, "signals": [], "error": str(e)})
    if len(q) < 2:
        return jsonify({"success": True, "results": []})
    try:
        result = live_signals_scanner.scan_entry_signals()
        all_sigs = result.get("mom20_signals", []) + result.get("mom20_overflow", [])
        matches = [{"ticker": s["ticker"], "price": s["price"], "rank": s.get("rank")}
                   for s in all_sigs if q in s["ticker"]]
        if matches:
            return jsonify({"success": True, "results": matches[:6]})
    except Exception:
        pass
    try:
        import yfinance as yf
        t = yf.Ticker(f"{q}.NS")
        price = getattr(t.fast_info, 'last_price', None)
        if price:
            return jsonify({"success": True, "results": [{"ticker": q, "price": round(float(price), 2)}]})
    except Exception:
        pass
    return jsonify({"success": True, "results": []})


@app.route("/api/mom20-portfolio/place-orders", methods=["POST"])
def mom20_portfolio_place_orders():
    """Place CNC market orders via Kite."""
    data = request.get_json() or {}
    user_id = data.get("user_id")
    orders_to_place = data.get("orders", [])

    if not user_id or user_id not in brokers:
        return jsonify({"success": False, "error": "Kite login required"}), 400
    broker = brokers[user_id]

    results = []
    for o in orders_to_place:
        ticker = o["ticker"]
        shares = int(o["shares"])
        side   = o["side"].upper()
        try:
            r = broker.place_sell_order(ticker, shares) if side == "SELL" else broker.place_buy_order(ticker, shares)
            results.append({"ticker": ticker, "side": side, "shares": shares,
                            "order_id": r.get("order_id"), "status": "PLACED"})
        except Exception as e:
            results.append({"ticker": ticker, "side": side, "shares": shares,
                            "order_id": None, "status": "ERROR", "error": str(e)})

    portfolio = load_portfolio()
    portfolio["pending_orders"] = results
    save_portfolio(portfolio)
    return jsonify({"success": True, "orders": results})


@app.route("/api/mom20-portfolio/order-status", methods=["GET"])
def mom20_portfolio_order_status():
    """Poll Kite for pending order statuses."""
    user_id = request.args.get("user_id")
    if not user_id or user_id not in brokers:
        return jsonify({"success": False, "error": "Kite login required"}), 400

    portfolio = load_portfolio()
    pending = portfolio.get("pending_orders", [])
    if not pending:
        return jsonify({"success": True, "orders": [], "all_complete": True})

    broker = brokers[user_id]
    try:
        kite_orders = broker._kite.orders() if broker._kite else []
        kite_map = {str(o["order_id"]): o for o in kite_orders}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    for o in pending:
        ko = kite_map.get(str(o.get("order_id", "")))
        if ko:
            o["status"]    = ko.get("status", o["status"])
            o["avg_price"] = ko.get("average_price")

    portfolio["pending_orders"] = pending
    save_portfolio(portfolio)
    all_complete = all(o.get("status", "").upper() in ("COMPLETE", "ERROR", "REJECTED") for o in pending)
    return jsonify({"success": True, "orders": pending, "all_complete": all_complete})


@app.route("/api/mom20-portfolio/history", methods=["GET"])
def mom20_portfolio_history():
    return jsonify({"success": True, "history": load_history()})


@app.route("/api/mom20-portfolio/reset", methods=["POST"])
def mom20_portfolio_reset():
    """Reset portfolio status back to paper (clears pending orders + invested prices)."""
    portfolio = load_portfolio()
    for s in portfolio.get("basket", []):
        s["invested_price"] = None
    portfolio["status"] = "paper" if portfolio.get("basket") else "empty"
    portfolio["pending_orders"] = []
    save_portfolio(portfolio)
    return jsonify({"success": True, "status": portfolio["status"]})


@app.route("/api/mom20-portfolio/confirm-investment", methods=["POST"])
def mom20_portfolio_confirm_investment():
    """
    After all orders complete: set invested_price = avg_price from Kite.
    Switches tracking mode from 'paper' → 'invested'.
    Expects {orders: [{ticker, avg_price}]}
    """
    data = request.get_json() or {}
    filled = {o["ticker"]: float(o["avg_price"]) for o in data.get("orders", []) if o.get("avg_price")}

    portfolio = load_portfolio()
    basket = portfolio.get("basket", [])
    for s in basket:
        if s["ticker"] in filled:
            s["invested_price"] = filled[s["ticker"]]

    portfolio["basket"] = basket
    portfolio["status"] = "invested"
    today_str = _dt.date.today().isoformat()
    portfolio["tracking_since"] = today_str
    portfolio["pending_orders"] = []

    # Finalize and append rebalance history record
    pending = portfolio.get("pending_rebalance")
    if pending and not pending.get("finalized"):
        # Update entry prices to actual fill prices
        for e in pending.get("entries", []):
            if e["ticker"] in filled:
                e["entry_price"] = filled[e["ticker"]]
        # Compute summary
        invested = sum(s.get("invested_price", s.get("saved_price", 0)) *
                       int((portfolio["capital"] * next(
                           (b["weight"] for b in basket if b["ticker"] == s.get("ticker")), 5
                       ) / 100) / max(s.get("invested_price", s.get("saved_price", 1)), 1))
                       for s in basket if s.get("invested_price") or s.get("saved_price"))
        n_pos = len(basket)
        pending["summary"] = {
            "positions": n_pos,
            "max_slots": 20,
            "capital": portfolio["capital"],
            "slot": pending.get("slot", round(portfolio["capital"] / max(n_pos, 1))),
        }
        pending["finalized"] = True
        pending["confirmed_date"] = today_str
        history = load_history()
        history.append(pending)
        save_history(history)
        portfolio["pending_rebalance"] = None

    save_portfolio(portfolio)
    return jsonify({"success": True, "status": "invested",
                    "tracking_since": portfolio["tracking_since"]})


# ============ Overflow Portfolio Manager ============

def _overflow_live_prices():
    """Returns (overflow_signals, regime_str, price_map)."""
    try:
        result = live_signals_scanner.scan_entry_signals()
        sigs   = result.get("mom20_overflow", []) + result.get("mom20_signals", [])
        regime = result.get("mom20_regime", "OFF")
        prices = {s["ticker"]: s["price"] for s in sigs}
        return sigs, regime, prices
    except Exception:
        return [], "OFF", {}


@app.route("/api/overflow-portfolio", methods=["GET"])
def overflow_portfolio_state():
    portfolio = load_portfolio(OVERFLOW_PORTFOLIO_FILE)
    _, regime, prices = _overflow_live_prices()
    perf = calc_performance(portfolio, prices)
    return jsonify({
        **perf,
        "capital": portfolio.get("capital", 2000000),
        "next_rebalance": str(get_next_rebalance_date()),
        "days_to_rebalance": days_to_rebalance(),
        "regime": regime,
        "basket": portfolio.get("basket", []),
        "pending_orders": portfolio.get("pending_orders", []),
    })


@app.route("/api/overflow-portfolio/save", methods=["POST"])
def overflow_portfolio_save():
    data = request.get_json() or {}
    capital = int(data.get("capital", 2000000))
    basket_in = data.get("basket", [])
    if not basket_in:
        return jsonify({"success": False, "error": "Basket is empty"}), 400
    portfolio = load_portfolio(OVERFLOW_PORTFOLIO_FILE)
    portfolio["capital"] = capital
    portfolio["status"] = "paper"
    portfolio["tracking_since"] = _dt.date.today().isoformat()
    portfolio["basket"] = [
        {
            "ticker": s["ticker"],
            "weight": s["weight"],
            "saved_price": s.get("price", 0),
            "invested_price": None,
        }
        for s in basket_in
    ]
    portfolio["pending_orders"] = []
    save_portfolio(portfolio, OVERFLOW_PORTFOLIO_FILE)
    return jsonify({"success": True, "status": "paper", "tracking_since": portfolio["tracking_since"]})


@app.route("/api/overflow-portfolio/search", methods=["GET"])
def overflow_portfolio_search():
    """Search tickers. q=__IMPORT__ returns current overflow signals."""
    q = request.args.get("q", "").upper().strip()
    if q == "__IMPORT__":
        try:
            result = live_signals_scanner.scan_entry_signals()
            signals = result.get("mom20_overflow", [])[:20]
            return jsonify({"success": True, "signals": [
                {"ticker": s["ticker"], "price": s["price"], "rank": s.get("rank")} for s in signals
            ]})
        except Exception as e:
            return jsonify({"success": False, "signals": [], "error": str(e)})
    if len(q) < 2:
        return jsonify({"success": True, "results": []})
    try:
        result = live_signals_scanner.scan_entry_signals()
        all_sigs = result.get("mom20_overflow", []) + result.get("mom20_signals", [])
        matches = [{"ticker": s["ticker"], "price": s["price"], "rank": s.get("rank")}
                   for s in all_sigs if q in s["ticker"]]
        if matches:
            return jsonify({"success": True, "results": matches[:6]})
    except Exception:
        pass
    try:
        import yfinance as yf
        t = yf.Ticker(f"{q}.NS")
        price = getattr(t.fast_info, 'last_price', None)
        if price:
            return jsonify({"success": True, "results": [{"ticker": q, "price": round(float(price), 2)}]})
    except Exception:
        pass
    return jsonify({"success": True, "results": []})


@app.route("/api/overflow-portfolio/place-orders", methods=["POST"])
def overflow_portfolio_place_orders():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    orders_to_place = data.get("orders", [])
    if not user_id or user_id not in brokers:
        return jsonify({"success": False, "error": "Kite login required"}), 400
    broker = brokers[user_id]
    results = []
    for o in orders_to_place:
        ticker = o["ticker"]
        shares = int(o["shares"])
        side   = o["side"].upper()
        try:
            r = broker.place_sell_order(ticker, shares) if side == "SELL" else broker.place_buy_order(ticker, shares)
            results.append({"ticker": ticker, "side": side, "shares": shares,
                            "order_id": r.get("order_id"), "status": "PLACED"})
        except Exception as e:
            results.append({"ticker": ticker, "side": side, "shares": shares,
                            "order_id": None, "status": "ERROR", "error": str(e)})
    portfolio = load_portfolio(OVERFLOW_PORTFOLIO_FILE)
    portfolio["pending_orders"] = results
    save_portfolio(portfolio, OVERFLOW_PORTFOLIO_FILE)
    return jsonify({"success": True, "orders": results})


@app.route("/api/overflow-portfolio/order-status", methods=["GET"])
def overflow_portfolio_order_status():
    user_id = request.args.get("user_id")
    if not user_id or user_id not in brokers:
        return jsonify({"success": False, "error": "Kite login required"}), 400
    portfolio = load_portfolio(OVERFLOW_PORTFOLIO_FILE)
    pending = portfolio.get("pending_orders", [])
    if not pending:
        return jsonify({"success": True, "orders": [], "all_complete": True})
    broker = brokers[user_id]
    try:
        kite_orders = broker._kite.orders() if broker._kite else []
        kite_map = {str(o["order_id"]): o for o in kite_orders}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    for o in pending:
        ko = kite_map.get(str(o.get("order_id", "")))
        if ko:
            o["status"]    = ko.get("status", o["status"])
            o["avg_price"] = ko.get("average_price")
    portfolio["pending_orders"] = pending
    save_portfolio(portfolio, OVERFLOW_PORTFOLIO_FILE)
    all_complete = all(o.get("status", "").upper() in ("COMPLETE", "ERROR", "REJECTED") for o in pending)
    return jsonify({"success": True, "orders": pending, "all_complete": all_complete})


@app.route("/api/overflow-portfolio/reset", methods=["POST"])
def overflow_portfolio_reset():
    """Reset portfolio status back to paper (clears pending orders + invested prices)."""
    portfolio = load_portfolio(OVERFLOW_PORTFOLIO_FILE)
    for s in portfolio.get("basket", []):
        s["invested_price"] = None
    portfolio["status"] = "paper" if portfolio.get("basket") else "empty"
    portfolio["pending_orders"] = []
    save_portfolio(portfolio, OVERFLOW_PORTFOLIO_FILE)
    return jsonify({"success": True, "status": portfolio["status"]})


@app.route("/api/overflow-portfolio/confirm-investment", methods=["POST"])
def overflow_portfolio_confirm_investment():
    data = request.get_json() or {}
    filled = {o["ticker"]: float(o["avg_price"]) for o in data.get("orders", []) if o.get("avg_price")}
    portfolio = load_portfolio(OVERFLOW_PORTFOLIO_FILE)
    basket = portfolio.get("basket", [])
    for s in basket:
        if s["ticker"] in filled:
            s["invested_price"] = filled[s["ticker"]]
    portfolio["basket"] = basket
    portfolio["status"] = "invested"
    portfolio["tracking_since"] = _dt.date.today().isoformat()
    portfolio["pending_orders"] = []
    save_portfolio(portfolio, OVERFLOW_PORTFOLIO_FILE)
    return jsonify({"success": True, "status": "invested",
                    "tracking_since": portfolio["tracking_since"]})


# ── Backtest runner (mom15_pit_report.py subprocess) ──────────────────────────
# Wraps `mom15_pit_report.py` as a subprocess so the UI can sweep params on the
# fly. Caches successful runs by hash of the request body.

import hashlib
import re
import subprocess
import time as _bt_time

_BACKTEST_LOCK = threading.Lock()
_BACKTEST_CACHE_DIR = os.path.join(DATA_STORE_PATH, "backtest_cache")
_BACKTEST_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKTEST_SCRIPT    = os.path.join(_BACKTEST_PROJECT_ROOT, "mom15_pit_report.py")
_AI_BACKTEST_SCRIPT = os.path.join(_BACKTEST_PROJECT_ROOT, "ai_universe_backtest.py")


def _bt_cache_key(payload: dict) -> str:
    """Stable hash of the request body — sort keys, json.dumps, sha256[:12]."""
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def _bt_build_args(payload: dict) -> tuple:
    """Map request JSON → (script_path, CLI flags).

    Returns (script, args) so the caller can pick the right executable.
    """
    universe = (payload.get("universe") or "").lower()

    # ── AI Universe — separate script ─────────────────────────────────────────
    if universe == "ai":
        args = []
        if payload.get("no_regime"):
            args.append("--no-regime")
        if payload.get("top_n") is not None:
            args += ["--top-n", str(int(payload["top_n"]))]
        if payload.get("buffer_in") is not None:
            args += ["--buffer-in", str(int(payload["buffer_in"]))]
        if payload.get("buffer_out") is not None:
            args += ["--buffer-out", str(int(payload["buffer_out"]))]
        if payload.get("max_per_sector") is not None:
            args += ["--max-per-sector", str(int(payload["max_per_sector"]))]
        if payload.get("ret12m_cap") is not None:
            args += ["--ret12m-cap", str(float(payload["ret12m_cap"]))]
        if payload.get("parabolic_filter"):
            args.append("--parabolic-filter")
        return _AI_BACKTEST_SCRIPT, args

    # ── mom15_pit_report.py universes ─────────────────────────────────────────
    args = []
    if universe == "n500":
        args.append("--n500")
    elif universe == "qqq":
        args.append("--qqq")
    elif universe == "n200":
        args.append("--mom20")
    # else: default Nifty200 Mom15 bi-monthly — script's default

    if payload.get("top_n") is not None:
        args += ["--top-n", str(int(payload["top_n"]))]
    if payload.get("buffer_in") is not None:
        args += ["--buffer-in", str(int(payload["buffer_in"]))]
    if payload.get("buffer_out") is not None:
        args += ["--buffer-out", str(int(payload["buffer_out"]))]
    if payload.get("beta_cap") is not None:
        args += ["--beta-cap", str(float(payload["beta_cap"]))]
    if payload.get("ema200_exit"):
        args.append("--ema200-exit")
    if (payload.get("rebal_day") or "").lower() == "mid":
        args += ["--rebal-day", "mid"]
    # Regime filter: "none" | "sma200" (default) | "ema200"
    regime = (payload.get("regime") or "").lower()
    if regime in ("none", "sma200", "ema200"):
        args += ["--regime", regime]
    return _BACKTEST_SCRIPT, args


def _bt_parse_summary(stdout: str) -> dict:
    """Extract FINAL SUMMARY block + year-by-year rows from the report stdout.

    Regexes are forgiving (whitespace tolerant) but assume the script's print
    format in `mom15_pit_report.py:print_final_summary` is unchanged.
    """
    summary = {}

    m = re.search(r"Period\s*:\s*(\S+)\s*→\s*(\S+)\s*\(([\d.]+)\s*years\)", stdout)
    if m:
        summary["period"] = f"{m.group(1)} → {m.group(2)}"
        summary["years"] = float(m.group(3))

    m = re.search(r"Total Return\s*:\s*([+\-]?[\d,\.]+)%", stdout)
    if m:
        summary["total_return_pct"] = float(m.group(1).replace(",", ""))

    m = re.search(r"CAGR\s*:\s*([+\-]?[\d,\.]+)%", stdout)
    if m:
        summary["cagr_pct"] = float(m.group(1).replace(",", ""))

    m = re.search(r"Win Rate\s*:\s*([\d\.]+)%\s*\((\d+)W\s*/\s*(\d+)L\)", stdout)
    if m:
        summary["win_rate_pct"] = float(m.group(1))
        summary["winners"] = int(m.group(2))
        summary["losers"] = int(m.group(3))

    m = re.search(r"Profit Factor\s*:\s*([\d\.]+|inf)", stdout)
    if m:
        v = m.group(1)
        summary["profit_factor"] = float("inf") if v == "inf" else float(v)

    m = re.search(r"Avg hold\s*:\s*(\d+)\s*days", stdout)
    if m:
        summary["avg_hold_days"] = int(m.group(1))

    m = re.search(r"Total charges\s*:\s*[₹$]\s*([\d,]+)", stdout)
    if m:
        summary["total_charges"] = int(m.group(1).replace(",", ""))

    # YEAR-BY-YEAR: lines like "  2015  + 12.8%  ███..." or "  2018  -  3.4%  ░░..."
    yby = []
    in_block = False
    for line in stdout.splitlines():
        if "YEAR-BY-YEAR" in line:
            in_block = True
            continue
        if in_block:
            ym = re.match(r"\s*(\d{4})\s+([+\-])\s*([\d\.]+)%", line)
            if ym:
                yr = int(ym.group(1))
                sign = -1 if ym.group(2) == "-" else 1
                pct = sign * float(ym.group(3))
                yby.append({"year": yr, "return_pct": pct})
            elif yby and not line.strip():
                # blank line after rows — end of block
                break
    if yby:
        summary["year_by_year"] = yby

    return summary


def _bt_run_subprocess(script: str, args: list, timeout: int = 600) -> tuple:
    """Run backtest script with given args. Returns (returncode, stdout, stderr, cmd)."""
    cmd = [sys.executable, script] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_BACKTEST_PROJECT_ROOT,
    )
    return proc.returncode, proc.stdout, proc.stderr, cmd


_BACKTEST_JOBS = {}  # job_id → {"status": "running"|"done"|"error", "started_at": ts, "result": dict|None}


def _bt_run_job(job_id, payload, key, cache_path):
    """Background worker — runs the subprocess and stores the result/error
    in _BACKTEST_JOBS[job_id]. Designed so GET /api/backtest/status can poll."""
    script, args = _bt_build_args(payload)
    with _BACKTEST_LOCK:
        # Another concurrent job may have populated the cache while we waited
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                cached["cached"] = True
                cached["elapsed_sec"] = 0.0
                _BACKTEST_JOBS[job_id] = {"status": "done", "result": cached}
                return
            except Exception:
                pass

        t0 = _bt_time.time()
        try:
            rc, stdout, stderr, cmd = _bt_run_subprocess(script, args, timeout=600)
        except subprocess.TimeoutExpired:
            _BACKTEST_JOBS[job_id] = {"status": "error", "result": {
                "success": False, "error": "backtest timed out after 600 seconds",
                "command": " ".join([sys.executable, script] + args)}}
            return
        except Exception as e:
            _BACKTEST_JOBS[job_id] = {"status": "error", "result": {
                "success": False, "error": f"subprocess failed: {e}",
                "command": " ".join([sys.executable, script] + args)}}
            return
        elapsed = _bt_time.time() - t0

        script_name = os.path.basename(script)
        if rc != 0:
            _BACKTEST_JOBS[job_id] = {"status": "error", "result": {
                "success": False,
                "error": f"{script_name} exited with code {rc}",
                "stderr": (stderr or "")[-2000:],
                "stdout_tail": (stdout or "")[-2000:],
                "command": " ".join(cmd),
                "elapsed_sec": round(elapsed, 2)}}
            return

        summary = _bt_parse_summary(stdout)
        rebalance_count = len(re.findall(r"REBALANCE #", stdout))

        if "cagr_pct" not in summary:
            _BACKTEST_JOBS[job_id] = {"status": "error", "result": {
                "success": False,
                "error": "failed to parse FINAL SUMMARY from stdout",
                "stdout_tail": (stdout or "")[-2000:],
                "command": " ".join(cmd),
                "elapsed_sec": round(elapsed, 2)}}
            return

        response = {
            "success": True,
            "summary": summary,
            "raw_report": stdout,
            "rebalance_count": rebalance_count,
            "command": " ".join(cmd),
            "cached": False,
            "elapsed_sec": round(elapsed, 2),
            "cache_key": key,
        }
        try:
            os.makedirs(_BACKTEST_CACHE_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(response, f)
        except Exception as e:
            print(f"[backtest] cache write failed: {e}")

        _BACKTEST_JOBS[job_id] = {"status": "done", "result": response}


@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    """Start (or fetch) a backtest run. Async — long subprocesses run in
    a background thread so the proxy can't 504 us out.

    Body: {universe, top_n, buffer_in, buffer_out, beta_cap, ema200_exit, rebal_day}
    Cache hit → returns full result inline (success: true, cached: true).
    Cache miss → spawns a background thread, returns immediately
                 (success: true, in_progress: true, job_id, cache_key).
    Frontend then polls GET /api/backtest/status?id=<job_id> until done.
    """
    payload = request.get_json(silent=True) or {}
    key = _bt_cache_key(payload)
    cache_path = os.path.join(_BACKTEST_CACHE_DIR, f"{key}.json")

    # Cache hit — instant
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            cached["cached"] = True
            cached["elapsed_sec"] = 0.0
            return jsonify(cached)
        except Exception:
            # Corrupt cache — fall through to fresh run
            pass

    # Spawn background job
    import uuid as _uuid
    job_id = _uuid.uuid4().hex[:12]
    _BACKTEST_JOBS[job_id] = {"status": "running", "started_at": _bt_time.time()}
    threading.Thread(target=_bt_run_job,
                     args=(job_id, payload, key, cache_path),
                     daemon=True, name=f"backtest-{job_id}").start()
    return jsonify({
        "success":     True,
        "in_progress": True,
        "job_id":      job_id,
        "cache_key":   key,
        "message":     "Backtest started in background. Poll /api/backtest/status?id=<job_id>.",
    })


@app.route("/api/backtest/status", methods=["GET"])
def api_backtest_status():
    """Poll the status of a backtest job started via POST /api/backtest/run."""
    job_id = (request.args.get("id") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{6,32}", job_id):
        return jsonify({"success": False, "error": "missing or invalid id"}), 400
    job = _BACKTEST_JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "unknown job_id (expired or never existed)"}), 404
    if job["status"] == "running":
        elapsed = _bt_time.time() - job.get("started_at", _bt_time.time())
        return jsonify({
            "success":     True,
            "in_progress": True,
            "elapsed_sec": round(elapsed, 1),
        })
    # done or error → return the stored result (which already has success flag set)
    return jsonify(job["result"])


@app.route("/api/backtest/history", methods=["GET"])
def api_backtest_history():
    """Return the rebalance history (raw_report) for a cached backtest run."""
    key = (request.args.get("key") or "").strip()
    if not key or not re.fullmatch(r"[0-9a-f]{6,64}", key):
        return jsonify({"success": False, "error": "missing or invalid key"}), 400
    cache_path = os.path.join(_BACKTEST_CACHE_DIR, f"{key}.json")
    if not os.path.exists(cache_path):
        return jsonify({"success": False, "error": "cache miss — run /api/backtest/run first"}), 404
    try:
        with open(cache_path) as f:
            cached = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "error": f"cache read failed: {e}"}), 500
    return jsonify({
        "success": True,
        "raw_report": cached.get("raw_report", ""),
        "rebalance_count": cached.get("rebalance_count", 0),
    })


# ============ Run Server ============

def run_server():
    """Run the Flask development server."""
    config = load_config()
    print(f"\n{'='*60}")
    print("  NIFTY 500 RELATIVE STRENGTH DASHBOARD")
    print(f"{'='*60}")
    print(f"  Server: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"  RS Period: {config['rs_period']} days | EMA Period: {config['ema_period']} days")
    print(f"  Auto-refresh: Every {config['cache_ttl_minutes']} minutes")
    print(f"  Mode: {'Paper Trading' if PAPER_TRADING_ONLY else 'Live Trading'}")
    print(f"{'='*60}\n")

    # Start ETF signal scheduler (daemon so it exits with the main process)
    scheduler_thread = threading.Thread(target=_etf_signal_scheduler, daemon=True, name="etf-scheduler")
    scheduler_thread.start()

    # Pre-warm disk-backed caches so the first user click is fast (plan A).
    prewarm_thread = threading.Thread(target=_prewarm_caches, daemon=True, name="cache-prewarm")
    prewarm_thread.start()

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG_MODE,
        threaded=True
    )


if __name__ == "__main__":
    run_server()
