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
    get_kite_users, DATA_STORE_PATH, LIVE_POSITIONS_FILE,
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


def _migrate_legacy_data():
    """One-time: copy live_positions.json -> live_positions_{first_user}.json."""
    import json, shutil
    if not kite_users:
        return
    first_uid = kite_users[0]["id"]
    dest = os.path.join(DATA_STORE_PATH, f"live_positions_{first_uid}.json")
    if os.path.exists(LIVE_POSITIONS_FILE) and not os.path.exists(dest):
        shutil.copy2(LIVE_POSITIONS_FILE, dest)

_migrate_legacy_data()


def _find_position_owner(position_id):
    """Search all user_engines for a position. Returns (user_id, position) or (None, None)."""
    for uid, engine in user_engines.items():
        for p in engine.get_positions():
            if p["id"] == position_id:
                return uid, p
    return None, None

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
    """Force re-scan J+T entry signals."""
    global live_signals_refresh_in_progress

    with live_signals_refresh_lock:
        if live_signals_refresh_in_progress:
            return jsonify({
                "success": False,
                "error": "Refresh already in progress"
            }), 409

        live_signals_refresh_in_progress = True

    try:
        # Fetch real-time LTP from Zerodha if connected
        from data.live_signals_engine import (NIFTY_50_TICKERS, NIFTY_NEXT50_TICKERS,
                                              NIFTY_200_NEXT100_TICKERS)
        from config.settings import load_config
        config = load_config()
        universe = config.get("live_signals_universe", 50)
        if universe <= 50:
            scan_tickers = NIFTY_50_TICKERS
        elif universe <= 100:
            scan_tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS
        else:
            scan_tickers = NIFTY_50_TICKERS + NIFTY_NEXT50_TICKERS + NIFTY_200_NEXT100_TICKERS

        ltp_map = _fetch_zerodha_ltp(scan_tickers)

        data = live_signals_scanner.scan_entry_signals(
            force_refresh=True,
            progress_callback=update_live_signals_progress,
            ltp_map=ltp_map
        )

        return jsonify({
            "success": True,
            "data": data,
            "message": f"Live signals refreshed {'with Zerodha real-time prices' if ltp_map else 'with yfinance data'}"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        with live_signals_refresh_lock:
            live_signals_refresh_in_progress = False


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


@app.route("/api/live-signals/positions", methods=["GET"])
def get_live_positions():
    """Get active positions merged across all users, with LTPs from Zerodha."""
    try:
        all_positions = []
        for uid, engine in user_engines.items():
            for p in engine.get_positions():
                if "user_id" not in p:
                    p["user_id"] = uid
                all_positions.append(p)

        # Fetch LTPs for position tickers via first connected broker
        ltps = {}
        if all_positions:
            tickers = list(set(p["ticker"] for p in all_positions))
            for broker in brokers.values():
                if broker.is_connected():
                    instruments = [f"NSE:{t}" for t in tickers]
                    ltp_data = broker.get_ltp(instruments)
                    for t in tickers:
                        key = f"NSE:{t}"
                        if key in ltp_data:
                            ltps[t] = ltp_data[key].get("last_price")
                    break

            # Fallback: use yfinance close for any tickers missing LTP
            missing = [t for t in tickers if t not in ltps]
            if missing:
                import yfinance as yf
                for t in missing:
                    try:
                        hist = yf.Ticker(f"{t}.NS").history(period="5d")
                        if not hist.empty:
                            ltps[t] = round(float(hist["Close"].iloc[-1]), 2)
                    except Exception:
                        pass

        # Attach sector info to each position for diversity cap
        from sector_mapping import STOCK_SECTOR_MAP
        for p in all_positions:
            p["sector"] = STOCK_SECTOR_MAP.get(p["ticker"], "OTHER")

        return jsonify({"success": True, "positions": all_positions, "ltps": ltps})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/live-signals/force-exit", methods=["POST"])
def force_exit_position():
    """Force-exit: sell all remaining shares on Zerodha, then remove locally."""
    data = request.get_json()
    position_id = data.get("position_id")
    if not position_id:
        return jsonify({"success": False, "error": "position_id required"}), 400

    # Auto-detect owner
    owner_uid, pos = _find_position_owner(position_id)
    if pos is None:
        return jsonify({"success": False, "error": "Position not found"}), 400

    broker = brokers.get(owner_uid)
    engine = user_engines.get(owner_uid)

    # Sell on Zerodha
    try:
        order_result = broker.place_sell_order(
            pos["ticker"], pos["remaining_shares"])
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Sell order failed: {e}"}), 500

    # Order placed — record as closed with reason MANUAL_FORCE_EXIT
    result = engine.close_position(
        position_id, pos["entry_price"], "MANUAL_FORCE_EXIT",
        pos["remaining_shares"])
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400

    return jsonify({
        "success": True,
        "message": f"Sold {pos['remaining_shares']} shares of {pos['ticker']}",
        "order_id": order_result.get("order_id"),
        "closed": result,
    })


@app.route("/api/live-signals/buy", methods=["POST"])
def buy_live_signal():
    """Place BUY order on Zerodha, then record position locally on success."""
    data = request.get_json()
    ticker = data.get("ticker")
    strategy = data.get("strategy")
    price = data.get("price")
    amount = data.get("amount", 100000)
    support = data.get("support", 0)
    ibs = data.get("ibs", 0)
    metadata = data.get("metadata", {})
    user_id = data.get("user_id")

    if not ticker or not strategy or not price:
        return jsonify({"success": False, "error": "ticker, strategy, price required"}), 400

    if strategy not in ("J", "T", "R", "MW", "RS", "RS63", "MOM15"):
        return jsonify({"success": False, "error": "strategy must be J, T, R, MW, RS, RS63, or MOM15"}), 400

    if not user_id or user_id not in brokers:
        return jsonify({"success": False, "error": "Valid user_id required"}), 400

    try:
        amount = float(amount)
        price = float(price)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid amount or price"}), 400

    shares = int(amount // price)
    if shares <= 0:
        return jsonify({"success": False, "error": "Amount too small for even 1 share"}), 400

    broker = brokers[user_id]
    engine = user_engines[user_id]

    # Place real order on Zerodha
    try:
        order_result = broker.place_buy_order(ticker, shares)
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Order failed: {e}"}), 500

    # Order placed — now record locally
    metadata["order_id"] = order_result.get("order_id")
    # Store Nifty close at entry for crash shield (J, R, MW)
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period="5d")
        if not nifty.empty:
            metadata["nifty_at_entry"] = float(nifty["Close"].iloc[-1])
    except Exception:
        pass
    # Store structural SL for R (support param doubles as swing_low_stop)
    if strategy == "R" and support > 0:
        metadata["r_swing_low_stop"] = support
    result = engine.add_position(
        ticker, strategy, price, amount, support, ibs, metadata)

    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400

    return jsonify({"success": True, "position": result,
                    "order_id": order_result.get("order_id")})


@app.route("/api/live-signals/manual-entry", methods=["POST"])
def manual_entry_live_signal():
    """Record a manually-executed RS63 entry (no broker order placed).
    User buys on Zerodha at their chosen price, then calls this to log the position."""
    data = request.get_json()
    ticker   = data.get("ticker")
    strategy = data.get("strategy", "RS63")
    price    = data.get("price")
    qty      = data.get("qty")
    support  = data.get("support", 0)
    ibs      = data.get("ibs", 0)
    metadata = data.get("metadata", {})
    user_id  = data.get("user_id")

    if not ticker or not price or not qty:
        return jsonify({"success": False, "error": "ticker, price, qty required"}), 400

    if strategy not in ("J", "T", "R", "MW", "RS", "RS63", "MOM15"):
        return jsonify({"success": False, "error": "Invalid strategy"}), 400

    if not user_id or user_id not in user_engines:
        return jsonify({"success": False, "error": "Valid user_id required"}), 400

    try:
        price = float(price)
        qty   = int(qty)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid price or qty"}), 400

    if qty <= 0:
        return jsonify({"success": False, "error": "qty must be > 0"}), 400

    amount = price * qty
    metadata["manual_entry"] = True

    # Store Nifty close at entry for crash shield
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period="5d")
        if not nifty.empty:
            metadata["nifty_at_entry"] = float(nifty["Close"].iloc[-1])
    except Exception:
        pass

    engine = user_engines[user_id]
    result = engine.add_position(ticker, strategy, price, amount, support, ibs, metadata)

    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400

    return jsonify({"success": True, "position": result})


@app.route("/api/live-signals/exits", methods=["GET"])
def get_live_exit_signals():
    """Check exit conditions for all active positions (merged across users)."""
    try:
        # Fetch live LTPs from Zerodha if connected
        ltps = {}
        all_tickers = set()
        for engine in user_engines.values():
            for p in engine.get_positions():
                all_tickers.add(p["ticker"])
        if all_tickers:
            for broker in brokers.values():
                if broker.is_connected():
                    instruments = [f"NSE:{t}" for t in all_tickers]
                    ltp_data = broker.get_ltp(instruments)
                    for t in all_tickers:
                        key = f"NSE:{t}"
                        if key in ltp_data:
                            ltps[t] = ltp_data[key].get("last_price")
                    break

        all_exits = []
        for uid, engine in user_engines.items():
            exits = engine.check_exit_signals(ltps=ltps if ltps else None)
            for e in exits:
                e["user_id"] = uid
            all_exits.extend(exits)
        return jsonify({"success": True, "exits": all_exits})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/live-signals/execute-exit", methods=["POST"])
def execute_live_exit():
    """Place SELL order on Zerodha, then close/partial-close position locally."""
    data = request.get_json()
    position_id = data.get("position_id")
    exit_price = data.get("exit_price")
    reason = data.get("reason", "MANUAL")
    shares = data.get("shares")

    if not position_id or exit_price is None or shares is None:
        return jsonify({"success": False, "error": "position_id, exit_price, shares required"}), 400

    shares = int(shares)

    # Auto-detect owner
    owner_uid, pos = _find_position_owner(position_id)
    if pos is None:
        return jsonify({"success": False, "error": "Position not found"}), 400

    broker = brokers.get(owner_uid)
    engine = user_engines.get(owner_uid)

    # Place real sell order on Zerodha
    try:
        order_result = broker.place_sell_order(pos["ticker"], shares)
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Sell order failed: {e}"}), 500

    # Order placed — now record locally
    try:
        result = engine.close_position(
            position_id, float(exit_price), reason, shares)
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 400
        result["order_id"] = order_result.get("order_id")
        return jsonify({"success": True, "closed": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/live-signals/closed", methods=["GET"])
def get_closed_trades():
    """Get closed positions (trade history) from all user engines."""
    try:
        all_closed = []
        for uid, engine in user_engines.items():
            data = engine._load_positions_data()
            for c in data.get("closed", []):
                c["user_id"] = uid
                all_closed.append(c)
        all_closed.sort(key=lambda x: x.get("exit_date", ""), reverse=True)
        return jsonify({"success": True, "closed": all_closed})
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
LSE_ETF_TICKERS = ["GDGB", "VEUR", "ISF", "EMXU", "LTAM", "IUES", "COPX", "WSML", "CJPN"]

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
        base_sym = ticker.replace(".NS", "").replace(".BO", "").replace(".L", "")

        # Period → fetch days (display + RS lookback buffer)
        period_days_extra = {"1mo": 100, "3mo": 160, "6mo": 250, "1y": 430}
        fetch_days = period_days_extra[period]
        from_date = _date.today() - _timedelta(days=fetch_days)
        to_date = _date.today()

        stock_closes = None
        stock_volumes = None
        broker = _get_connected_broker()
        if broker:
            stock_closes = _kite_daily_closes(broker, base_sym, from_date, to_date)

        # Fallback to yfinance
        if stock_closes is None or len(stock_closes) < 5:
            extra_map = {"1mo": "4mo", "3mo": "1y", "6mo": "2y", "1y": "2y"}
            fetch_period = extra_map[period]
            if base_sym in ETF_YF_OVERRIDES:
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
    """Scan all ETFs: compute Z-score ranking + entry/exit signals."""
    try:
        broker = _get_connected_broker()
        kite = broker._kite if broker else None
        result = etf_engine.scan(kite=kite)
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})

        # Merge Z-score fields and re-rank by Z-score (authoritative strategy ranking)
        try:
            from data.etf_core_zscore_backtest import score_live
            zscore_list = score_live()
            zscore_map = {d["symbol"]: d for d in zscore_list}
            watchlist = result.get("watchlist", [])
            for item in watchlist:
                zd = zscore_map.get(item["symbol"], {})
                item["mom_score"] = zd.get("score")
                item["beta"]      = zd.get("beta")
                item["ret_12m"]   = zd.get("ret_12m")
                item["ret_3m"]    = zd.get("ret_3m")

            # Re-sort by Z-score rank so UI matches Telegram (strategy uses Z-score)
            scored   = [x for x in watchlist if x.get("mom_score") is not None]
            unscored = [x for x in watchlist if x.get("mom_score") is None]
            scored.sort(key=lambda x: -x["mom_score"])
            for i, item in enumerate(scored):
                item["rank"] = i + 1
                # Recompute signal for non-position items based on Z-score rank
                if not item.get("in_position") and not item.get("in_watchlist"):
                    item["signal"] = "ENTRY" if item["rank"] <= 5 else "RANKED"
            for item in unscored:
                item["rank"] = None
            result["watchlist"] = scored + unscored
        except Exception:
            pass

        # Remove RS63/RSI/IBS-based signals — Z-score rank is the only signal now
        result["entry_signals"]   = []
        result["exit_signals"]    = []
        result["reentry_signals"] = []

        return jsonify({"success": True, **result})
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
            from data.etf_core_zscore_backtest import score_live
            from data.notifier import format_etf_zscore_alert
            etf_ranked = score_live()
            etf_msg = format_etf_zscore_alert(etf_ranked)
            if etf_msg:
                # Dedup: fire when top-5 symbols OR scores change (rounded to 1dp)
                print(f"[ETF scheduler] ETF Z-Score — alerting")
                send_message(etf_msg)
            else:
                print(f"[ETF scheduler] ETF Z-Score no signals")
                last_etf_key = None
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

                mom20_result = live_signals_scanner.scan_entry_signals(force_refresh=True)

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


# ============ User Registry ============

from data.user_registry import (
    load_users, add_user, get_user, update_user, delete_user, ensure_all_dirs,
    mom20_portfolio_path, mom20_history_path,
    etf_positions_path, etf_history_path,
    baskets_dir, trade_books_dir,
)
from data.mom20_basket import generate_basket, to_zerodha_csv, parse_trade_book, sync_portfolio_from_trades

ensure_all_dirs()


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
    except Exception as e:
        return jsonify({"success": False, "error": f"signals unavailable: {e}"})

    # Load user's current portfolio
    pf_path = mom20_portfolio_path(user_id)
    try:
        with open(pf_path) as f:
            portfolio = json.load(f)
    except Exception:
        portfolio = {"status": "empty", "basket": []}

    basket_data = generate_basket(user, signals, portfolio)
    return jsonify({"success": True, **basket_data})


@app.route("/api/portfolio-users/<user_id>/mom20-basket/download", methods=["GET"])
def api_mom20_basket_download(user_id):
    """Generate and download Zerodha basket CSV for this user."""
    import flask
    user = get_user(user_id)
    if not user:
        return jsonify({"success": False, "error": "user not found"})

    try:
        result = live_signals_scanner.scan_entry_signals(force_refresh=False)
        signals = result.get("mom20_signals", [])
    except Exception as e:
        return jsonify({"success": False, "error": f"signals unavailable: {e}"})

    pf_path = mom20_portfolio_path(user_id)
    try:
        with open(pf_path) as f:
            portfolio = json.load(f)
    except Exception:
        portfolio = {"status": "empty", "basket": []}

    basket_data = generate_basket(user, signals, portfolio)
    csv_content = to_zerodha_csv(basket_data)

    # Save a copy locally
    import datetime
    fname = f"mom20_{datetime.date.today().isoformat()}.csv"
    save_path = os.path.join(baskets_dir(user_id), fname)
    os.makedirs(baskets_dir(user_id), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(csv_content)

    return flask.Response(
        csv_content,
        mimetype="text/csv",
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

    updated = sync_portfolio_from_trades(portfolio, trades, basket_data)

    # Save updated portfolio
    tmp = pf_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(updated, f, indent=2)
    os.replace(tmp, pf_path)

    # Save trade book file
    import datetime
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
        "rebalance_date":    datetime.date.today().isoformat(),
        "trade_book_uploaded": datetime.datetime.now().isoformat(),
        "trades_parsed":     len(trades),
        "buys":  [t for t in trades if t["action"] == "BUY"],
        "sells": [t for t in trades if t["action"] == "SELL"],
        "status": "synced",
    })
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return jsonify({"success": True, "trades_synced": len(trades),
                    "portfolio_size": len(updated.get("basket", []))})


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

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG_MODE,
        threaded=True
    )


if __name__ == "__main__":
    run_server()
