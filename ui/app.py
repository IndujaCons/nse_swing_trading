"""
Flask Application for RS Dashboard
"""

import os
import sys
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

for _u in kite_users:
    brokers[_u["id"]] = KiteBroker(
        user_id=_u["id"], name=_u["name"],
        api_key=_u["api_key"], api_secret=_u["api_secret"])
    user_engines[_u["id"]] = LiveSignalsEngine(user_id=_u["id"])

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
    """Get cached J+K scan results."""
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


@app.route("/api/live-signals/refresh", methods=["POST"])
def refresh_live_signals():
    """Force re-scan J+K entry signals."""
    global live_signals_refresh_in_progress

    with live_signals_refresh_lock:
        if live_signals_refresh_in_progress:
            return jsonify({
                "success": False,
                "error": "Refresh already in progress"
            }), 409

        live_signals_refresh_in_progress = True

    try:
        data = live_signals_scanner.scan_entry_signals(
            force_refresh=True,
            progress_callback=update_live_signals_progress
        )

        return jsonify({
            "success": True,
            "data": data,
            "message": "Live signals refreshed successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
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


@app.route("/api/live-signals/positions", methods=["GET"])
def get_live_positions():
    """Get active positions merged across all users."""
    try:
        all_positions = []
        for uid, engine in user_engines.items():
            for p in engine.get_positions():
                if "user_id" not in p:
                    p["user_id"] = uid
                all_positions.append(p)
        return jsonify({"success": True, "positions": all_positions})
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

    if strategy not in ("J", "T"):
        return jsonify({"success": False, "error": "strategy must be J or T"}), 400

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
    # Store Nifty close at entry for drop shield (J and K)
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period="5d")
        if not nifty.empty:
            metadata["nifty_at_entry"] = float(nifty["Close"].iloc[-1])
    except Exception:
        pass
    result = engine.add_position(
        ticker, strategy, price, amount, support, ibs, metadata)

    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400

    return jsonify({"success": True, "position": result,
                    "order_id": order_result.get("order_id")})


@app.route("/api/live-signals/exits", methods=["GET"])
def get_live_exit_signals():
    """Check exit conditions for all active positions (merged across users)."""
    try:
        all_exits = []
        for uid, engine in user_engines.items():
            exits = engine.check_exit_signals()
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

    if strategy not in ("J", "T"):
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
    if universe not in (50, 100):
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


@app.route("/api/momentum/backtest-portfolio", methods=["POST"])
def run_portfolio_backtest():
    """Run portfolio-level backtest (10L capital, J+K strategies)."""
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
    if universe not in (50, 100):
        universe = 50
    capital_lakhs = int(data.get("capital_lakhs", 10))
    if capital_lakhs not in (10, 20, 50, 80, 100):
        capital_lakhs = 10
    per_stock = int(data.get("per_stock", 50000))
    if per_stock not in (50000, 100000, 200000, 500000):
        per_stock = 50000
    strategies = data.get("strategies", ["J", "T"])
    valid_strats = {"J", "T"}
    strategies = [s for s in strategies if s in valid_strats]
    if not strategies:
        strategies = ["J", "T"]
    entries_per_day = int(data.get("entries_per_day", 1))
    if entries_per_day not in (1, 2):
        entries_per_day = 1
    three_stage_exit = bool(data.get("three_stage_exit", True))

    try:
        backtester = MomentumBacktester()
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
            no_gap_down=True
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
        if val in [50, 100, 150] and val != config.get("live_signals_universe"):
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

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG_MODE,
        threaded=True
    )


if __name__ == "__main__":
    run_server()
