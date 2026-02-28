"""
Configuration settings for the RS Dashboard
"""

import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_STORE_PATH = os.path.join(BASE_DIR, "data_store")
CONFIG_FILE = os.path.join(DATA_STORE_PATH, "config.json")
DAILY_STATE_FILE = os.path.join(DATA_STORE_PATH, "daily_state.json")
SCREENER_CACHE_FILE = os.path.join(DATA_STORE_PATH, "screener_cache.json")
HAMMER_CACHE_FILE = os.path.join(DATA_STORE_PATH, "hammer_cache.json")
LIVE_SIGNALS_CACHE_FILE = os.path.join(DATA_STORE_PATH, "live_signals_cache.json")
LIVE_POSITIONS_FILE = os.path.join(DATA_STORE_PATH, "live_positions.json")
LIVE_SIGNALS_HISTORY_FILE = os.path.join(DATA_STORE_PATH, "live_signals_history.csv")
MOMENTUM_CACHE_FILE = os.path.join(DATA_STORE_PATH, "momentum_cache.json")
MOMENTUM_STATE_FILE = os.path.join(DATA_STORE_PATH, "momentum_state.json")
NSE_INSTRUMENTS_FILE = os.path.join(DATA_STORE_PATH, "nse_instruments.json")

# Default settings
DEFAULTS = {
    "stock_universe": 250,
    "rs_period": 63,
    "ema_period": 63,
    "cache_ttl_minutes": 15,
    "support_lookback_days": 120,
    "support_proximity_pct": 3.0,
    "hammer_body_ratio": 0.33,
    "hammer_lower_shadow_ratio": 2.0,
    "hammer_upper_shadow_max_pct": 0.10,
    "momentum_rsi2_threshold": 75,
    "live_signals_universe": 50,
}

# Flask settings
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 8080
DEBUG_MODE = True

# Zerodha Kite settings
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_REDIRECT_URL = f"http://{FLASK_HOST}:{FLASK_PORT}/api/login/callback"

# Trading mode (paper only - live disabled)
PAPER_TRADING_ONLY = True

# Index symbol
NIFTY_50_SYMBOL = "^NSEI"


def _ensure_data_dir():
    """Ensure data directory exists."""
    os.makedirs(DATA_STORE_PATH, exist_ok=True)


def load_config() -> dict:
    """Load configuration from file, or return defaults."""
    _ensure_data_dir()

    if not os.path.exists(CONFIG_FILE):
        return DEFAULTS.copy()

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Merge with defaults for any missing keys
        merged = DEFAULTS.copy()
        merged.update(config)
        return merged
    except (json.JSONDecodeError, IOError):
        return DEFAULTS.copy()


def save_config(config: dict):
    """Save configuration to file."""
    _ensure_data_dir()

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_setting(key: str):
    """Get a single setting value."""
    config = load_config()
    return config.get(key, DEFAULTS.get(key))


def update_setting(key: str, value):
    """Update a single setting and save."""
    config = load_config()
    config[key] = value
    save_config(config)


# For backward compatibility - these now read from config
def get_stock_universe():
    return get_setting("stock_universe")


def get_rs_period():
    return get_setting("rs_period")


def get_ema_period():
    return get_setting("ema_period")


def get_cache_ttl():
    return get_setting("cache_ttl_minutes")


def get_kite_users():
    """Discover KITE_USER{N}_* env vars. Returns list of user config dicts.

    Falls back to legacy KITE_API_KEY / KITE_API_SECRET if no numbered vars exist.
    """
    users = []
    n = 1
    while True:
        api_key = os.environ.get(f"KITE_USER{n}_API_KEY")
        if api_key is None:
            break
        users.append({
            "id": os.environ.get(f"KITE_USER{n}_ID", f"user{n}"),
            "name": os.environ.get(f"KITE_USER{n}_NAME", f"User{n}"),
            "api_key": api_key,
            "api_secret": os.environ.get(f"KITE_USER{n}_API_SECRET", ""),
        })
        n += 1

    if not users:
        # Legacy fallback
        legacy_key = os.environ.get("KITE_API_KEY", "")
        legacy_secret = os.environ.get("KITE_API_SECRET", "")
        if legacy_key:
            users.append({
                "id": "default",
                "name": "Default",
                "api_key": legacy_key,
                "api_secret": legacy_secret,
            })

    return users


# Legacy constants (use functions above for dynamic values)
RS_PERIOD = DEFAULTS["rs_period"]
EMA_PERIOD = DEFAULTS["ema_period"]
CACHE_TTL_MINUTES = DEFAULTS["cache_ttl_minutes"]
