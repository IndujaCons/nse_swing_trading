"""
User Registry — Multi-user portfolio management
================================================
Each user has their own data directory under data_store/{user_id}/
with separate Mom20 and ETF position/history files.
"""

import os
import json
import re
from datetime import date

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_STORE = os.path.join(_BASE_DIR, "data_store")
USERS_FILE = os.path.join(DATA_STORE, "users.json")

_USER_DEFAULT = {
    "id": "",
    "name": "",
    "strategies": {
        "mom20": {"active": False, "capital": 0},
        "etf":   {"active": False, "capital": 0},
    }
}


# ── User CRUD ──────────────────────────────────────────────────────────────────

def load_users() -> list:
    if not os.path.exists(USERS_FILE):
        return []
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_users(users: list):
    os.makedirs(DATA_STORE, exist_ok=True)
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)


def get_user(user_id: str) -> dict | None:
    return next((u for u in load_users() if u["id"] == user_id), None)


def add_user(name: str, mom20_capital: int = 0, etf_capital: int = 0) -> dict:
    users = load_users()

    # Generate unique slug id
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    existing = {u["id"] for u in users}
    uid, i = slug, 2
    while uid in existing:
        uid = f"{slug}_{i}"; i += 1

    user = {
        "id": uid,
        "name": name,
        "created": date.today().isoformat(),
        "strategies": {
            "mom20": {"active": mom20_capital > 0, "capital": mom20_capital},
            "etf":   {"active": etf_capital  > 0, "capital": etf_capital},
        }
    }
    users.append(user)
    save_users(users)
    _init_user_dir(uid)
    return user


def update_user(user_id: str, mom20_capital: int = None, etf_capital: int = None) -> dict | None:
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            if mom20_capital is not None:
                u["strategies"]["mom20"]["capital"] = mom20_capital
                u["strategies"]["mom20"]["active"]  = mom20_capital > 0
            if etf_capital is not None:
                u["strategies"]["etf"]["capital"] = etf_capital
                u["strategies"]["etf"]["active"]  = etf_capital > 0
            save_users(users)
            return u
    return None


def delete_user(user_id: str) -> bool:
    """Remove user from registry. Renames data dir to {id}_deleted for safety."""
    import shutil
    from datetime import date as _date
    users = load_users()
    remaining = [u for u in users if u["id"] != user_id]
    if len(remaining) == len(users):
        return False  # not found
    save_users(remaining)
    d = user_dir(user_id)
    if os.path.exists(d):
        archive = d + "_deleted_" + _date.today().isoformat()
        # avoid collision if archived twice same day
        suffix, i = archive, 2
        while os.path.exists(suffix):
            suffix = f"{archive}_{i}"; i += 1
        shutil.move(d, suffix)
    return True


# ── Per-user file paths ────────────────────────────────────────────────────────

def user_dir(user_id: str) -> str:
    return os.path.join(DATA_STORE, user_id)


def mom20_portfolio_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "mom20_portfolio.json")


def mom20_history_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "mom20_history.json")


def etf_positions_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "etf_positions.json")


def etf_history_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "etf_history.json")


def baskets_dir(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "baskets")


def trade_books_dir(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "trade_books")


# ── Init ───────────────────────────────────────────────────────────────────────

def _init_user_dir(user_id: str):
    """Create directory structure and empty JSON files for a new user."""
    os.makedirs(baskets_dir(user_id),     exist_ok=True)
    os.makedirs(trade_books_dir(user_id), exist_ok=True)

    defaults = {
        mom20_portfolio_path(user_id): {"status": "empty", "basket": [], "capital": 0},
        mom20_history_path(user_id):   [],
        etf_positions_path(user_id):   [],
        etf_history_path(user_id):     [],
    }
    for path, default in defaults.items():
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(default, f, indent=2)


def ensure_all_dirs():
    """Call on startup — ensures all registered users have their directories."""
    for u in load_users():
        _init_user_dir(u["id"])
