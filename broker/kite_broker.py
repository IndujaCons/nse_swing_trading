"""
Zerodha Kite Integration (Paper Trading Mode)
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    KITE_API_KEY, KITE_API_SECRET, KITE_REDIRECT_URL,
    PAPER_TRADING_ONLY, DATA_STORE_PATH
)


class PaperTrade:
    """Represents a paper trade."""

    def __init__(self, ticker: str, action: str, price: float, quantity: int = 1):
        self.ticker = ticker
        self.action = action  # 'BUY' or 'SELL'
        self.price = price
        self.quantity = quantity
        self.timestamp = datetime.now().isoformat()
        self.order_id = f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S')}-{ticker}"

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "ticker": self.ticker,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp,
            "status": "EXECUTED"
        }


class KiteBroker:
    """
    Zerodha Kite broker integration.
    Currently supports paper trading only - live trading is disabled.
    """

    def __init__(self):
        self.api_key = KITE_API_KEY
        self.api_secret = KITE_API_SECRET
        self.redirect_url = KITE_REDIRECT_URL
        self.access_token = None
        self.user_id = None
        self.user_name = None
        self._kite = None
        self._paper_trades_file = os.path.join(DATA_STORE_PATH, "paper_trades.json")
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(DATA_STORE_PATH, exist_ok=True)

    def get_login_url(self) -> str:
        """Get Zerodha login URL for OAuth flow."""
        if not self.api_key:
            return None
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"

    def exchange_token(self, request_token: str) -> bool:
        """
        Exchange request token for access token.

        Args:
            request_token: Token received from Zerodha callback

        Returns:
            True if successful, False otherwise
        """
        if not self.api_key or not self.api_secret:
            return False

        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.api_key)
            data = kite.generate_session(request_token, api_secret=self.api_secret)

            self.access_token = data["access_token"]
            self.user_id = data.get("user_id")
            self.user_name = data.get("user_name")
            self._kite = kite
            self._kite.set_access_token(self.access_token)
            return True
        except ImportError:
            # kiteconnect not installed - paper trading only
            return False
        except Exception as e:
            print(f"Token exchange error: {e}")
            return False

    def set_access_token(self, access_token: str):
        """Set access token directly (for session restoration)."""
        self.access_token = access_token
        if self.api_key:
            try:
                from kiteconnect import KiteConnect
                self._kite = KiteConnect(api_key=self.api_key)
                self._kite.set_access_token(access_token)
            except ImportError:
                pass

    def is_connected(self) -> bool:
        """Check if connected to Zerodha."""
        return self.access_token is not None

    def get_connection_status(self) -> Dict:
        """Get current connection status."""
        return {
            "connected": self.is_connected(),
            "user_id": self.user_id,
            "user_name": self.user_name,
            "api_key_configured": bool(self.api_key),
            "paper_mode": PAPER_TRADING_ONLY
        }

    def disconnect(self):
        """Disconnect from Zerodha."""
        self.access_token = None
        self.user_id = None
        self.user_name = None
        self._kite = None

    # Paper Trading Functions

    def _load_paper_trades(self) -> List[Dict]:
        """Load paper trades from file."""
        if not os.path.exists(self._paper_trades_file):
            return []

        try:
            with open(self._paper_trades_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_paper_trades(self, trades: List[Dict]):
        """Save paper trades to file."""
        with open(self._paper_trades_file, 'w') as f:
            json.dump(trades, f, indent=2)

    def execute_paper_trade(self, ticker: str, action: str, price: float, quantity: int = 1) -> Dict:
        """
        Execute a paper trade.

        Args:
            ticker: Stock ticker symbol
            action: 'BUY' or 'SELL'
            price: Current price
            quantity: Number of shares

        Returns:
            Trade confirmation dict
        """
        trade = PaperTrade(ticker, action.upper(), price, quantity)

        # Save to paper trades history
        trades = self._load_paper_trades()
        trades.append(trade.to_dict())
        self._save_paper_trades(trades)

        return trade.to_dict()

    def get_paper_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent paper trades."""
        trades = self._load_paper_trades()
        return trades[-limit:] if len(trades) > limit else trades

    def get_paper_portfolio(self) -> Dict:
        """
        Calculate current paper portfolio from trades.

        Returns:
            Dict with positions and summary
        """
        trades = self._load_paper_trades()
        positions = {}

        for trade in trades:
            ticker = trade["ticker"]
            if ticker not in positions:
                positions[ticker] = {
                    "ticker": ticker,
                    "quantity": 0,
                    "avg_price": 0,
                    "total_cost": 0
                }

            pos = positions[ticker]
            if trade["action"] == "BUY":
                new_qty = pos["quantity"] + trade["quantity"]
                new_cost = pos["total_cost"] + (trade["price"] * trade["quantity"])
                pos["quantity"] = new_qty
                pos["total_cost"] = new_cost
                pos["avg_price"] = new_cost / new_qty if new_qty > 0 else 0
            elif trade["action"] == "SELL":
                pos["quantity"] -= trade["quantity"]
                if pos["quantity"] <= 0:
                    pos["quantity"] = 0
                    pos["avg_price"] = 0
                    pos["total_cost"] = 0

        # Filter out closed positions
        active_positions = {k: v for k, v in positions.items() if v["quantity"] > 0}

        return {
            "positions": list(active_positions.values()),
            "total_trades": len(trades)
        }

    def clear_paper_trades(self):
        """Clear all paper trades."""
        self._save_paper_trades([])

    # Live Trading Functions (Disabled)

    def execute_live_trade(self, ticker: str, action: str, quantity: int) -> Dict:
        """
        Execute a live trade.
        Currently disabled - paper trading only.
        """
        if PAPER_TRADING_ONLY:
            raise RuntimeError("Live trading is disabled. Paper trading mode only.")

        if not self.is_connected():
            raise RuntimeError("Not connected to Zerodha")

        # Live trading implementation would go here
        # Currently disabled for safety
        raise NotImplementedError("Live trading not implemented")
