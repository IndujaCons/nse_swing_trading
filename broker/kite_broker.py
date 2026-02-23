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
    KITE_REDIRECT_URL, PAPER_TRADING_ONLY, DATA_STORE_PATH
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

    def __init__(self, user_id="default", name="", api_key="", api_secret=""):
        self.config_user_id = user_id
        self.config_name = name
        self.api_key = api_key
        self.api_secret = api_secret
        self.redirect_url = KITE_REDIRECT_URL
        self.access_token = None
        self.user_id = None
        self.user_name = None
        self._kite = None
        suffix = f"_{user_id}" if user_id != "default" else ""
        self._paper_trades_file = os.path.join(DATA_STORE_PATH, f"paper_trades{suffix}.json")
        self._session_file = os.path.join(DATA_STORE_PATH, f"kite_session{suffix}.json")
        self._ensure_data_dir()
        self._restore_session()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(DATA_STORE_PATH, exist_ok=True)

    def _save_session(self):
        """Persist access token to disk so it survives server restarts."""
        data = {
            "access_token": self.access_token,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "saved_at": datetime.now().isoformat(),
        }
        try:
            with open(self._session_file, 'w') as f:
                json.dump(data, f)
        except IOError:
            pass

    def _restore_session(self):
        """Load saved access token from disk (if any)."""
        if not os.path.exists(self._session_file):
            return
        try:
            with open(self._session_file, 'r') as f:
                data = json.load(f)
            token = data.get("access_token")
            if token:
                self.set_access_token(token)
                self.user_id = data.get("user_id")
                self.user_name = data.get("user_name")
        except (json.JSONDecodeError, IOError):
            pass

    def _clear_session(self):
        """Remove saved session file."""
        try:
            if os.path.exists(self._session_file):
                os.remove(self._session_file)
        except IOError:
            pass

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
            self._save_session()
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
            "config_user_id": self.config_user_id,
            "config_name": self.config_name,
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
        self._clear_session()

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

    # Live Order Placement

    def place_buy_order(self, ticker: str, quantity: int, price: float = None) -> Dict:
        """
        Place a CNC BUY order on Zerodha.

        Args:
            ticker: NSE trading symbol (e.g. "INFY", "M&M")
            quantity: Number of shares
            price: Limit price. If None, places MARKET order.

        Returns:
            Dict with order_id on success.

        Raises:
            RuntimeError if not connected.
        """
        if not self.is_connected() or self._kite is None:
            raise RuntimeError("Not connected to Zerodha. Please login first.")

        order_params = {
            "variety": self._kite.VARIETY_REGULAR,
            "exchange": self._kite.EXCHANGE_NSE,
            "tradingsymbol": ticker,
            "transaction_type": self._kite.TRANSACTION_TYPE_BUY,
            "quantity": quantity,
            "product": self._kite.PRODUCT_CNC,
        }

        if price is not None:
            order_params["order_type"] = self._kite.ORDER_TYPE_LIMIT
            order_params["price"] = price
        else:
            order_params["order_type"] = self._kite.ORDER_TYPE_MARKET

        order_id = self._kite.place_order(**order_params)
        return {"order_id": order_id, "status": "PLACED"}

    def place_sell_order(self, ticker: str, quantity: int, price: float = None) -> Dict:
        """
        Place a CNC SELL order on Zerodha.

        Args:
            ticker: NSE trading symbol
            quantity: Number of shares
            price: Limit price. If None, places MARKET order.

        Returns:
            Dict with order_id on success.

        Raises:
            RuntimeError if not connected.
        """
        if not self.is_connected() or self._kite is None:
            raise RuntimeError("Not connected to Zerodha. Please login first.")

        order_params = {
            "variety": self._kite.VARIETY_REGULAR,
            "exchange": self._kite.EXCHANGE_NSE,
            "tradingsymbol": ticker,
            "transaction_type": self._kite.TRANSACTION_TYPE_SELL,
            "quantity": quantity,
            "product": self._kite.PRODUCT_CNC,
        }

        if price is not None:
            order_params["order_type"] = self._kite.ORDER_TYPE_LIMIT
            order_params["price"] = price
        else:
            order_params["order_type"] = self._kite.ORDER_TYPE_MARKET

        order_id = self._kite.place_order(**order_params)
        return {"order_id": order_id, "status": "PLACED"}
