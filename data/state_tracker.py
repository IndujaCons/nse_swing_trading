"""
State tracker for daily EMA position snapshots
"""

import json
import os
from datetime import datetime, date
from typing import Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import DAILY_STATE_FILE


class StateTracker:
    """Tracks daily EMA positions for state transition detection."""

    def __init__(self):
        self.state_file = DAILY_STATE_FILE
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    def load_state(self) -> Dict:
        """Load the previous day's state from file."""
        if not os.path.exists(self.state_file):
            return {"date": None, "stocks": {}}

        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"date": None, "stocks": {}}

    def save_state(self, stocks_state: Dict[str, bool]):
        """
        Save today's EMA positions.

        Args:
            stocks_state: Dict mapping ticker to above_ema boolean
        """
        state = {
            "date": date.today().isoformat(),
            "stocks": stocks_state
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_yesterday_position(self, ticker: str) -> Optional[bool]:
        """
        Get yesterday's EMA position for a stock.

        Returns:
            True if was above EMA, False if below, None if no data
        """
        state = self.load_state()

        # Check if state is from a previous day
        if state["date"] == date.today().isoformat():
            # State is from today - we need yesterday's data
            # In this case, return None as we don't have yesterday
            return None

        return state["stocks"].get(ticker)

    def is_state_stale(self) -> bool:
        """Check if state data needs to be refreshed."""
        state = self.load_state()
        if state["date"] is None:
            return True

        state_date = datetime.fromisoformat(state["date"]).date()
        return state_date < date.today()

    def get_state_date(self) -> Optional[str]:
        """Get the date of the stored state."""
        state = self.load_state()
        return state.get("date")
