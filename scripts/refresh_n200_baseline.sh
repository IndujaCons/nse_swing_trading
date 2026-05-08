#!/bin/bash
# ============================================================
# N200 Baseline Refresh — RS Screener
# Bulk-fetches 500 days × 200 N200 tickers + Nifty 50 + Nifty 200,
# pickles to data_store/cache/n200_baseline_<date>.pkl. The Live
# Signals scan reads this pkl and only fetches today's intraday bar
# on demand — turning a 30-60s scan into ~5-10s during market hours.
#
# Setup:
#   1. Ensure venv has yfinance + pandas installed
#   2. Add to crontab (crontab -e):
#       0 8,16 * * 1-5 /home/ubuntu/relative_strength/scripts/refresh_n200_baseline.sh \
#           >> /home/ubuntu/logs/n200_baseline.log 2>&1
#      08:00 IST — before 09:15 NSE open (warm cache for first user click).
#      16:00 IST — after 15:30 NSE close (capture today's final bar).
#
# Manual run (bootstrap or after a missed cron):
#   bash scripts/refresh_n200_baseline.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
if [ -x "$PROJECT_DIR/venv/bin/python3" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python3"
fi

TS="$(date '+%Y-%m-%d %H:%M:%S IST')"
echo "[$TS] Starting N200 baseline refresh in $PROJECT_DIR"

cd "$PROJECT_DIR" || exit 2
"$PYTHON" scripts/refresh_n200_baseline.py
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "[$TS] Baseline refresh complete ✓"
else
    echo "[$TS] Baseline refresh FAILED (exit $STATUS)"
fi
exit $STATUS
