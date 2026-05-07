#!/bin/bash
# ============================================================
# Synth Sector Index Refresh — RS Screener
# Rebuilds the 5 synthetic sector close series used by Phase 1
# sector ranking (HEALTHCARE / OIL & GAS / DEFENCE / INDIA MFG /
# CONS DURABLES — yfinance has no series for these).
#
# Output: data/cache/synth_<name>_<YYYY-MM-DD>.pkl  (one per index)
#
# These pkls are gitignored, so the deploy box has no copies after
# rsync. Without them, score_live_sectors() drops 5 sectors and the
# top-5 sector list (used by Mom20 ETF top-up + sector dashboard)
# becomes wrong. Run this daily before market open.
#
# Setup:
#   1. Ensure the venv has yfinance + pandas + requests installed
#   2. Add to crontab (crontab -e):
#        15 8 * * 1-5 /path/to/scripts/refresh_synth.sh >> /path/to/logs/synth_refresh.log 2>&1
#      08:15 IST — well before 09:15 NSE open, after yfinance has
#      yesterday's close stable.
#
# Manual run (bootstrap or after a missed cron):
#   bash scripts/refresh_synth.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
if [ -x "$PROJECT_DIR/venv/bin/python3" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python3"
fi

TS="$(date '+%Y-%m-%d %H:%M:%S IST')"
echo "[$TS] Starting synth refresh in $PROJECT_DIR"

cd "$PROJECT_DIR" || exit 2
"$PYTHON" data/synth_sector_index.py --all
STATUS=$?

# Tidy up old daily snapshots (keep last 7 days per index)
find "$PROJECT_DIR/data/cache" -name "synth_*.pkl" -mtime +7 -delete 2>/dev/null

if [ $STATUS -eq 0 ]; then
    echo "[$TS] Synth refresh complete ✓"
else
    echo "[$TS] Synth refresh FAILED (exit $STATUS)"
fi
exit $STATUS
