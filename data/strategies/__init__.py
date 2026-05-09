"""Per-strategy compute functions extracted from data/live_signals_engine.py.

Each module exports a single `compute_*(...)` function that takes per-ticker
OHLCV data plus relevant context and returns a feature/signal dict (or None).
The big `scan_entry_signals()` loop is now ~30 lines that orchestrates these.

This is purely a code-organization refactor — no signal logic was changed.
The pre/post-loop cross-sectional ranking (Mom20 Z-score, Alpha20 ranking,
RS-IBD percentile filter, RS63 volume ranking) stays in scan_entry_signals
since it operates on the collected raw features across all tickers.
"""
