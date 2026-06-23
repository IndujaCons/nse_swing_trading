# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Response Style

- Never open with filler phrases ("Great question!", "Of course!", "Certainly!"). Start with the actual answer.
- Match response length to task complexity. Simple questions get direct, short answers. Complex tasks get full, detailed responses. Never pad with restatements of the question or closing sentences that repeat what you just said.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- Before any significant task, show 2–3 approaches. Wait for the user to choose before proceeding.
- If uncertain about any fact, statistic, date, or technical detail: say so explicitly. Never fill gaps with plausible-sounding information.
- For architecture decisions, complex debugging, or non-trivial features: work through the problem step by step before writing any code. Show your reasoning. Identify where you're uncertain. Then implement.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Only modify files, functions, and lines directly related to the current task. If you notice something worth fixing elsewhere, mention it in a note at the end. Do not touch it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Before altering content already created (rewriting sections, removing paragraphs, restructuring flow, changing tone): stop. Describe exactly what you're about to change and why. Wait for confirmation before proceeding.

The test: Every changed line should trace directly to the user's request.

## 4. Confirmations

**Stop before irreversible or external actions. Always.**

Before deleting any file, overwriting existing code, dropping database records, or removing dependencies:
- Stop. List exactly what will be affected.
- Ask for explicit confirmation. "You mentioned this earlier" is NOT confirmation.
- Only proceed after the user says yes in the current message.

The following require explicit in-session confirmation, no exceptions:
- Deploying or pushing to any environment
- Running migrations or schema changes
- Sending any external API call
- Executing any command with irreversible side effects

Never send, post, publish, share, or schedule anything on the user's behalf without explicit confirmation in the current message. This includes emails, calendar invites, document shares, or any external action.

## 5. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## Strategy state lives in memory, not code

Frozen strategy configs, backtest results, and rejected variants are tracked in `~/.claude/projects/-Users-jay-dev-relative-strength/memory/` (MEMORY.md is the index). When asked about a strategy's current parameters or performance, read memory first — the code may contain prior or experimental params. Files like `memory/rejected_*.md` document tested-and-dropped variants; check them before re-suggesting an idea.

## Common commands

Backtests (all support `--refresh` to re-download data; otherwise read pickle cache from `data/cache/`):

```bash
python3 mom15_pit_report.py                              # Mom15: Nifty200, top15, 2-monthly
python3 mom15_pit_report.py --mom20                      # Mom20: Nifty200, top20, monthly (FROZEN production)
python3 mom15_pit_report.py --n500                       # Mom500: Nifty500 (research only)
python3 mom15_pit_report.py --qqq                        # QQQ universe (US)
python3 mom15_pit_report.py --sp500                      # S&P 500 universe
python3 mom15_pit_report.py --overflow                   # Overflow basket
python3 mom15_pit_report.py --no-regime --beta-cap 1.0   # Override flags

python3 rs63_pit_report.py                               # RS63 satellite (Vivek Bajaj adapted)
python3 rs63_pit_report.py --max-daily-entries 2         # Operating variant
python3 rs63_pit_report.py --end-date 2026-04-02         # Freeze evaluation date

python3 data/etf_core_zscore_backtest.py                 # ETF Z-score monthly rotation (FROZEN)
python3 data/etf_core_zscore_backtest.py --midmonth      # Rebal on 15th instead of 1st

python3 data/goldm_backtest.py                           # GOLDM MCX intraday ORB
```

Run the dashboard:
```bash
python3 run.py                                           # Flask UI on $FLASK_PORT (default 8080)
```

Deploy / ops:
```bash
bash deploy/redeploy.sh <EC2_IP>                         # rsync + systemctl restart
bash scripts/backup_data.sh                              # S3 sync data_store/ (set RS_S3_BUCKET)
```

There is no test suite, lint config, or build step. `requirements.txt` is the dependency list; install into `venv/`.

## Architecture

**Two layers, loosely coupled**:

1. **Backtest scripts** at the repo root (`mom15_pit_report.py`, `rs63_pit_report.py`, `data/etf_core_zscore_backtest.py`, `data/goldm_backtest.py`). Each is self-contained: it loads PIT universe data, fetches/caches prices via `yfinance`, simulates with the Zerodha charges formula in this file (every script has its own `calc_charges`), and prints a markdown report. They share no runtime state with the UI.

2. **Flask UI** (`run.py` → `ui/app.py`) wires together engine classes from `data/`:
   - `screener_engine.py`, `live_signals_engine.py`, `momentum_engine.py`, `etf_engine.py`, `goldm_engine.py`
   - Per-user instances keyed by `user_id`; broker access through `broker/kite_broker.py`
   - State persisted as JSON in `data_store/` (gitignored) — files like `live_positions_{user_id}.json`, `mom20_portfolio.json`, etc.
   - `PAPER_TRADING_ONLY = True` is hard-set in `config/settings.py`; live order placement is disabled

**Multi-user model**: `config/settings.py:get_kite_users()` discovers users from `KITE_USER{N}_API_KEY` / `_NAME` / `_ID` / `_API_SECRET` env vars in `.env` (with legacy single-user fallback). Each user gets their own engine instances and per-user JSON files in `data_store/{user_id}/`. Templates for those files live in `data_store_templates/` and ARE committed; actual user data is gitignored.

**PIT (point-in-time) universe data** is in `nse_const/`:
- `nifty200_pit.json`, `nifty500_pit.json`, `qqq_pit.json`, `sp500_pit.json` — built from index reconstitution PDFs by the `build_pit_*.py` scripts in the same directory
- All backtests pass through `get_pit_universe(pit_data, day)` so survivorship bias is avoided
- Ticker renames (e.g. ZOMATO→ETERNAL) handled by `TICKER_ALIASES` in `data/momentum_backtest.py`

**Charges and tax**: every backtest's `calc_charges()` implements the Zerodha formula below; `calc_tax()` applies 20% STCG with non-STT charges deductible. When changing fee logic, update all `calc_charges` copies (no shared module).

**Data caches** in `data/cache/*.pkl` are large pickled DataFrames keyed by ticker → daily OHLCV. Don't commit; `--refresh` rebuilds them.

## Zerodha Brokerage (Equity Delivery, NSE)

Brokerage: **₹0** (free for equity delivery)

### Statutory Charges Formula

| Charge | Rate | Applied On |
|--------|------|------------|
| STT | 0.1% | Total turnover (buy + sell) |
| Exchange txn | 0.00307% | Total turnover (buy + sell) |
| SEBI | 0.0001% | Total turnover (buy + sell) |
| Stamp duty | 0.015% | Buy side only |
| GST | 18% | Exchange txn + SEBI charges |

### Round-trip Cost Formula

For a buy/sell combo with buy turnover `B` and sell turnover `S`:

```
total_turnover = B + S
stt            = 0.001    * total_turnover
exchange       = 0.0000307 * total_turnover
sebi           = 0.000001  * total_turnover
stamp          = 0.00015   * B
gst            = 0.18      * (exchange + sebi)

total_charges  = stt + exchange + sebi + stamp + gst
```

### Example: ₹2L position (breakeven trade)

Buy ₹2,00,000 + Sell ₹2,00,000 = turnover ₹4,00,000

- STT: ₹400
- Exchange: ₹12.28
- SEBI: ₹0.40
- Stamp: ₹30
- GST: ₹2.28
- **Total: ~₹445 per round trip**
- **As % of position: ~0.22%**

### Verified Against Zerodha Calculator

**Trade 1:** Buy 1000, Sell 1100, Qty 800 (turnover 16.8L)
- Charges: ₹1,862.85 (formula gives ₹1,862.85)

**Trade 2:** Buy 1000, Sell 1100, Qty 200 (turnover 4.2L)
- Charges: ₹465.71 (formula gives ₹465.71)

### Tax

- **STCG (Short-Term Capital Gains):** 20% on net profit
- Holding period < 12 months = short-term

### Expense Deduction for STCG

All charges **except STT** can be claimed as expenses to reduce taxable profit:

```
claimable_expenses = exchange + sebi + stamp + gst   (NOT stt)
taxable_profit     = gross_profit - stt - claimable_expenses - claimable_expenses_as_deduction
                   = gross_profit - total_charges
                   (but STT is NOT deductible, so effectively:)

taxable_profit     = gross_profit - (exchange + sebi + stamp + gst)
stcg_tax           = 0.20 * taxable_profit
net_profit         = gross_profit - total_charges - stcg_tax
```

- **STT** is paid but NOT deductible under Income Tax Act
- **Exchange txn, SEBI, Stamp duty, GST** are all deductible expenses
