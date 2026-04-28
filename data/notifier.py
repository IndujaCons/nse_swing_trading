"""
Telegram Notifier — ETF Core Signal Alerts
==========================================
Sends entry/exit notifications to a Telegram chat whenever
ETF Core scan detects actionable signals.

Config (via .env):
    TELEGRAM_BOT_TOKEN  — from @BotFather
    TELEGRAM_CHAT_ID    — your personal chat ID (get via /get_chat_id bot or api/getUpdates)
"""

import os
import requests


def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "")


def _chat_ids() -> list[str]:
    """Returns list of all configured chat IDs (supports TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID_2)."""
    ids = []
    for key in ("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID_2"):
        v = os.getenv(key, "").strip()
        if v:
            ids.append(v)
    return ids


def send_message(text: str) -> bool:
    """Send a plain text / HTML message to all configured Telegram chats.
    Returns True if at least one message succeeded."""
    token = _token()
    chat_ids = _chat_ids()
    if not token or not chat_ids:
        print("[notifier] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    any_ok = False
    for chat_id in chat_ids:
        try:
            r = requests.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if not r.ok:
                print(f"[notifier] Telegram error {r.status_code} for {chat_id}: {r.text[:200]}")
            else:
                any_ok = True
        except Exception as e:
            print(f"[notifier] Request failed for {chat_id}: {e}")
    return any_ok


def format_etf_alert(scan_result: dict) -> str | None:
    """
    Format ETF Core scan result as a Telegram HTML message.
    Returns None if there are no actionable signals (no entry / exit / re-entry).
    """
    entry_signals = scan_result.get("entry_signals", [])
    exit_signals = scan_result.get("exit_signals", [])
    reentry_signals = scan_result.get("reentry_signals", [])

    if not entry_signals and not exit_signals and not reentry_signals:
        return None

    scan_time = scan_result.get("scan_time", "")
    vix = scan_result.get("vix")
    vix_str = f" | VIX {vix:.1f}" if vix else ""
    lines = [f"<b>📊 ETF Core Signal</b> — {scan_time}{vix_str}"]

    if entry_signals:
        lines.append("")
        lines.append("🟢 <b>ENTRY</b>")
        for s in entry_signals:
            intl_note = " ⚠️ INTL CAP" if s.get("intl_blocked") else ""
            lines.append(
                f"  • <b>{s['symbol']}</b> (#{s['rank']}) "
                f"RS63={s['rs63']:.2f}% @ ₹{s['price']}{intl_note}"
            )

    if reentry_signals:
        lines.append("")
        lines.append("🔵 <b>RE-ENTRY</b>")
        for s in reentry_signals:
            lines.append(
                f"  • <b>{s['symbol']}</b> (#{s['rank']}) "
                f"RS63={s['rs63']:.2f}% @ ₹{s['price']}"
            )

    if exit_signals:
        lines.append("")
        lines.append("🔴 <b>EXIT</b>")
        for s in exit_signals:
            reasons = ", ".join(s["reasons"])
            lines.append(
                f"  • <b>{s['symbol']}</b> @ ₹{s['price']} — {reasons}"
            )

    return "\n".join(lines)


def format_rs63_alert(scan_result: dict, duration_map: dict | None = None) -> str | None:
    """
    Format RS63 live signals scan result as a compact Telegram message.
    duration_map: {ticker: hours_present} — shows 'new', '1h', '2h', '3h+' column.
    Returns None if no RS63 signals are present.
    """
    signals = scan_result.get("rs63_signals", [])
    if not signals:
        return None

    from datetime import datetime, timezone, timedelta
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %H:%M")

    def _dur_label(ticker):
        if not duration_map:
            return ""
        h = duration_map.get(ticker, 0)
        if h < 0.9:
            return "new"
        elif h < 2:
            return "1h"
        elif h < 3:
            return "2h"
        else:
            return "3h+"

    signals = sorted(signals, key=lambda s: s.get('stop_pct', 99))

    lines = [f"<b>📈 RS63</b> {len(signals)}sig | {ist}", "<code>"]
    lines.append(f"{'#':<2}{'Ticker':<9} {'Px':>5} {'D/1H':>9} {'RSI':>3} {'Stp%':>4} {'Age':>3}")
    lines.append("─" * 38)
    for i, s in enumerate(signals, 1):
        marker = "★ " if i <= 2 else "  "
        ticker = s['ticker'][:9]
        price  = str(int(s['price']))
        rs63   = f"{s['rs63']:.1f}"
        rs1h_v = s.get('rs63_1h')
        rs1h   = f"{rs1h_v:+.1f}" if rs1h_v is not None else "—"
        d1h    = f"{rs63}/{rs1h}"
        rsi    = str(int(round(s.get('rsi', 0))))
        sp     = s.get('stop_pct')
        stp    = f"{sp:.1f}%" if sp is not None else "  —"
        dur    = _dur_label(s['ticker'])
        lines.append(f"{marker}{i:<2}{ticker:<9} {price:>5} {d1h:>9} {rsi:>3} {stp:>4} {dur:>3}")
        lines.append("")
    lines.append("</code>")
    return "\n".join(lines)


def format_rs63_exit_alert(exit_signals: list) -> str | None:
    """Format RS63 exit signals as a Telegram HTML message. Returns None if no exits."""
    # Filter RS63 exits only
    rs63_exits = [e for e in exit_signals if e.get("strategy") == "RS63"]
    if not rs63_exits:
        return None

    from datetime import datetime, timezone, timedelta
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🚨 RS63 Exit Signal</b> — {ist} IST"]
    for e in rs63_exits:
        pnl = e.get("pnl_pct", 0)
        pnl_str = f"+{pnl:.1f}%" if pnl >= 0 else f"{pnl:.1f}%"
        pnl_color = "🟢" if pnl >= 0 else "🔴"
        lines.append(
            f"  {pnl_color} <b>{e['ticker']}</b> @ ₹{e.get('current_price', '?')} "
            f"({pnl_str}) — {e.get('reason', '?')}"
        )
    return "\n".join(lines)


def format_etf_zscore_alert(ranked: list) -> str | None:
    """Format ETF Z-Score top-10 ranking as a Telegram message.
    ranked: output of etf_core_zscore_backtest.score_live()
    Entry if rank≤5, hold if rank≤10 (frozen config)."""
    if not ranked:
        return None

    from datetime import datetime, timezone, timedelta
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %H:%M")
    ENTRY_THRESH, HOLD_THRESH = 5, 10

    # Build lookup for UCITS substitutes to show as grey sub-lines
    UCITS_SUBS = {"SOXX": "SEMI.L", "EWY": "CSKR.L"}
    sub_map = {s["symbol"]: s for s in ranked}

    lines = [f"<b>🌐 ETF Z-Score Top 10</b> | {ist}", "<code>"]
    lines.append(f"{'#':<3} {'ETF':<12} {'Close':>7} {'12m':>6} {'3m':>6} {'Score':>5}")
    lines.append("─" * 44)
    shown = 0
    for s in ranked:
        if s["symbol"] in UCITS_SUBS.values():
            continue   # skip UCITS entries from main ranking
        if shown >= 10:
            break
        shown += 1
        tag = "★" if s["rank"] <= ENTRY_THRESH else ("◀" if s["rank"] <= HOLD_THRESH else " ")
        lines.append(
            f"{s['rank']:<3} {s['symbol']:<12} {s['price']:>7,.0f}"
            f" {s['ret_12m']:>+5.0f}% {s['ret_3m']:>+5.0f}%"
            f" {s['score']:>5.2f} {tag}"
        )
        # Append UCITS substitute as grey reference line
        sub_sym = UCITS_SUBS.get(s["symbol"])
        if sub_sym and sub_sym in sub_map:
            u = sub_map[sub_sym]
            lines.append(
                f"  └ {u['symbol']:<10} {u['price']:>7,.2f}"
                f" {u['ret_12m']:>+5.0f}% {u['ret_3m']:>+5.0f}%"
                f" {'(UCITS)':>8}"
            )
    lines.append("</code>")
    lines.append(f"★ entry (rank≤{ENTRY_THRESH})  ◀ hold (rank≤{HOLD_THRESH})")
    return "\n".join(lines)


def format_mom20_alert(scan_result: dict) -> str | None:
    """Format Mom20 top-40 momentum ranking as a Telegram message.
    Sends once daily when signals change. Returns None if no signals."""
    signals = scan_result.get("mom20_signals", [])
    if not signals:
        return None

    from datetime import datetime, timezone, timedelta
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %H:%M")
    regime = scan_result.get("mom20_regime", "—")
    regime_tag = "🟢 ON" if regime == "ON" else "🔴 OFF"

    lines = [f"<b>📊 Mom20 Top 40</b> | Regime {regime_tag} | {ist}", "<code>"]
    lines.append(f"{'#':<3} {'Ticker':<10} {'Close':>7} {'12m':>6} {'3m':>6} {'Score':>5}")
    lines.append("─" * 42)
    for s in signals[:40]:
        rank  = s.get("rank", "?")
        tick  = s["ticker"][:10]
        px    = f"{s['price']:,.0f}"
        r12   = f"{s['ret_12m']:+.0f}%"
        r3    = f"{s.get('ret_3m', s.get('ret_6m', 0)):+.0f}%"
        sc    = f"{s['momentum_score']:.2f}"
        tag   = "★" if rank <= 15 else "◀"
        lines.append(f"{rank:<3} {tick:<10} {px:>7} {r12:>6} {r3:>6} {sc:>5} {tag}")
    lines.append("</code>")
    lines.append("★ entry (rank≤15)  ◀ hold buffer (16–40)")
    return "\n".join(lines)


def format_mom20_overflow_alert(scan_result: dict) -> str | None:
    """Format Mom20 overflow — high-momentum stocks excluded only by β>1.2 cap.
    These are RS63 satellite candidates not captured in Mom20 (β≤1.2)."""
    overflow = scan_result.get("mom20_overflow", [])
    if not overflow:
        return None

    from datetime import datetime, timezone, timedelta
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %H:%M")
    regime = scan_result.get("mom20_regime", "—")
    regime_tag = "🟢 ON" if regime == "ON" else "🔴 OFF"

    lines = [f"<b>⚡ Mom20 Overflow — RS63 Candidates</b> | Regime {regime_tag} | {ist}",
             "<i>High-momentum, β&gt;1.2 — not in Mom20 (β≤1.2)</i>", "<code>"]
    lines.append(f"{'#':<3} {'Ticker':<10} {'Close':>7} {'12m':>6} {'3m':>6} {'Score':>5} {'β':>4}")
    lines.append("─" * 46)
    for s in overflow:
        rank  = s.get("rank", "?")
        tick  = s["ticker"][:10]
        px    = f"{s['price']:,.0f}"
        r12   = f"{s['ret_12m']:+.0f}%"
        r3    = f"{s['ret_3m']:+.0f}%"
        sc    = f"{s['momentum_score']:.2f}"
        beta  = f"{s['beta']:.2f}"
        lines.append(f"{rank:<3} {tick:<10} {px:>7} {r12:>6} {r3:>6} {sc:>5} {beta:>4}")
    lines.append("</code>")
    lines.append("⚡ Consider as RS63 entries when RS63 &gt; 0 + RSI &gt; 50")
    return "\n".join(lines)


def get_chat_id() -> str | None:
    """Helper to fetch your chat_id from getUpdates (call once after messaging the bot)."""
    token = _token()
    if not token:
        print("[notifier] TELEGRAM_BOT_TOKEN not set")
        return None
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        results = data.get("result", [])
        if not results:
            print("[notifier] No messages yet — send any message to your bot first, then retry.")
            return None
        chat_id = str(results[-1]["message"]["chat"]["id"])
        print(f"[notifier] Your chat_id: {chat_id}")
        return chat_id
    except Exception as e:
        print(f"[notifier] getUpdates failed: {e}")
        return None
