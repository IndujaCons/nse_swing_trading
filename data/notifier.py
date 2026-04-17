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


def _chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "")


def send_message(text: str) -> bool:
    """Send a plain text / HTML message to the configured Telegram chat.
    Returns True if the request succeeded, False otherwise."""
    token = _token()
    chat_id = _chat_id()
    if not token or not chat_id:
        print("[notifier] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not r.ok:
            print(f"[notifier] Telegram error {r.status_code}: {r.text[:200]}")
        return r.ok
    except Exception as e:
        print(f"[notifier] Request failed: {e}")
        return False


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


def format_rs63_alert(scan_result: dict) -> str | None:
    """
    Format RS63 live signals scan result as a Telegram HTML message.
    Returns None if no RS63 signals are present.
    """
    signals = scan_result.get("rs63_signals", [])
    if not signals:
        return None

    scan_time = scan_result.get("scan_time", "")
    lines = [f"<b>📈 RS63 Entry Signals</b> — {scan_time}"]
    lines.append(f"<i>{len(signals)} stocks qualifying today</i>")
    lines.append("")

    for s in signals[:10]:  # cap at 10 to keep message readable
        lines.append(
            f"  #{s.get('rank', '?')} <b>{s['ticker']}</b> "
            f"@ ₹{s['price']:,.0f} | RS63={s['rs63']}% | RSI={s['rsi']} | "
            f"SL=₹{s['sl_price']:,.0f} ({s['stop_pct']}%)"
        )

    if len(signals) > 10:
        lines.append(f"  <i>…and {len(signals) - 10} more</i>")

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
