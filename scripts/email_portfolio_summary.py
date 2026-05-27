#!/usr/bin/env python3
"""
email_portfolio_summary.py
──────────────────────────
Sends each user their portfolio summary by email at 3:30 PM IST on trading days.
Run via VPS cron: 00 10 * * 1-5  (10:00 UTC = 15:30 IST)

Requires in .env:
    GMAIL_USER=your@gmail.com
    GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

Users without an "email" field in users.json are silently skipped.
"""

import os
import sys
import json
import smtplib
import yfinance as yf
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from data.user_registry import (
    load_users,
    mom20_portfolio_path, mom20_history_path,
    etf_positions_path, etf_history_path,
    techmo_portfolio_path, techmo_history_path,
)

# ── NSE holidays 2026 ──────────────────────────────────────────────────────────
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 2),    # Mahashivratri
    date(2026, 3, 31),   # Eid ul-Fitr
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 6),    # Mahavir Jayanti
    date(2026, 4, 10),   # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 27),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2026, 10, 21),  # Diwali Laxmi Puja
    date(2026, 10, 22),  # Diwali Balipratipada
    date(2026, 11, 5),   # Guru Nanak Jayanti
    date(2026, 11, 25),  # Christmas (observed)
    date(2026, 12, 25),  # Christmas
}

# ── helpers ────────────────────────────────────────────────────────────────────

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NSE_HOLIDAYS_2026


def fmt_inr(v: float) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e7:
        return f"₹{v/1e7:.2f}Cr"
    if abs(v) >= 1e5:
        return f"₹{v/1e5:.2f}L"
    return f"₹{v:,.0f}"


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def fetch_live_prices(tickers: list[str]) -> dict:
    """Return {ticker: last_price} using yfinance fast_info."""
    if not tickers:
        return {}
    prices = {}
    for tk in tickers:
        yf_tk = tk + ".NS" if not tk.endswith(".NS") else tk
        try:
            info = yf.Ticker(yf_tk).fast_info
            prices[tk] = float(info.last_price or 0)
        except Exception:
            prices[tk] = 0.0
    return prices


# ── realised P&L helpers ───────────────────────────────────────────────────────

def _realised_from_mom20_history(hist: list) -> float:
    """Walk buy/sell history chronologically to compute total realised P&L."""
    pos_book = {}
    total = 0.0
    for rb in hist:
        for t in rb.get("buys", []):
            tk, qty, price = t.get("ticker"), t.get("qty", 0), t.get("price", 0)
            if not tk or not qty or not price:
                continue
            if tk in pos_book:
                old = pos_book[tk]
                new_qty = old["qty"] + qty
                pos_book[tk] = {"entry_price": (old["qty"] * old["entry_price"] + qty * price) / new_qty, "qty": new_qty}
            else:
                pos_book[tk] = {"entry_price": price, "qty": qty}
        for t in rb.get("sells", []):
            tk  = t.get("ticker")
            pnl = t.get("pnl")
            if pnl is None and tk in pos_book:
                ep  = pos_book[tk]["entry_price"]
                pnl = (t.get("price", 0) - ep) * t.get("qty", 0)
            total += pnl or 0
            pos_book.pop(tk, None)
    return round(total, 2)


# ── portfolio builders ─────────────────────────────────────────────────────────

def _build_rows(positions: list, prices: dict) -> list:
    rows = []
    for p in positions:
        tk     = p["ticker"]
        qty    = p.get("qty", 0)
        avg    = p.get("entry_price", 0)
        ltp    = prices.get(tk, avg)
        cost   = qty * avg
        mval   = qty * ltp
        pnl    = mval - cost
        pnl_pc = (pnl / cost * 100) if cost else 0
        rows.append((tk, qty, avg, ltp, pnl, pnl_pc))
    return rows


def build_mom20_summary(user_id: str, capital: float):
    port   = load_json(mom20_portfolio_path(user_id), {})
    basket = port.get("basket", [])
    if not basket:
        return None, capital, 0.0

    prices = fetch_live_prices([p["ticker"] for p in basket])
    rows   = _build_rows(basket, prices)

    hist     = load_json(mom20_history_path(user_id), [])
    realised = _realised_from_mom20_history(hist)

    return rows, sum(r[3] * r[1] for r in rows), realised


def build_etf_summary(user_id: str, capital: float):
    positions = load_json(etf_positions_path(user_id), [])
    if not positions:
        return None, capital, 0.0

    prices = fetch_live_prices([p["ticker"] for p in positions])
    rows   = _build_rows(positions, prices)

    hist     = load_json(etf_history_path(user_id), [])
    realised = sum(entry.get("pnl_abs", 0) for entry in hist)

    return rows, sum(r[3] * r[1] for r in rows), realised


def build_techmo_summary(user_id: str, capital: float):
    port   = load_json(techmo_portfolio_path(user_id), {})
    basket = port.get("basket", [])
    if not basket:
        return None, capital, 0.0

    prices = fetch_live_prices([p["ticker"] for p in basket])
    rows   = _build_rows(basket, prices)

    hist     = load_json(techmo_history_path(user_id), [])
    realised = sum(s.get("pnl", 0) for rb in hist for s in rb.get("sells", []))

    return rows, sum(r[3] * r[1] for r in rows), realised


# ── HTML builder ───────────────────────────────────────────────────────────────

def _pnl_color(pnl):
    return "#16a34a" if pnl >= 0 else "#dc2626"


def _strategy_table(title: str, currency: str, rows: list, realised: float = 0.0) -> str:
    if not rows:
        return ""

    header_color = {
        "Mom20": "#1d4ed8",
        "ETF":   "#0891b2",
        "TechMo": "#7c3aed",
    }.get(title, "#374151")

    def row_html(tk, qty, avg, ltp, pnl, pnl_pc):
        c = _pnl_color(pnl)
        sym_fmt = f"{tk}"
        avg_fmt = f"{currency}{avg:,.2f}"
        ltp_fmt = f"{currency}{ltp:,.2f}"
        pnl_fmt = f'<span style="color:{c};font-weight:700;">{currency}{abs(pnl):,.0f} ({pnl_pc:+.1f}%)</span>'
        return f"""
        <tr style="border-bottom:1px solid #f3f4f6;">
            <td style="padding:6px 10px;font-weight:600;">{sym_fmt}</td>
            <td style="padding:6px 10px;text-align:right;">{qty}</td>
            <td style="padding:6px 10px;text-align:right;">{avg_fmt}</td>
            <td style="padding:6px 10px;text-align:right;">{ltp_fmt}</td>
            <td style="padding:6px 10px;text-align:right;">{pnl_fmt}</td>
        </tr>"""

    total_invested = sum(r[1] * r[2] for r in rows)   # qty * avg
    total_current  = sum(r[1] * r[3] for r in rows)   # qty * ltp
    total_pnl      = total_current - total_invested
    pnl_pct        = (total_pnl / total_invested * 100) if total_invested else 0
    tc = _pnl_color(total_pnl)
    pnl_sign = "+" if total_pnl >= 0 else "-"

    rows_html = "".join(row_html(*r) for r in rows)

    return f"""
    <div style="margin-bottom:24px;">
        <div style="background:{header_color};color:white;padding:8px 12px;border-radius:6px 6px 0 0;
                    font-weight:700;font-size:13px;letter-spacing:0.5px;">{title} &nbsp;·&nbsp; {len(rows)} positions</div>
        <table style="width:100%;border-collapse:collapse;font-size:12px;background:#fff;
                      border:1px solid #e5e7eb;border-top:none;">
            <thead>
                <tr style="background:#f9fafb;color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:0.4px;">
                    <th style="padding:6px 10px;text-align:left;">Ticker</th>
                    <th style="padding:6px 10px;text-align:right;">Qty</th>
                    <th style="padding:6px 10px;text-align:right;">Avg Cost</th>
                    <th style="padding:6px 10px;text-align:right;">LTP</th>
                    <th style="padding:6px 10px;text-align:right;">Unrealised P&amp;L</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e5e7eb;border-top:2px solid {header_color};border-radius:0 0 6px 6px;background:#f0f4ff;">
            <tr>
                <td width="25%" style="text-align:center;padding:10px 6px;border-right:1px solid #d1d5db;">
                    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;">Invested</div>
                    <div style="font-size:13px;font-weight:700;color:#1f2937;">{currency}{total_invested:,.0f}</div>
                </td>
                <td width="25%" style="text-align:center;padding:10px 6px;border-right:1px solid #d1d5db;">
                    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;">Current Value</div>
                    <div style="font-size:13px;font-weight:700;color:#1f2937;">{currency}{total_current:,.0f}</div>
                </td>
                <td width="50%" style="text-align:center;padding:10px 6px;">
                    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;">Returns (Realised + Unrealised)</div>
                    <div style="font-size:13px;font-weight:700;color:{_pnl_color(total_pnl + realised)};">{'+' if (total_pnl + realised) >= 0 else '-'}{currency}{abs(total_pnl + realised):,.0f}</div>
                </td>
            </tr>
        </table>
    </div>"""


def build_html_email(user: dict, today: date) -> str:
    name     = user.get("name", "Investor")
    strats   = user.get("strategies", {})
    mom20_cfg  = strats.get("mom20",  {})
    etf_cfg    = strats.get("etf",    {})
    techmo_cfg = strats.get("techmo", {})

    sections = ""
    grand_value = 0.0

    if mom20_cfg.get("active"):
        rows, val, realised = build_mom20_summary(user["id"], mom20_cfg.get("capital", 0))
        sections += _strategy_table("Mom20", "₹", rows or [], realised)
        grand_value += val

    if etf_cfg.get("active"):
        rows, val, realised = build_etf_summary(user["id"], etf_cfg.get("capital", 0))
        sections += _strategy_table("ETF", "₹", rows or [], realised)
        grand_value += val

    if techmo_cfg.get("active"):
        rows, val, realised = build_techmo_summary(user["id"], techmo_cfg.get("capital", 0))
        sections += _strategy_table("TechMo", "$", rows or [], realised)
        grand_value += val

    if not sections:
        sections = '<p style="color:#6b7280;font-size:13px;">No active positions found.</p>'

    dt_str = today.strftime("%d %b %Y")

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f3f4f6;margin:0;padding:20px;">
<div style="max-width:600px;margin:0 auto;">

  <div style="background:#111827;color:white;padding:16px 20px;border-radius:8px 8px 0 0;">
    <div style="font-size:18px;font-weight:700;">Mom Portfolio Dashboard</div>
    <div style="font-size:12px;color:#9ca3af;margin-top:2px;">Daily Summary — {dt_str} · 3:30 PM IST</div>
  </div>

  <div style="background:white;padding:20px;border:1px solid #e5e7eb;border-top:none;">
    <div style="font-size:14px;color:#374151;margin-bottom:16px;">
        Hi {name.split()[0]}, here's your portfolio snapshot.
    </div>

    {sections}
  </div>

  <div style="background:#f9fafb;border:1px solid #e5e7eb;border-top:none;padding:10px 20px;
              border-radius:0 0 8px 8px;font-size:11px;color:#9ca3af;text-align:center;">
    Prices via yfinance · Paper trading only · Not financial advice
  </div>

</div>
</body>
</html>"""


# ── send ───────────────────────────────────────────────────────────────────────

def send_email(to_addr: str, subject: str, html_body: str):
    gmail_user = os.environ.get("GMAIL_USER", "")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not gmail_user or not gmail_pass:
        raise RuntimeError("GMAIL_USER / GMAIL_APP_PASSWORD not set in .env")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = to_addr
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.ehlo()
        s.starttls()
        s.login(gmail_user, gmail_pass)
        s.sendmail(gmail_user, to_addr, msg.as_string())


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    today = date.today()

    if not is_trading_day(today):
        print(f"[{today}] Not a trading day — skipping.")
        return

    users = load_users()
    if not users:
        print("No users found.")
        return

    subject = f"Portfolio Summary — {today.strftime('%d %b %Y')}"
    sent = 0
    for user in users:
        email = (user.get("email") or "").strip()
        if not email:
            print(f"  Skipping {user['name']} (no email configured)")
            continue

        # Check user has at least one active strategy
        strats = user.get("strategies", {})
        if not any(v.get("active") for v in strats.values()):
            print(f"  Skipping {user['name']} (no active strategies)")
            continue

        try:
            html = build_html_email(user, today)
            send_email(email, subject, html)
            print(f"  Sent to {user['name']} <{email}>")
            sent += 1
        except Exception as e:
            print(f"  ERROR sending to {user['name']}: {e}")

    print(f"Done. {sent}/{len(users)} emails sent.")


if __name__ == "__main__":
    main()
