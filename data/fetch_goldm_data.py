#!/usr/bin/env python3
"""
Fetch GOLDM 5-minute historical data from Zerodha Kite API.
Requires: At least one Kite session active (login via the RS screener app).

Usage: python3 data/fetch_goldm_data.py
Output: data/goldm_5min.csv
"""

import os
import sys
import time
import json
import glob
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "goldm_5min.csv")


def main():
    print("=== GOLDM 5-Min Data Fetcher ===\n")

    # Find a valid Kite session
    session_files = sorted(glob.glob(os.path.join(DATA_STORE, "kite_session_*.json")))
    if not session_files:
        print("ERROR: No Kite session files found. Login via the RS screener app first.")
        return

    kite = None
    for sf in session_files:
        with open(sf) as f:
            data = json.load(f)
        access_token = data.get("access_token")
        api_key = data.get("api_key")
        user_name = data.get("user_name", "unknown")

        if not access_token:
            continue

        # If api_key not in session file, try to get from env
        if not api_key:
            # Read from the app's env-based config
            idx = sf.split("_")[-1].replace(".json", "")  # user1, user2
            n = idx.replace("user", "")
            api_key = os.environ.get(f"KITE_USER{n}_API_KEY", "")

        if not api_key:
            # Try reading from the running app
            try:
                import requests
                r = requests.get("http://localhost:8080/api/users", timeout=3)
                resp = r.json()
                users = resp.get("users", [])
                uid = data.get("user_id", "")
                for u in users:
                    if u.get("user_id") == uid or u.get("config_user_id") == idx:
                        # We need the api_key which isn't exposed in the API
                        # Let's just try all session files with the kiteconnect profile
                        break
            except Exception:
                pass

        # Try to connect using just the access_token + profile
        try:
            from kiteconnect import KiteConnect
            # Get api_key from the .env or dotenv file
            env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
            if os.path.exists(env_file):
                with open(env_file) as ef:
                    for line in ef:
                        line = line.strip()
                        if line.startswith("KITE_USER") and "API_KEY" in line and "SECRET" not in line:
                            key, val = line.split("=", 1)
                            # Try each api_key with this access_token
                            test_kite = KiteConnect(api_key=val.strip())
                            test_kite.set_access_token(access_token)
                            try:
                                profile = test_kite.profile()
                                kite = test_kite
                                user_name = profile.get("user_name", user_name)
                                print(f"Connected as: {user_name}")
                                break
                            except Exception:
                                continue
            if kite:
                break
        except ImportError:
            print("ERROR: kiteconnect not installed. pip install kiteconnect")
            return
        except Exception:
            continue

    if not kite:
        print("ERROR: Could not establish Kite connection from saved sessions.")
        print("Make sure you're logged in via the RS screener app.")
        return

    # Step 1: Find GOLDM instrument token
    print("\nFetching MCX instruments...")
    instruments = kite.instruments("MCX")
    goldm_futs = [i for i in instruments
                  if "GOLDM" in i["tradingsymbol"] and i["instrument_type"] == "FUT"]

    if not goldm_futs:
        print("ERROR: No GOLDM futures found on MCX")
        return

    goldm_futs.sort(key=lambda i: i["expiry"])
    print(f"Found {len(goldm_futs)} GOLDM futures:")
    for g in goldm_futs[:5]:
        print(f"  {g['tradingsymbol']}  expiry={g['expiry']}  token={g['instrument_token']}")

    # Use nearest expiry
    nearest = goldm_futs[0]
    token = nearest["instrument_token"]
    symbol = nearest["tradingsymbol"]
    print(f"\nFetching data for: {symbol} (token={token})")

    # Step 2: Fetch 5-min data in 55-day chunks
    all_candles = []
    end_dt = datetime.now()

    for chunk in range(6):  # ~330 days
        chunk_end = end_dt - timedelta(days=chunk * 55)
        chunk_start = chunk_end - timedelta(days=55)

        print(f"  {chunk_start.date()} to {chunk_end.date()}...", end=" ")
        try:
            candles = kite.historical_data(
                token, chunk_start, chunk_end, interval="5minute")
            if candles:
                all_candles.extend(candles)
                print(f"{len(candles)} candles")
            else:
                print("no data")
        except Exception as e:
            print(f"error: {e}")

        time.sleep(0.5)

    if not all_candles:
        print("\nERROR: No data fetched. The contract may not have history.")
        print("Try fetching a different contract month.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n=== DONE ===")
    print(f"  Candles: {len(df)}")
    print(f"  Range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Price: ₹{df['close'].min():.0f} - ₹{df['close'].max():.0f}")


if __name__ == "__main__":
    main()
