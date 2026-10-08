"""Build full-history chart data from the FYERS price dataset."""

import json
from pathlib import Path

import pandas as pd

DATA = Path("data")
RESULTS = DATA / "results.json"
PRICES = DATA / "prices_fyers.csv.gz"
OUT = DATA / "chart_data.json"


def build_one(key, df):
    x = df[df["key"] == key].copy()

    if x.empty:
        return None

    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date")

    close = pd.to_numeric(x["close"], errors="coerce")

    x["ema11"] = close.ewm(span=11, adjust=False).mean()
    x["ema21"] = close.ewm(span=21, adjust=False).mean()
    x["sma50"] = close.rolling(50).mean()

    return {
        "symbol": key.split(":", 1)[1],
        "exch": key.split(":", 1)[0],
        "dates": x["date"].dt.strftime("%Y-%m-%d").tolist(),
        "open": pd.to_numeric(x["open"], errors="coerce").round(4).tolist(),
        "high": pd.to_numeric(x["high"], errors="coerce").round(4).tolist(),
        "low": pd.to_numeric(x["low"], errors="coerce").round(4).tolist(),
        "close": close.round(4).tolist(),
        "volume": pd.to_numeric(x["volume"], errors="coerce").fillna(0).astype("int64").tolist(),
        "ema11": x["ema11"].round(4).tolist(),
        "ema21": x["ema21"].round(4).tolist(),
        "sma50": [
            round(float(v), 4) if pd.notna(v) else None
            for v in x["sma50"]
        ],
    }


def main():
    payload = json.loads(RESULTS.read_text())

    keys = sorted({
        row["key"]
        for rows in payload["scans"].values()
        for row in rows
    })

    print(f"Building full chart history for {len(keys)} scanned stocks...")
    print(f"Source: {PRICES}")

    df = pd.read_csv(PRICES)

    required = {"key", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"Missing columns in price data: {sorted(missing)}")

    chart = {}
    failures = []

    for i, key in enumerate(keys, 1):
        try:
            data = build_one(key, df)

            if data is None:
                failures.append((key, "no data"))
                print(f"[{i}/{len(keys)}] {key}: FAILED - no data")
                continue

            chart[key] = data
            print(f"[{i}/{len(keys)}] {key}: {len(data['dates'])} sessions")

        except Exception as exc:
            failures.append((key, str(exc)))
            print(f"[{i}/{len(keys)}] {key}: FAILED - {exc}")

    OUT.write_text(json.dumps(chart, separators=(",", ":")))

    print()
    print("=" * 64)
    print("FYERS CHART HISTORY COMPLETE")
    print("=" * 64)
    print(f"Saved    : {len(chart)} histories")
    print(f"Failures : {len(failures)}")
    print(f"Output   : {OUT}")


if __name__ == "__main__":
    main()
