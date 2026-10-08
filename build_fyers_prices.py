import re
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/fyers_cache")
UNIVERSE_FILE = Path("fyers_universe.csv")
OUT = Path("data/prices_fyers.csv.gz")


def safe_symbol_from_file(path):
    name = path.stem
    if not name.startswith("NSE_"):
        return None
    return name[4:]


def main():
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing {CACHE_DIR}")

    universe = pd.read_csv(UNIVERSE_FILE)

    # Build lookup from the NSE universe.
    lookup = {}

    for _, row in universe.iterrows():
        key = str(row.get("key", "")).strip()
        symbol = str(row.get("symbol", "")).strip()

        if not symbol or symbol.lower() == "nan":
            continue

        if key.startswith("NSE:"):
            # fyers_data.py sanitizes filenames using:
            # re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            # Recreate that mapping so symbols containing &, etc.
            # still resolve correctly.
            filename_symbol = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", symbol
            )

            lookup[filename_symbol] = {
                "key": key,
                "symbol": symbol,
                "exch": "NSE",
                "isin": str(row.get("isin", "")).strip(),
                "name": str(row.get("name", "")).strip(),
            }

    frames = []
    files = sorted(CACHE_DIR.glob("NSE_*.csv"))

    print(f"NSE cache files found: {len(files):,}")

    for n, path in enumerate(files, 1):
        symbol = safe_symbol_from_file(path)

        if not symbol or symbol not in lookup:
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"SKIP {path.name}: {e}")
            continue

        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            print(f"SKIP {path.name}: missing columns")
            continue

        meta = lookup[symbol]

        df = df[list(required)].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=["date", "open", "high", "low", "close", "volume"]
        )

        df = df[
            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)
        ]

        if df.empty:
            continue

        df["key"] = meta["key"]
        df["exch"] = meta["exch"]
        df["symbol"] = meta["symbol"]
        df["isin"] = meta["isin"]
        df["name"] = meta["name"]

        # Same meaning expected by run_scans.py:
        # traded value in rupees.
        df["value"] = df["close"] * df["volume"]

        # FYERS data has no separate corporate-action-adjusted/raw price.
        df["raw_close"] = df["close"]

        frames.append(
            df[
                [
                    "key",
                    "exch",
                    "symbol",
                    "isin",
                    "name",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "value",
                    "raw_close",
                ]
            ]
        )

        if n % 250 == 0:
            print(f"Processed: {n:,}/{len(files):,}")

    if not frames:
        raise RuntimeError("No NSE FYERS data was converted.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["key", "date"])
    out = out.drop_duplicates(["key", "date"], keep="last")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print()
    print("=" * 64)
    print("FYERS PRICE BUILD COMPLETE")
    print("=" * 64)
    print(f"Rows   : {len(out):,}")
    print(f"Stocks : {out['key'].nunique():,}")
    print(f"First  : {out['date'].min().date()}")
    print(f"Last   : {out['date'].max().date()}")
    print(f"Output : {OUT}")
    print()
    print(out.groupby("exch")["key"].nunique().to_string())


if __name__ == "__main__":
    main()
