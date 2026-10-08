import argparse
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from fyers_apiv3 import fyersModel


# ============================================================
# CONFIG
# ============================================================

DATA = Path("data")
CACHE_DIR = DATA / "fyers_cache"
UNIVERSE_FILE = Path("fyers_universe.csv")
TOKEN_FILE = Path("fyers_token.txt")
PROGRESS_FILE = DATA / "fyers_progress.json"

# FYERS App ID
# This is NOT the access token.
CLIENT_ID = "VSS2BPXSRN-200"

# Daily candles
RESOLUTION = "D"

# FYERS maximum daily history range is 366 days.
CHUNK_DAYS = 365

# Stay below FYERS request-rate limit.
REQUEST_DELAY = 1.25

# Keep safety margin below daily limit.
DAILY_REQUEST_LIMIT = 4900


# ============================================================
# TOKEN / FYERS CLIENT
# ============================================================

def load_token():
    """
    Local:
        reads fyers_token.txt

    GitHub Actions:
        reads FYERS_TOKEN environment variable first.
    """

    env_token = os.environ.get("FYERS_TOKEN", "").strip()

    if env_token:
        return env_token

    if not TOKEN_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TOKEN_FILE}. Run fyers_login.py first."
        )

    token = TOKEN_FILE.read_text().strip()

    if not token:
        raise ValueError("fyers_token.txt is empty.")

    return token


def make_fyers():
    token = load_token()

    if CLIENT_ID == "YOUR_FYERS_APP_ID_HERE":
        raise ValueError(
            "Put your FYERS App ID into CLIENT_ID in fyers_data.py."
        )

    return fyersModel.FyersModel(
        client_id=CLIENT_ID,
        is_async=False,
        token=token,
        log_path=""
    )


# ============================================================
# UNIVERSE
# ============================================================

def load_universe():
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {UNIVERSE_FILE}"
        )

    df = pd.read_csv(UNIVERSE_FILE)

    if df.empty:
        raise ValueError("Universe file is empty.")

    return df


def value_from_row(row, names):
    for name in names:
        if name in row.index:
            value = row[name]

            if pd.notna(value):
                s = str(value).strip()

                if s and s.lower() not in ("nan", "none"):
                    return s

    return None


def safe_filename(key):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        key
    )


def get_key(row):
    """
    Prefer the existing universe key.

    Examples:
        NSE:SBIN
        NSE:RELIANCE
    """

    key = value_from_row(
        row,
        [
            "key",
            "Key",
            "scanner_key",
        ]
    )

    if key:
        return key

    exch = value_from_row(
        row,
        [
            "exch",
            "exchange",
            "Exchange",
        ]
    )

    symbol = value_from_row(
        row,
        [
            "ticker",
            "symbol_name",
            "stock_symbol",
            "name_symbol",
            "symbol",
        ]
    )

    if exch and symbol:
        return f"{exch}:{symbol}"

    raise ValueError(
        "Could not determine key from universe row."
    )


def get_exchange(row, key):
    exch = value_from_row(
        row,
        [
            "exch",
            "exchange",
            "Exchange",
        ]
    )

    if exch:
        return exch.upper()

    if ":" in key:
        return key.split(":", 1)[0].upper()

    return ""


def get_display_symbol(row, key):
    """
    Return:
        SBIN

    instead of:
        NSE:SBIN
    """

    key_symbol = (
        key.split(":", 1)[1]
        if ":" in key
        else key
    )

    return key_symbol


def get_name(row):
    return value_from_row(
        row,
        [
            "name",
            "company_name",
            "company",
            "Name",
        ]
    ) or ""


def get_fyers_symbol(row, key):
    """
    Return the actual FYERS request symbol.

    Examples:
        NSE:SBIN-EQ
        NSE:RELIANCE-EQ
    """

    preferred = [
        "fyers_symbol",
        "fy_symbol",
        "api_symbol",
        "request_symbol",
        "trading_symbol",
        "fyers",
    ]

    for col in preferred:
        if col in row.index and pd.notna(row[col]):
            s = str(row[col]).strip()

            if ":" in s and s.upper().startswith(
                ("NSE:", "BSE:")
            ):
                return s

    if "symbol" in row.index and pd.notna(row["symbol"]):
        s = str(row["symbol"]).strip()

        if ":" in s and s.upper().startswith(
            ("NSE:", "BSE:")
        ):
            if s != key:
                return s

    return key


# ============================================================
# PROGRESS
# ============================================================

def load_progress():
    if not PROGRESS_FILE.exists():
        return {
            "version": 3,
            "day": str(date.today()),
            "requests_today": 0,
            "completed": [],
        }

    try:
        obj = json.loads(
            PROGRESS_FILE.read_text()
        )
    except Exception:
        obj = {}

    today = str(date.today())

    if obj.get("day") != today:
        obj["day"] = today
        obj["requests_today"] = 0

    obj.setdefault("version", 3)
    obj.setdefault("requests_today", 0)
    obj.setdefault("completed", [])

    return obj


def save_progress(progress):
    DATA.mkdir(
        parents=True,
        exist_ok=True
    )

    PROGRESS_FILE.write_text(
        json.dumps(
            progress,
            indent=2
        )
    )


def requests_available(progress):
    used = int(
        progress.get(
            "requests_today",
            0
        )
    )

    return max(
        0,
        DAILY_REQUEST_LIMIT - used
    )


def mark_request(progress):
    progress["requests_today"] = (
        int(
            progress.get(
                "requests_today",
                0
            )
        )
        + 1
    )

    save_progress(progress)


# ============================================================
# FYERS API
# ============================================================

def request_history(
    fyers,
    fyers_symbol,
    range_from,
    range_to,
    progress,
):
    """
    Request one history chunk.
    """

    if requests_available(progress) <= 0:
        return None, "DAILY_LIMIT"

    payload = {
        "symbol": fyers_symbol,
        "resolution": RESOLUTION,
        "date_format": "1",
        "range_from": range_from.strftime(
            "%Y-%m-%d"
        ),
        "range_to": range_to.strftime(
            "%Y-%m-%d"
        ),
        "cont_flag": "1",
    }

    try:
        response = fyers.history(
            data=payload
        )
    except Exception as e:
        return None, f"EXCEPTION: {e}"

    mark_request(progress)

    if not isinstance(response, dict):
        return None, "INVALID_RESPONSE"

    status = response.get("s")

    if status != "ok":
        msg = (
            response.get("message")
            or response.get("code")
            or "unknown error"
        )

        return None, f"API_ERROR: {msg}"

    candles = response.get(
        "candles",
        []
    )

    if not candles:
        return [], None

    return candles, None


def candles_to_df(candles):
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if not candles:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    for candle in candles:
        if len(candle) < 6:
            continue

        try:
            epoch = int(candle[0])

            dt = datetime.utcfromtimestamp(
                epoch
            ).date()

            rows.append(
                [
                    dt,
                    float(candle[1]),
                    float(candle[2]),
                    float(candle[3]),
                    float(candle[4]),
                    float(candle[5]),
                ]
            )

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    df = df.drop_duplicates(
        "date"
    )

    df = df.sort_values(
        "date"
    )

    return df


# ============================================================
# CACHE
# ============================================================

def cache_path(key):
    return (
        CACHE_DIR
        / f"{safe_filename(key)}.csv"
    )


def load_cache(key):
    path = cache_path(key)

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = pd.read_csv(path)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "date",
            "close"
        ]
    )

    df = df.drop_duplicates(
        "date",
        keep="last"
    )

    df = df.sort_values(
        "date"
    )

    return df


def save_cache(
    key,
    df,
):
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out = df.copy()

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce"
        )

    out = out.dropna(
        subset=[
            "date",
            "close"
        ]
    )

    out = out.drop_duplicates(
        "date",
        keep="last"
    )

    out = out.sort_values(
        "date"
    )

    path = cache_path(key)

    out.to_csv(
        path,
        index=False
    )


def merge_candles(
    existing,
    new,
):
    if existing is None or existing.empty:
        merged = new.copy()

    elif new is None or new.empty:
        merged = existing.copy()

    else:
        merged = pd.concat(
            [
                existing,
                new
            ],
            ignore_index=True
        )

    if merged.empty:
        return merged

    merged["date"] = pd.to_datetime(
        merged["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    merged = merged.drop_duplicates(
        "date",
        keep="last"
    )

    merged = merged.sort_values(
        "date"
    )

    return merged


def get_cache_info(key):
    df = load_cache(key)

    if df.empty:
        return {
            "rows": 0,
            "first": None,
            "last": None,
        }

    return {
        "rows": len(df),
        "first": df["date"].min(),
        "last": df["date"].max(),
    }


# ============================================================
# DAILY INCREMENTAL UPDATE
# ============================================================

def update_latest(
    fyers,
    row,
    progress,
):
    """
    Update ONE stock with candles missing after
    the latest cached candle.

    No unnecessary 365-day re-download.

    Example:

        Cache last date:
            2026-10-06

        Today:
            2026-10-07

        Request:
            2026-10-07 -> 2026-10-07

    If there is no cache, bootstrap the latest 365 days.
    """

    key = get_key(row)

    fyers_symbol = get_fyers_symbol(
        row,
        key
    )

    today = date.today()

    existing = load_cache(key)

    # --------------------------------------------------------
    # No cache -> bootstrap latest year
    # --------------------------------------------------------

    if existing.empty:
        range_to = today

        range_from = (
            today
            - timedelta(days=CHUNK_DAYS)
        )

    else:
        latest = pd.to_datetime(
            existing["date"],
            errors="coerce"
        ).max().date()

        # Already current.
        if latest >= today:
            return existing, "CURRENT"

        range_from = (
            latest
            + timedelta(days=1)
        )

        range_to = today

        # Safety against FYERS maximum range.
        if (
            range_to - range_from
        ).days > CHUNK_DAYS:
            range_from = (
                range_to
                - timedelta(days=CHUNK_DAYS)
            )

    candles, error = request_history(
        fyers,
        fyers_symbol,
        range_from,
        range_to,
        progress
    )

    if error:
        return existing, error

    new_df = candles_to_df(
        candles
    )

    if new_df.empty:
        return existing, "NO_NEW_DATA"

    merged = merge_candles(
        existing,
        new_df
    )

    save_cache(
        key,
        merged
    )

    return merged, len(new_df)


def run_update(
    universe,
    limit=None,
):
    """
    Daily incremental updater.

    Fetches only data after the latest cached
    candle for each stock.
    """

    fyers = make_fyers()

    progress = load_progress()

    rows = list(
        universe.iterrows()
    )

    if limit:
        rows = rows[:limit]

    print("=" * 64)
    print("FYERS DAILY INCREMENTAL UPDATE")
    print("=" * 64)

    print(
        f"Universe          : {len(universe):,}"
    )

    print(
        f"Selected          : {len(rows):,}"
    )

    print(
        f"Requests today    : "
        f"{progress.get('requests_today', 0):,}"
    )

    print(
        f"Requests remaining: "
        f"{requests_available(progress):,}"
    )

    print()

    updated = 0
    current = 0
    no_data = 0
    errors = 0

    for idx, (_, row) in enumerate(
        rows,
        start=1
    ):
        if requests_available(
            progress
        ) <= 0:

            print()
            print(
                "Daily API request limit reached."
            )

            break

        key = get_key(row)

        before = get_cache_info(
            key
        )

        df, result = update_latest(
            fyers,
            row,
            progress
        )

        after = get_cache_info(
            key
        )

        if result == "CURRENT":
            current += 1

            print(
                f"[{idx:5}/{len(rows)}] "
                f"{key:<25} CURRENT "
                f"{after['last']}"
            )

        elif result == "NO_NEW_DATA":
            no_data += 1

            print(
                f"[{idx:5}/{len(rows)}] "
                f"{key:<25} "
                f"NO NEW DATA "
                f"{after['last']}"
            )

        elif isinstance(result, int):
            updated += 1

            added = (
                after["rows"]
                - before["rows"]
            )

            print(
                f"[{idx:5}/{len(rows)}] "
                f"{key:<25} "
                f"+{added:>3} candles "
                f"{after['last']}"
            )

        else:
            errors += 1

            print(
                f"[{idx:5}/{len(rows)}] "
                f"{key:<25} "
                f"{result}"
            )

        # Do not unnecessarily sleep after
        # stocks that required no API call.
        if result not in (
            "CURRENT",
        ):
            time.sleep(
                REQUEST_DELAY
            )

    save_progress(
        progress
    )

    print()
    print("=" * 64)
    print("DAILY UPDATE COMPLETE")
    print("=" * 64)

    print(
        f"Updated       : {updated:,}"
    )

    print(
        f"Already current: {current:,}"
    )

    print(
        f"No new data   : {no_data:,}"
    )

    print(
        f"Errors        : {errors:,}"
    )

    print(
        f"Requests used : "
        f"{progress.get('requests_today', 0):,}"
    )

    print(
        f"Requests left : "
        f"{requests_available(progress):,}"
    )


# ============================================================
# FULL HISTORY DOWNLOAD
# ============================================================

def initial_latest_chunk(
    fyers,
    row,
    progress,
):
    key = get_key(row)

    fyers_symbol = get_fyers_symbol(
        row,
        key
    )

    today = date.today()

    range_to = today

    range_from = (
        today
        - timedelta(days=CHUNK_DAYS)
    )

    existing = load_cache(key)

    if not existing.empty:
        latest = pd.to_datetime(
            existing["date"],
            errors="coerce"
        ).max().date()

        if latest >= (
            today
            - timedelta(days=7)
        ):
            return existing, "EXISTS"

    candles, error = request_history(
        fyers,
        fyers_symbol,
        range_from,
        range_to,
        progress
    )

    if error:
        return existing, error

    new_df = candles_to_df(
        candles
    )

    merged = merge_candles(
        existing,
        new_df
    )

    if not merged.empty:
        save_cache(
            key,
            merged
        )

    return merged, len(new_df)


def backfill_one_chunk(
    fyers,
    row,
    progress,
):
    """
    Fetch exactly ONE older 365-day chunk.
    """

    key = get_key(row)

    existing = load_cache(key)

    if existing.empty:
        return None, "NO_CACHE"

    oldest = pd.to_datetime(
        existing["date"],
        errors="coerce"
    ).min().date()

    range_to = (
        oldest
        - timedelta(days=1)
    )

    range_from = (
        range_to
        - timedelta(days=CHUNK_DAYS - 1)
    )

    fyers_symbol = get_fyers_symbol(
        row,
        key
    )

    candles, error = request_history(
        fyers,
        fyers_symbol,
        range_from,
        range_to,
        progress
    )

    if error:
        return existing, error

    if not candles:
        return existing, "NO_OLDER_DATA"

    new_df = candles_to_df(
        candles
    )

    if new_df.empty:
        return existing, "NO_OLDER_DATA"

    merged = merge_candles(
        existing,
        new_df
    )

    save_cache(
        key,
        merged
    )

    return merged, len(new_df)


def run_full(
    universe,
    limit=None
):
    """
    Full historical downloader.

    Phase 1:
        latest ~365 days

    Phase 2:
        one older chunk per stock

    Safe to stop and resume.
    """

    fyers = make_fyers()

    progress = load_progress()

    completed = set(
        progress.get(
            "completed",
            []
        )
    )

    rows = list(
        universe.iterrows()
    )

    if limit:
        rows = rows[:limit]

    print("=" * 64)
    print("FYERS FULL HISTORY DOWNLOAD")
    print("=" * 64)

    print(
        f"Universe          : {len(universe):,}"
    )

    print(
        f"Selected          : {len(rows):,}"
    )

    print(
        f"Requests today    : "
        f"{progress.get('requests_today', 0):,}"
    )

    print(
        f"Requests remaining: "
        f"{requests_available(progress):,}"
    )

    # --------------------------------------------------------
    # PHASE 1
    # --------------------------------------------------------

    print()
    print("-" * 64)
    print("PHASE 1: LATEST 365 DAYS")
    print("-" * 64)

    for idx, (_, row) in enumerate(
        rows,
        start=1
    ):
        if requests_available(
            progress
        ) <= 0:

            print(
                "Daily API request limit reached."
            )

            break

        key = get_key(row)

        info = get_cache_info(
            key
        )

        if info["rows"] > 0:
            if info["last"] >= str(
                date.today()
                - timedelta(days=7)
            ):
                print(
                    f"[{idx:5}/{len(rows)}] "
                    f"{key:<25} "
                    f"existing "
                    f"{info['rows']:>4} candles "
                    f"{info['first']} -> "
                    f"{info['last']}"
                )

                continue

        df, result = initial_latest_chunk(
            fyers,
            row,
            progress
        )

        info = get_cache_info(
            key
        )

        print(
            f"[{idx:5}/{len(rows)}] "
            f"{key:<25} "
            f"{result!s:<15} "
            f"{info['rows']:>4} candles "
            f"{info['first']} -> "
            f"{info['last']}"
        )

        time.sleep(
            REQUEST_DELAY
        )

    # --------------------------------------------------------
    # PHASE 2
    # --------------------------------------------------------

    print()
    print("-" * 64)
    print(
        "PHASE 2: BACKFILL TO EARLIEST "
        "AVAILABLE HISTORY"
    )
    print("-" * 64)

    for idx, (_, row) in enumerate(
        rows,
        start=1
    ):
        if requests_available(
            progress
        ) <= 0:

            print(
                "Daily API request limit "
                "reached during backfill."
            )

            break

        key = get_key(row)

        if key in completed:
            continue

        df, result = backfill_one_chunk(
            fyers,
            row,
            progress
        )

        if result == "NO_CACHE":
            continue

        info = get_cache_info(
            key
        )

        if result == "NO_OLDER_DATA":
            completed.add(key)

            progress["completed"] = sorted(
                completed
            )

            save_progress(
                progress
            )

            print(
                f"[DONE ] "
                f"{key:<25} "
                f"earliest available "
                f"{info['first']} "
                f"({info['rows']} candles)"
            )

        elif isinstance(result, int):

            print(
                f"[BACKF] "
                f"{key:<25} "
                f"+{result:>4} candles "
                f"total={info['rows']:>5} "
                f"{info['first']} -> "
                f"{info['last']}"
            )

        else:

            print(
                f"[WARN ] "
                f"{key:<25} "
                f"{result}"
            )

        time.sleep(
            REQUEST_DELAY
        )

    progress["completed"] = sorted(
        completed
    )

    save_progress(
        progress
    )

    print()
    print("=" * 64)
    print("FULL HISTORY PASS COMPLETE")
    print("=" * 64)

    print(
        f"Requests today : "
        f"{progress.get('requests_today', 0):,}"
    )

    print(
        f"Remaining today: "
        f"{requests_available(progress):,}"
    )

    print(
        f"Completed      : "
        f"{len(completed):,}"
    )


# ============================================================
# BUILD PRICES DATABASE
# ============================================================

def build_prices():
    """
    Combine FYERS cache into:

        data/prices.csv.gz

    Compatible with run_scans.py.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    universe = load_universe()

    lookup = {}

    for _, row in universe.iterrows():

        key = get_key(row)

        lookup[key] = {
            "symbol": get_display_symbol(
                row,
                key
            ),
            "exch": get_exchange(
                row,
                key
            ),
            "isin": value_from_row(
                row,
                [
                    "isin",
                    "ISIN"
                ]
            ) or "",
            "name": get_name(row),
        }

    frames = []

    files = sorted(
        CACHE_DIR.glob("*.csv")
    )

    print(
        f"Cache files found: "
        f"{len(files):,}"
    )

    for n, path in enumerate(
        files,
        start=1
    ):
        try:
            df = pd.read_csv(
                path
            )

            if df.empty:
                continue

            stem = path.stem

            matched_key = None

            for key in lookup:
                if (
                    safe_filename(key)
                    == stem
                ):
                    matched_key = key
                    break

            if matched_key is None:
                continue

            meta = lookup[
                matched_key
            ]

            needed = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            if not all(
                col in df.columns
                for col in needed
            ):
                continue

            out = df[
                needed
            ].copy()

            out["date"] = pd.to_datetime(
                out["date"],
                errors="coerce"
            ).dt.strftime(
                "%Y-%m-%d"
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                out[col] = pd.to_numeric(
                    out[col],
                    errors="coerce"
                )

            out = out.dropna(
                subset=[
                    "date",
                    "close"
                ]
            )

            out["key"] = matched_key
            out["symbol"] = meta[
                "symbol"
            ]
            out["exch"] = meta[
                "exch"
            ]
            out["isin"] = meta[
                "isin"
            ]
            out["name"] = meta[
                "name"
            ]

            out["value"] = (
                out["close"]
                * out["volume"]
            )

            out["raw_close"] = (
                out["close"]
            )

            frames.append(
                out[
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
                print(
                    f"Processed: "
                    f"{n:,}/{len(files):,}"
                )

        except Exception as e:

            print(
                f"Skipping "
                f"{path.name}: {e}"
            )

    if not frames:
        print(
            "No cache files found."
        )

        return

    result = pd.concat(
        frames,
        ignore_index=True
    )

    result = result.drop_duplicates(
        subset=[
            "key",
            "date"
        ],
        keep="last"
    )

    result = result.sort_values(
        [
            "key",
            "date"
        ]
    )

    DATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        DATA
        / "prices.csv.gz"
    )

    result.to_csv(
        output,
        index=False,
        compression="gzip"
    )

    # Scanner compatibility.
    events_file = (
        DATA
        / "events.csv"
    )

    if not events_file.exists():

        pd.DataFrame(
            columns=[
                "key",
                "date",
                "method",
            ]
        ).to_csv(
            events_file,
            index=False
        )

    print()
    print("=" * 64)
    print("PRICES DATABASE BUILT")
    print("=" * 64)

    print(
        f"Rows       : "
        f"{len(result):,}"
    )

    print(
        f"Stocks     : "
        f"{result['key'].nunique():,}"
    )

    print(
        f"First date : "
        f"{result['date'].min()}"
    )

    print(
        f"Last date  : "
        f"{result['date'].max()}"
    )

    print()
    print("Exchange split:")

    print(
        result["exch"].value_counts()
    )

    print()
    print(
        f"Saved: {output}"
    )


# ============================================================
# STATUS
# ============================================================

def check_status():
    universe = load_universe()

    total = len(universe)

    cached = 0
    completed = 0
    rows_total = 0

    progress = load_progress()

    completed_set = set(
        progress.get(
            "completed",
            []
        )
    )

    first_dates = []
    last_dates = []

    for _, row in universe.iterrows():

        key = get_key(row)

        info = get_cache_info(
            key
        )

        if info["rows"] > 0:

            cached += 1

            rows_total += info[
                "rows"
            ]

            if info["first"]:
                first_dates.append(
                    info["first"]
                )

            if info["last"]:
                last_dates.append(
                    info["last"]
                )

        if key in completed_set:
            completed += 1

    print("=" * 64)
    print("FYERS DATA STATUS")
    print("=" * 64)

    print(
        f"Universe stocks : "
        f"{total:,}"
    )

    print(
        f"Cached stocks   : "
        f"{cached:,}"
    )

    print(
        f"Missing stocks  : "
        f"{total - cached:,}"
    )

    print(
        f"Completed full  : "
        f"{completed:,}"
    )

    print(
        f"Cached rows     : "
        f"{rows_total:,}"
    )

    if first_dates:
        print(
            f"Earliest cached: "
            f"{min(first_dates)}"
        )

    if last_dates:
        print(
            f"Latest cached  : "
            f"{max(last_dates)}"
        )

    print()

    print(
        f"Requests today : "
        f"{progress.get('requests_today', 0):,}"
    )

    print(
        f"Requests remain: "
        f"{requests_available(progress):,}"
    )


# ============================================================
# STOCK INSPECTOR
# ============================================================

def inspect_stock(symbol):
    """
    Examples:

        python fyers_data.py --stock NSE:SBIN
        python fyers_data.py --stock SBIN
    """

    universe = load_universe()

    target = symbol.upper().strip()

    matched = None

    for _, row in universe.iterrows():

        key = get_key(row)

        fyers_symbol = get_fyers_symbol(
            row,
            key
        )

        display_symbol = (
            get_display_symbol(
                row,
                key
            )
        )

        if (
            key.upper() == target
            or fyers_symbol.upper() == target
            or display_symbol.upper() == target
        ):
            matched = row
            break

    if matched is None:

        print(
            f"Stock not found in universe: "
            f"{symbol}"
        )

        return

    key = get_key(
        matched
    )

    info = get_cache_info(
        key
    )

    print("=" * 64)
    print("STOCK HISTORY STATUS")
    print("=" * 64)

    print(
        f"Key           : {key}"
    )

    print(
        f"FYERS symbol  : "
        f"{get_fyers_symbol(matched, key)}"
    )

    print(
        f"Rows          : "
        f"{info['rows']}"
    )

    print(
        f"First candle  : "
        f"{info['first']}"
    )

    print(
        f"Last candle   : "
        f"{info['last']}"
    )

    if info["rows"] > 0:

        df = load_cache(
            key
        )

        print()

        print(
            df.tail(5).to_string(
                index=False
            )
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FYERS daily market-data "
            "downloader and price builder"
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Download latest history and "
            "progressively backfill older history."
        )
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Fetch only candles missing "
            "after the latest cached date."
        )
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help=(
            "Build data/prices.csv.gz "
            "from FYERS cache."
        )
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Show FYERS cache status."
    )

    parser.add_argument(
        "--stock",
        type=str,
        help=(
            "Inspect a stock, e.g. "
            "NSE:SBIN or SBIN."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N "
            "stocks. Useful for testing."
        )
    )

    args = parser.parse_args()

    selected = sum(
        [
            bool(args.full),
            bool(args.update),
            bool(args.build),
            bool(args.check),
            bool(args.stock),
        ]
    )

    if selected == 0:
        parser.print_help()
        return

    if selected > 1:
        parser.error(
            "Use only one of "
            "--full, --update, --build, "
            "--check, or --stock."
        )

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    if args.check:

        check_status()
        return

    # --------------------------------------------------------
    # STOCK
    # --------------------------------------------------------

    if args.stock:

        inspect_stock(
            args.stock
        )

        return

    # --------------------------------------------------------
    # LOAD UNIVERSE
    # --------------------------------------------------------

    universe = load_universe()

    # --------------------------------------------------------
    # FULL
    # --------------------------------------------------------

    if args.full:

        run_full(
            universe,
            limit=args.limit
        )

        return

    # --------------------------------------------------------
    # DAILY UPDATE
    # --------------------------------------------------------

    if args.update:

        run_update(
            universe,
            limit=args.limit
        )

        return

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    if args.build:

        build_prices()
        return


if __name__ == "__main__":
    main()
