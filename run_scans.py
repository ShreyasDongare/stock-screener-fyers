"""
Stage 3: four exact cash-segment scans supplied for the Wolf Momentum Screener.

Scans
-----
1. 1M / 3M / 6M Scan
2. 3M 30% Scan
3. HTF Scan
4. 52Week High Scan
5. Volume Scan (NSE only)

Input:
    data/prices.csv.gz

Output:
    data/results.json
    data/chart_data.json


Common filters requested for all four scans:
    - ADX(14) must be >= 20
    - stock must have gained >= 30% over at least one of:
      31, 93, or 186 trading days
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")

# ------------------------------------------------------------
# Exact user-supplied constants
# ------------------------------------------------------------
CFG = {
    "common": {
        "min_adx": 20.0,
        "min_return_pct": 30.0,
        "return_windows": (31, 93, 186),
        # Tradability filter: remove stocks that repeatedly behave like
        # locked/illiquid UC counters rather than normally tradeable movers.
        "tradability": {
            "lookback_days": 20,
            "min_avg_value_rupees": 10_000_000.0,
            "near_uc_pct": 4.9,
            "max_open_close_pct": 0.15,
            "min_near_uc_days": 7,
            "min_open_close_days": 7,
        },
    },
    "m136": {
        "name": "1M / 3M / 6M Scan",
        "adr_min_pct": 3.0,
        "ema_days": 60,
        "breakout_windows": {
            "1M": 31,
            "3M": 93,
            "6M": 186,
        },
    },
    "m30": {
        "name": "3M 30% Scan",
        "return_days": 63,
        "min_return_pct": 30.0,
        "ema_days": 75,
        "adr_min_pct": 3.0,
        "min_value_rupees": 10_000_000.0,
    },
    "htf": {
        "name": "HTF Scan",
        "lookback_days": 60,
        "min_return_multiple": 1.5,
        "high_lookback_days": 252,
        "max_high_distance_pct": 15.0,
        "volume_sma_days": 20,
        "max_volume_ratio": 1.5,
        "min_close": 50.0,
    },
    "high52w": {
        "name": "52Week High",
        "value_min_rupees": 10_000_000.0,
        "adr_min_pct": 3.0,
        "high_lookback_days": 252,
        "max_high_distance_pct": 10.0,
    },
    "ep": {
        "name": "Episodic Pivot",
        "enabled": True,
        "lookback_days": 3,
        "min_gap_pct": 8.0,
        "max_gap_pct": 40.0,
        "min_vol_ratio": 3.0,
        "min_close_pos": 0.60,
        "max_prior_3m_pct": 40.0,
        "prior_days": 63,
    },
    "volume": {
        "name": "Volume Scan",
        "min_price": 30.0,
        "min_change_pct": 3.0,
        "avg_volume_days": 20,
        "min_avg_volume": 200_000.0,
        "min_rvol": 3.0,
    },
    "one_month": {
        "name": "1 Month High",
        "min_price": 30.0,
        "avg_volume_days": 20,
        "min_avg_volume": 200_000.0,
        "lookback_days": 21,
    },
    "three_month": {
        "name": "3 Month Performance",
        "min_price": 30.0,
        "avg_volume_days": 20,
        "min_avg_volume": 200_000.0,
        "lookback_days": 63,
        "min_return_pct": 30.0,
    },
    "one_month_perf": {
        "name": "1 Month Performance",
        "min_price": 30.0,
        "avg_volume_days": 20,
        "min_avg_volume": 200_000.0,
        "lookback_days": 21,
        "min_return_pct": 30.0,
    },
}
# ------------------------------------------------------------


def num(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if np.isfinite(x) else None


def true_range(high, low, close):
    """True Range(1), matching the supplied Chartink expression."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])


def adx_wilder(high, low, close, period=14):
    """Wilder ADX(14), using standard directional movement and ATR smoothing."""
    high_s = pd.Series(high, dtype=float)
    low_s = pd.Series(low, dtype=float)
    close_s = pd.Series(close, dtype=float)

    up_move = high_s.diff()
    down_move = -low_s.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high_s.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high_s.index,
    )

    prev_close = close_s.shift(1)
    tr = pd.concat([
        high_s - low_s,
        (high_s - prev_close).abs(),
        (low_s - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_sm = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_sm = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_sm / atr.replace(0, np.nan)
    minus_di = 100 * minus_sm / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.to_numpy()


def load_events():
    """
    Events remain optional.

    The FYERS pipeline currently has no exchange-event adjustment
    detector equivalent to the old Bhavcopy builder, so the scanner
    must still work without events.csv.
    """
    path = DATA / "events.csv"

    if not path.exists():
        return set(), set()

    try:
        ev = pd.read_csv(path)
    except Exception:
        return set(), set()

    if ev.empty or "key" not in ev.columns or "method" not in ev.columns:
        return set(), set()

    review_keys = set(
        ev.loc[
            ev["method"].astype(str).eq("review"),
            "key",
        ].astype(str)
    )

    recent_keys = set()

    if "date" in ev.columns:
        try:
            dates = pd.to_datetime(
                ev["date"],
                errors="coerce",
            )
            cutoff = dates.max() - pd.Timedelta(days=60)
            recent_keys = set(
                ev.loc[
                    dates >= cutoff,
                    "key",
                ].astype(str)
            )
        except Exception:
            recent_keys = set()

    return review_keys, recent_keys


def base_row(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    adx,
    review_keys,
    recent_keys,
):
    i = len(g) - 1

    adr = (
        pd.Series(tr)
        .rolling(20)
        .mean()
        .iloc[i]
        / c[i]
        * 100
    )

    value_rupees = c[i] * v[i]

    row0 = g.iloc[i]

    def return_from_bars(days):
        if i < days or not np.isfinite(c[i - days]) or c[i - days] == 0:
            return np.nan
        return (c[i] / c[i - days] - 1) * 100

    note = []

    if key in review_keys:
        note.append(
            "unadjusted corporate action in history, check chart"
        )

    if key in recent_keys:
        note.append(
            "corporate-action event in last 60 days"
        )

    return {
        "key": key,
        "symbol": str(row0["symbol"]),
        "exch": str(row0["exch"]),
        "name": str(row0.get("name", "")),
        "close": num(c[i]),
        "chg": num(
            (c[i] / c[i - 1] - 1) * 100
            if i >= 1 and c[i - 1] != 0
            else np.nan,
            2,
        ),
        "volume": num(v[i], 0),
        "value_cr": num(value_rupees / 1e7, 2),
        # Display-only momentum returns. These do NOT participate in any scan filter.
        # Keep the same trading-day windows already used by the 1M/3M/6M breakout scan.
        "ret1m": num(return_from_bars(31), 2),
        "ret3m": num(return_from_bars(93), 2),
        "ret6m": num(return_from_bars(186), 2),
        "adr": num(adr, 2),
        "adx": num(adx[i], 2),
        "note": "; ".join(note),
        "spark": [
            num(x)
            for x in c[-40:]
        ],
    }


def analyse_m136(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    Exact:

      SMA(True Range(1),20) / Close * 100 >= 3
      AND Close > EMA60
      AND (
            Close > previous 31-day max Close
            OR Close > previous 93-day max Close
            OR Close > previous 186-day max Close
          )

    Today is excluded from each rolling maximum.
    """
    n = len(g)
    if n <= 186:
        return None

    ema60 = (
        pd.Series(c)
        .ewm(span=60, adjust=False)
        .mean()
        .to_numpy()
    )

    adr = (
        pd.Series(tr)
        .rolling(20)
        .mean()
        .iloc[-1]
        / c[-1]
        * 100
    )

    if not np.isfinite(adr):
        return None

    if adr < CFG["m136"]["adr_min_pct"]:
        return None

    if not c[-1] > ema60[-1]:
        return None

    hits = []

    for label, days in CFG["m136"]["breakout_windows"].items():
        if n <= days:
            continue

        prior_max = np.max(
            c[-days - 1:-1]
        )

        if c[-1] > prior_max:
            hits.append(label)

    if not hits:
        return None

    def pct_from_bars(days):
        old_close = c[-1 - days]
        if old_close == 0:
            return np.nan
        return (c[-1] / old_close - 1) * 100

    return {
        **base,
        "breakout": " / ".join(hits),
        "ema60": num(ema60[-1], 2),
        "tr_adr": num(adr, 2),
        "signal": "Close above prior high"
            + (f" ({' / '.join(hits)})"),
    }


def analyse_m30(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    Exact:

      (Close - 63-days-ago Close) / 63-days-ago Close * 100 > 30
      AND Close > EMA75
      AND SMA(True Range(1),20) / Close * 100 >= 3
      AND Close * Volume > 10,000,000
    """
    n = len(g)

    if n <= 63:
        return None

    close_3m = c[-1 - CFG["m30"]["return_days"]]

    if close_3m == 0:
        return None

    ret3m = (
        c[-1] / close_3m - 1
    ) * 100

    ema75 = (
        pd.Series(c)
        .ewm(span=75, adjust=False)
        .mean()
        .to_numpy()
    )

    adr = (
        pd.Series(tr)
        .rolling(20)
        .mean()
        .iloc[-1]
        / c[-1]
        * 100
    )

    value_rupees = c[-1] * v[-1]

    if not np.isfinite(ret3m):
        return None

    if ret3m <= CFG["m30"]["min_return_pct"]:
        return None

    if not c[-1] > ema75[-1]:
        return None

    if not np.isfinite(adr) or adr < CFG["m30"]["adr_min_pct"]:
        return None

    if value_rupees <= CFG["m30"]["min_value_rupees"]:
        return None

    return {
        **base,
        "ema75": num(ema75[-1], 2),
        "tr_adr": num(adr, 2),
        "signal": "3M gain > 30%",
    }


def analyse_htf(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    Exact:

      Close / 60-days-ago Close > 1.5
      AND Close >= previous 252-day max High * 0.85
      AND Volume < SMA(Volume,20) * 1.5
      AND Close > 50

    Today is excluded from the 252-day highest-high.
    """
    n = len(g)

    if n <= 252:
        return None

    close_60 = c[-1 - CFG["htf"]["lookback_days"]]

    if close_60 == 0:
        return None

    multiple = c[-1] / close_60

    prior_high = np.max(
        h[-1 - CFG["htf"]["high_lookback_days"]:-1]
    )

    volume_sma = (
        pd.Series(v)
        .rolling(CFG["htf"]["volume_sma_days"])
        .mean()
        .iloc[-1]
    )

    vol_ratio = (
        v[-1] / volume_sma
        if volume_sma > 0
        else np.nan
    )

    off_high = (
        1 - c[-1] / prior_high
    ) * 100

    if multiple <= CFG["htf"]["min_return_multiple"]:
        return None

    if c[-1] < prior_high * (
        1 - CFG["htf"]["max_high_distance_pct"] / 100
    ):
        return None

    if not (
        np.isfinite(vol_ratio)
        and vol_ratio < CFG["htf"]["max_volume_ratio"]
    ):
        return None

    if c[-1] <= CFG["htf"]["min_close"]:
        return None

    return {
        **base,
        "gain60": num((multiple - 1) * 100, 2),
        "multiple60": num(multiple, 3),
        "off_high": num(off_high, 2),
        "prior_252_high": num(prior_high, 2),
        "vol_ratio": num(vol_ratio, 2),
        "signal": "High Tight Flag candidate",
    }


def analyse_volume(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    NSE-only volume momentum scan:

      Close >= 30
      AND daily change >= 3%
      AND 20-day average volume >= 200,000 shares
      AND current volume / 20-day average volume > 3x
    """
    n = len(g)
    if n < CFG["volume"]["avg_volume_days"]:
        return None

    if str(g.iloc[-1]["exch"]).upper() != "NSE":
        return None

    close = c[-1]
    prev_close = c[-2] if n >= 2 else np.nan
    chg = (close / prev_close - 1) * 100 if prev_close else np.nan

    avg_volume = (
        pd.Series(v)
        .rolling(CFG["volume"]["avg_volume_days"])
        .mean()
        .iloc[-1]
    )

    if not (
        np.isfinite(close)
        and close >= CFG["volume"]["min_price"]
        and np.isfinite(chg)
        and chg >= CFG["volume"]["min_change_pct"]
        and np.isfinite(avg_volume)
        and avg_volume >= CFG["volume"]["min_avg_volume"]
        and np.isfinite(v[-1])
        and avg_volume > 0
    ):
        return None

    rvol = v[-1] / avg_volume
    if not np.isfinite(rvol) or rvol <= CFG["volume"]["min_rvol"]:
        return None

    return {
        **base,
        "avg_volume": num(avg_volume, 0),
        "rvol": num(rvol, 2),
        "signal": "Volume surge > 3x average",
    }


def analyse_three_month(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    NSE-only 3-month performance scan:

      Close >= 30
      AND 20-day average volume >= 200,000 shares
      AND 3-month price gain >= 30%.

    Performance uses 63 trading sessions.
    """
    n = len(g)
    cfg = CFG["three_month"]

    if n <= cfg["lookback_days"]:
        return None

    if str(g.iloc[-1]["exch"]).upper() != "NSE":
        return None

    close = c[-1]
    old_close = c[-1 - cfg["lookback_days"]]
    avg_volume = (
        pd.Series(v)
        .rolling(cfg["avg_volume_days"])
        .mean()
        .iloc[-1]
    )

    ret3m = (
        (close / old_close - 1) * 100
        if np.isfinite(old_close) and old_close != 0
        else np.nan
    )

    if not (
        np.isfinite(close)
        and close >= cfg["min_price"]
        and np.isfinite(avg_volume)
        and avg_volume >= cfg["min_avg_volume"]
        and np.isfinite(ret3m)
        and ret3m >= cfg["min_return_pct"]
    ):
        return None

    return {
        **base,
        "ret3m_perf": num(ret3m, 2),
        "avg_volume": num(avg_volume, 0),
        "signal": "3-month gain >= 30%",
    }


def analyse_one_month_perf(
    key, g, c, h, l, v, tr, base,
):
    """NSE-only 1-month performance scan: price >= 30, 20-day average
    volume >= 200,000 shares, and gain >= 30% over 21 trading sessions."""
    n=len(g)
    cfg=CFG["one_month_perf"]
    if n <= cfg["lookback_days"] or str(g.iloc[-1]["exch"]).upper() != "NSE":
        return None
    close=c[-1]
    old_close=c[-1-cfg["lookback_days"]]
    avg_volume=pd.Series(v).rolling(cfg["avg_volume_days"]).mean().iloc[-1]
    ret1m_perf=((close/old_close)-1)*100 if np.isfinite(old_close) and old_close != 0 else np.nan
    if not (np.isfinite(close) and close >= cfg["min_price"] and
            np.isfinite(avg_volume) and avg_volume >= cfg["min_avg_volume"] and
            np.isfinite(ret1m_perf) and ret1m_perf >= cfg["min_return_pct"]):
        return None
    return {**base, "ret1m_perf": num(ret1m_perf,2),
            "avg_volume": num(avg_volume,0),
            "signal": "1-month gain >= 30%"}


def analyse_one_month(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    NSE-only 1-month scan:

      Close >= 30
      AND 20-day average volume >= 200,000 shares
      AND current close is a new 1-month high.

    The 1-month high uses the previous 21 trading sessions and excludes
    today's close, so "new high" means today's close is above that prior
    21-session closing high.
    """
    n = len(g)
    cfg = CFG["one_month"]

    if n <= cfg["lookback_days"]:
        return None

    if str(g.iloc[-1]["exch"]).upper() != "NSE":
        return None

    close = c[-1]
    avg_volume = (
        pd.Series(v)
        .rolling(cfg["avg_volume_days"])
        .mean()
        .iloc[-1]
    )
    prior_1m_high = np.max(c[-1 - cfg["lookback_days"]:-1])

    if not (
        np.isfinite(close)
        and close >= cfg["min_price"]
        and np.isfinite(avg_volume)
        and avg_volume >= cfg["min_avg_volume"]
        and np.isfinite(prior_1m_high)
        and close > prior_1m_high
    ):
        return None

    return {
        **base,
        "avg_volume": num(avg_volume, 0),
        "one_month_high": num(prior_1m_high, 2),
        "signal": "New 1-month high",
    }


def analyse_ep(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    Episodic Pivot based on the earlier scanner logic.

    Checks today and the previous two sessions for:
      - gap-up open of 8% to 40% versus previous close
      - volume >= 3x prior 20-session average
      - close in the upper 40% of the day's range
      - prior 3-month gain <= 40%
      - latest close >= the EP-day open

    Entry = highest high from EP day through today.
    Stop  = EP-day low.
    """
    cfg = CFG["ep"]

    if not cfg["enabled"]:
        return None

    n = len(g)
    lookback = int(cfg["lookback_days"])

    if n < max(64, lookback + 1):
        return None

    opens = g["open"].to_numpy(float)

    for k in range(n - 1, max(-1, n - lookback - 1), -1):
        if k < 64:
            break

        avg_v = float(pd.Series(v[k - 20:k]).mean())
        prev_close = c[k - 1]

        if not np.isfinite(avg_v) or avg_v <= 0:
            continue

        if not np.isfinite(prev_close) or prev_close == 0:
            continue

        gap = (opens[k] / prev_close - 1) * 100

        if not (
            np.isfinite(gap)
            and cfg["min_gap_pct"] <= gap <= cfg["max_gap_pct"]
        ):
            continue

        vol_ratio = v[k] / avg_v

        day_range = h[k] - l[k]
        close_pos = (
            (c[k] - l[k]) / day_range
            if np.isfinite(day_range) and day_range > 0
            else 1.0
        )

        old_idx = k - int(cfg["prior_days"])
        if old_idx < 0 or not np.isfinite(c[old_idx]) or c[old_idx] == 0:
            continue

        prior_3m = (c[k - 1] / c[old_idx] - 1) * 100

        if not (
            np.isfinite(vol_ratio)
            and vol_ratio >= cfg["min_vol_ratio"]
            and np.isfinite(close_pos)
            and close_pos >= cfg["min_close_pos"]
            and np.isfinite(prior_3m)
            and prior_3m <= cfg["max_prior_3m_pct"]
            and c[-1] >= opens[k]
        ):
            continue

        entry = float(np.max(h[k:n]))
        stop = float(l[k])

        if entry <= stop:
            continue

        days_ago = (n - 1) - k

        return {
            **base,
            "entry": num(entry, 2),
            "stop": num(stop, 2),
            "risk": num((entry - stop) / entry * 100, 1),
            "state": "Today" if days_ago == 0 else f"{days_ago}d ago",
            "gap": num(gap, 1),
            "vol_ratio": num(vol_ratio, 1),
            "prior_3m": num(prior_3m, 0),
            "days_ago": int(days_ago),
            "signal": (
                "Gap-up on heavy volume: buy above EP-day high, "
                "stop at EP-day low"
            ),
        }

    return None


def analyse_high52w(
    key,
    g,
    c,
    h,
    l,
    v,
    tr,
    base,
):
    """
    Exact:

      Close * Volume > 10,000,000
      AND SMA(True Range(1),20) / Close * 100 >= 3
      AND Close >= previous 252-day max High * 0.90
      AND Close <= previous 252-day max High

    Today is excluded from the prior 252-day max.
    """
    n = len(g)

    if n <= 252:
        return None

    value_rupees = c[-1] * v[-1]

    adr = (
        pd.Series(tr)
        .rolling(20)
        .mean()
        .iloc[-1]
        / c[-1]
        * 100
    )

    prior_high = np.max(
        h[-1 - CFG["high52w"]["high_lookback_days"]:-1]
    )

    off_high = (
        1 - c[-1] / prior_high
    ) * 100

    if value_rupees <= CFG["high52w"]["value_min_rupees"]:
        return None

    if not np.isfinite(adr) or adr < CFG["high52w"]["adr_min_pct"]:
        return None

    if c[-1] < prior_high * (
        1 - CFG["high52w"]["max_high_distance_pct"] / 100
    ):
        return None

    if c[-1] > prior_high:
        return None

    return {
        **base,
        "off_high": num(off_high, 2),
        "prior_252_high": num(prior_high, 2),
        "tr_adr": num(adr, 2),
        "signal": "Within 10% of prior 52-week high",
    }


def analyse_stock(
    key,
    g,
    review_keys,
    recent_keys,
):
    g = g.sort_values("date").reset_index(drop=True)

    n = len(g)

    c = g["close"].to_numpy(float)
    h = g["high"].to_numpy(float)
    l = g["low"].to_numpy(float)
    v = g["volume"].to_numpy(float)

    c = np.asarray(c)
    h = np.asarray(h)
    l = np.asarray(l)
    v = np.asarray(v)

    if not (
        len(g)
        and np.isfinite(c[-1])
        and np.isfinite(h[-1])
        and np.isfinite(l[-1])
        and np.isfinite(v[-1])
    ):
        return {}

    tr = true_range(h, l, c)
    adx = adx_wilder(h, l, c, 14)

    # Volume scan is evaluated before the common momentum filters.
    base = base_row(
        key, g, c, h, l, v, tr, adx, review_keys, recent_keys
    )
    out = {}
    r = analyse_volume(key, g, c, h, l, v, tr, base)
    if r:
        out["volume"] = r

    r = analyse_three_month(key, g, c, h, l, v, tr, base)
    if r:
        out["three_month"] = r

    r = analyse_one_month(key, g, c, h, l, v, tr, base)
    if r:
        out["one_month"] = r

    r = analyse_one_month_perf(key, g, c, h, l, v, tr, base)
    if r:
        out["one_month_perf"] = r

    if n <= max(CFG["common"]["return_windows"]):
        return out

    latest_adx = adx[-1]
    if not np.isfinite(latest_adx) or latest_adx < CFG["common"]["min_adx"]:
        return out

    returns = []
    for days in CFG["common"]["return_windows"]:
        old_close = c[-1 - days]
        if old_close == 0 or not np.isfinite(old_close):
            returns.append(np.nan)
        else:
            returns.append((c[-1] / old_close - 1) * 100)

    if not any(
        np.isfinite(r) and r >= CFG["common"]["min_return_pct"]
        for r in returns
    ):
        return out

    # Tradability filter:
    # First require a minimum 20-session average traded value so the
    # scanner focuses on stocks that can realistically be entered/exited.
    # Then apply the stricter locked/UC pattern check only as an additional
    # rejection condition. Occasional UC days in an otherwise liquid stock
    # are therefore kept.
    tcfg = CFG["common"]["tradability"]
    lookback = tcfg["lookback_days"]

    if n >= lookback + 1:
        recent_close = c[-lookback:]
        recent_open = g["open"].to_numpy(float)[-lookback:]
        recent_volume = v[-lookback:]

        trading_value = recent_close * recent_volume
        avg_value = float(np.nanmean(trading_value))

        if (
            not np.isfinite(avg_value)
            or avg_value < tcfg["min_avg_value_rupees"]
        ):
            return out

        # Detect persistent locked/UC behaviour after the basic liquidity
        # gate. This does not reject a liquid stock merely for having a few
        # upper-circuit sessions.
        daily_change = (
            recent_close[1:] / recent_close[:-1] - 1
        ) * 100

        open_close_pct = (
            np.abs(recent_open[1:] - recent_close[1:])
            / recent_close[1:]
        ) * 100

        near_uc_days = int(
            np.sum(daily_change >= tcfg["near_uc_pct"])
        )
        open_close_days = int(
            np.sum(open_close_pct <= tcfg["max_open_close_pct"])
        )

        if (
            near_uc_days >= tcfg["min_near_uc_days"]
            and open_close_days >= tcfg["min_open_close_days"]
        ):
            return out

    r = analyse_ep(
        key,
        g,
        c,
        h,
        l,
        v,
        tr,
        base,
    )
    if r:
        out["ep"] = r

    r = analyse_m136(
        key,
        g,
        c,
        h,
        l,
        v,
        tr,
        base,
    )
    if r:
        out["m136"] = r

    r = analyse_m30(
        key,
        g,
        c,
        h,
        l,
        v,
        tr,
        base,
    )
    if r:
        out["m30"] = r

    r = analyse_htf(
        key,
        g,
        c,
        h,
        l,
        v,
        tr,
        base,
    )
    if r:
        out["htf"] = r

    r = analyse_high52w(
        key,
        g,
        c,
        h,
        l,
        v,
        tr,
        base,
    )
    if r:
        out["high52w"] = r

    return out


def build_chart_data(df, keys):
    """
    Keep the dashboard chart payload compact.
    The scanner database itself still contains full history.
    """
    chart = {}

    for key in sorted(keys):
        g = df[df["key"] == key].sort_values("date").tail(252)

        if g.empty:
            continue

        close = g["close"].to_numpy(float)
        ss = pd.Series(close)

        e11 = (
            ss.ewm(span=11, adjust=False)
            .mean()
            .to_numpy()
        )
        e21 = (
            ss.ewm(span=21, adjust=False)
            .mean()
            .to_numpy()
        )
        s50 = (
            ss.rolling(50)
            .mean()
            .to_numpy()
        )

        chart[key] = {
            "symbol": str(g.iloc[-1]["symbol"]),
            "exch": str(g.iloc[-1]["exch"]),
            "dates": g["date"].astype(str).tolist(),
            "open": [num(x) for x in g["open"]],
            "high": [num(x) for x in g["high"]],
            "low": [num(x) for x in g["low"]],
            "close": [num(x) for x in g["close"]],
            "volume": [num(x, 0) for x in g["volume"]],
            "ema11": [num(x) for x in e11],
            "ema21": [num(x) for x in e21],
            "sma50": [
                num(x)
                for x in s50
            ],
        }

    return chart


def main():
    price_path = DATA / "prices_fyers.csv.gz"

    if not price_path.exists():
        raise FileNotFoundError(
            f"Missing {price_path}. Build the FYERS prices database first."
        )

    df = pd.read_csv(
        price_path
    )

    required = {
        "key",
        "date",
        "symbol",
        "exch",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "prices.csv.gz is missing columns: "
            + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "key",
            "date",
            "close",
            "high",
            "low",
            "volume",
        ]
    )

    df = df.sort_values(
        ["key", "date"]
    ).reset_index(drop=True)

    review_keys, recent_keys = load_events()

    last_date = df["date"].max()

    hits = {
        "m136": [],
        "m30": [],
        "htf": [],
        "high52w": [],
        "ep": [],
        "volume": [],
        "one_month": [],
        "three_month": [],
        "one_month_perf": [],
    }

    total = 0
    traded = 0

    for key, g in df.groupby(
        "key",
        sort=False,
    ):
        total += 1

        if g["date"].iloc[-1] != last_date:
            continue

        traded += 1

        result = analyse_stock(
            str(key),
            g,
            review_keys,
            recent_keys,
        )

        for scan_name, row in result.items():
            hits[scan_name].append(row)

    # Stable useful ordering for dashboard.
    hits["m136"].sort(
        key=lambda r: (
            -float(r.get("close", 0) or 0)
        )
    )

    hits["m30"].sort(
        key=lambda r: (
            -float(r.get("ret3m", 0) or 0)
        )
    )

    hits["htf"].sort(
        key=lambda r: (
            -float(r.get("gain60", 0) or 0)
        )
    )

    hits["high52w"].sort(
        key=lambda r: (
            float(r.get("off_high", 999) or 999)
        )
    )

    hits["ep"].sort(
        key=lambda r: (
            int(r.get("days_ago", 999) or 999),
            -float(r.get("vol_ratio", 0) or 0),
        )
    )

    hits["volume"].sort(
        key=lambda r: (
            -float(r.get("rvol", 0) or 0)
        )
    )

    hits["one_month"].sort(
        key=lambda r: (
            -float(r.get("close", 0) or 0)
        )
    )

    hits["three_month"].sort(
        key=lambda r: (
            -float(r.get("ret3m_perf", 0) or 0)
        )
    )

    hits["one_month_perf"].sort(
        key=lambda r: (
            -float(r.get("ret1m_perf", 0) or 0)
        )
    )

    passed_keys = {
        r["key"]
        for rows in hits.values()
        for r in rows
    }

    chart = build_chart_data(
        df,
        passed_keys,
    )

    # Scanner config is exported into results.json so the dashboard
    # has one machine-readable definition of the applied rules.
    config_out = {
        **CFG,
        "definitions": {
            "m136": (
                "SMA(True Range(1),20) / Close * 100 >= 3 AND "
                "Close > EMA60 AND "
                "(Close > previous 31-day max Close OR "
                "Close > previous 93-day max Close OR "
                "Close > previous 186-day max Close)"
            ),
            "m30": (
                "(Close - 63-days-ago Close) / 63-days-ago Close * 100 > 30 "
                "AND Close > EMA75 AND "
                "SMA(True Range(1),20) / Close * 100 >= 3 AND "
                "Close * Volume > ₹10,000,000"
            ),
            "htf": (
                "Close / 60-days-ago Close > 1.5 AND "
                "Close >= previous 252-day max High * 0.85 AND "
                "Volume < SMA(Volume,20) * 1.5 AND "
                "Close > ₹50"
            ),
            "high52w": (
                "Close * Volume > ₹10,000,000 AND "

                "SMA(True Range(1),20) / Close * 100 >= 3 AND "
                "Close >= previous 252-day max High * 0.90 AND "
                "Close <= previous 252-day max High"
            ),
            "ep": (
                "Check today and previous 2 sessions for an Open vs previous close "
                "gap of 8%-40% AND volume >= 3x prior 20-session average AND "
                "close in upper 40% of the day's range AND prior 3-month gain <= 40% "
                "AND latest close >= EP-day open. Entry = highest high from EP day "
                "through today; stop = EP-day low."
            ),
            "volume": (
                "NSE only AND Close >= ₹30 AND daily change >= 3% AND "
                "20-day average volume >= 200,000 shares AND "
                "current volume / 20-day average volume > 3x"
            ),
            "one_month": (
                "NSE only AND Close >= ₹30 AND 20-day average volume >= 200,000 "
                "shares AND Close > previous 21-trading-day closing high"
            ),
            "three_month": (
                "NSE only AND Close >= ₹30 AND 20-day average volume >= 200,000 "
                "shares AND 3-month price gain >= 30% over 63 trading sessions"
            ),
            "one_month_perf": (
                "NSE only AND Close >= ₹30 AND 20-day average volume >= 200,000 "
                "shares AND 1-month price gain >= 30% over 21 trading sessions"
            ),
        },
    }

    stats = {
        "stocks": int(total),
        "traded_last_day": int(traded),
        "with_a_signal": int(
            len(passed_keys)
        ),
    }

    out = {
        "asof": last_date.strftime("%Y-%m-%d"),
        "generated": datetime.now().isoformat(
            timespec="seconds"
        ),
        "config": config_out,
        "stats": stats,
        "scans": hits,
    }

    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    (DATA / "results.json").write_text(
        json.dumps(
            out,
            separators=(",", ":"),
        )
    )

    (DATA / "chart_data.json").write_text(
        json.dumps(
            chart,
            separators=(",", ":"),
        )
    )

    print(
        f"As of {out['asof']}. "
        f"Stocks: {total:,}   "
        f"traded on last day: {traded:,}"
    )

    for name, rows in hits.items():
        print(
            f"\n== {name.upper()}: {len(rows)} hits"
        )

        for r in rows[:10]:
            print(
                f"  {r['exch']}:{r['symbol']:<14} "
                f"close {r['close']:>9}  "
                f"chg {r['chg']:>7}%"
            )


if __name__ == "__main__":
    main()
