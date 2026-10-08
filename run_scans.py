"""
Stage 3 (v2): three scans on the adjusted prices from build_prices.py.

  1. Strong Stage 2   : Minervini-style trend template + relative strength rank
  2. Pullback         : close within +/-2% of EMA11 / EMA21 / SMA50 (tightness only)
  3. High tight flag  : big run-up, then a tight flag (forming or triggered)
  4. Breakout         : top performers in a tight, higher-low flag (Qullamaggie style).
                        Setup = trigger above the flag high. Triggered = closed above it on volume.
  5. Episodic pivot   : gap-up on heavy volume after a relatively quiet 6-month period

Run:  python run_scans.py
Out:  data/results.json   (the dashboard reads this)
All settings are in CFG below.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")

# ------------------------- settings -------------------------
CFG = {
    "min_bars": 80,                 # minimum history for any scan
    "min_adr_pct": 3.0,             # 20-day average daily range %, applies to ALL scans
    "min_avg_value_cr": 1.0,        # 20-day avg traded value in rupees crore, applies to ALL scans
    "min_adx": 20,                   # remove stocks with ADX below 20 (trend too weak/choppy)
    "max_risk_pct": None,           # pullback / flag only: e.g. 10 drops setups with stop wider than 10%
    "stage2": {                     # needs 253+ bars (SMA200 + 52-week range + 12-month RS)
        "slope_days": 22,           # 200-day average must be higher than this many days ago
        "min_above_low_pct": 30,    # close at least 30% above the 52-week low
        "max_below_high_pct": 25,   # close within 25% of the 52-week high
        "min_rs": 70,               # relative strength rank 1-99 among liquid stocks
    },
    "pullback": {"tight_pct": 2.0, "levels": ["EMA11", "EMA21", "SMA50"]},
    "momentum": {                   # close now vs close N trading days ago
        "enabled": False,                       # off: set True to require the returns below
        "match": "any",                         # "any" = one of the three is enough, "all" = every one
        "apply_to": ["stage2", "pullback"],     # the flag scan has its own run-up rule
        "days": {"1m": 21, "3m": 63, "6m": 126},
        "min_pct": {"1m": 30, "3m": 50, "6m": 100},
    },
    "ema_then": {                   # stock must also have closed above its EMA 21 back then
        "enabled": True,
        "days_ago": 21,             # 21 trading days is about 30 calendar days
        "apply_to": ["stage2", "pullback"],
    },
    "breakout": {                   # leaders in a tight flag, buy the break of the flag high
        "enabled": True,
        "min_adr_pct": 4.0,         # needs a stock that moves: 20-day average daily range
        "min_rank": 90,             # momentum rank 1-99: best of the 1M / 3M / 6M return ranks
        "lookback_days": 30,        # flag high must be within this many days
        "flag_min_days": 5,         # flag at least a week old (no one-day "flags")
        "runup_days": 63,           # look this far before the flag high for the prior run-up
        "min_runup_pct": 30,        # low-to-flag-high gain before the flag
        "max_depth_pct": 25,        # flag may not dip more than this from its high
        "max_below_pct": 7,         # setup: close within this % under the flag high
        "tight_ratio": 0.9,         # check 1: last-5-day range <= 0.9 x the average of the last 30 days
        "vol_dry_ratio": 0.9,       # check 3: last-5-day volume <= 0.9 x the 50 days before that
        "min_checks": 2,            # need 2 of 3: tight range, higher lows, volume dry-up
        "break_vol_ratio": 1.5,     # triggered: today's volume >= 1.5 x its 20-day average
        "max_extended_pct": 8,      # triggered: skip if the close is already >8% over the flag high
        "max_risk_adr": 1.5,        # skip if stop is wider than 1.5 x ADR
    },
    "ep": {                         # India-focused, moderately loose episodic pivot
        "enabled": True,
        "lookback_days": 3,         # today and the 2 days before
        "min_gap_pct": 6,           # open vs previous close
        "max_gap_pct": 35,          # avoid extreme/data/corporate-action gaps
        "min_vol_ratio": 2,         # volume >= 2 x 20-day average
        "min_close_pos": 0.55,      # close in the upper 45% of the day's range
        "prior_days": 126,           # about 6 months of trading days
        "max_prior_move_pct": 50,   # allow stocks that moved up/down <= 50% before the event
    },
    "htf": {"flagpole_days": 40, "min_gain_pct": 70, "flag_min_days": 15,
            "flag_max_days": 25, "max_pullback_pct": 25},
}
# ------------------------------------------------------------


def num(x, d=2):
    return round(float(x), d) if np.isfinite(x) else None


def analyse(key, g, review_keys, recent_keys):
    """Returns (info, hits). info feeds the RS ranking, hits are scan results."""
    n = len(g)
    info = {"key": key, "rs_raw": None, "perf": None}
    if n < CFG["min_bars"]:
        return info, {}
    c, h, v = (g[k].to_numpy(float) for k in ("close", "high", "volume"))
    o = g["open"].to_numpy(float)
    l = g["low"].to_numpy(float)
    l = np.where(l > 0, l, c)
    val = g["value"].to_numpy(float)
    i = n - 1

    M = CFG["momentum"]
    mom = None
    if n > max(M["days"].values()):
        mom = {k: (c[i] / c[i - d] - 1) * 100 for k, d in M["days"].items()}
    # Universal ADX filter: remove weak/choppy stocks with ADX < 20.
    # Wilder-style 14-period ADX.
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    period = 14
    tr_s = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_s = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_s = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_s / tr_s.replace(0, np.nan)
    minus_di = 100 * minus_s / tr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean().to_numpy()
    if not np.isfinite(adx[i]) or adx[i] < CFG["min_adx"]:
        return info, {}

    pick = any if M["match"] == "any" else all
    mom_ok = mom is not None and pick(mom[k] >= M["min_pct"][k] for k in M["min_pct"])
    gate = {nm: (mom_ok if (M["enabled"] and nm in M["apply_to"]) else True)
            for nm in ("stage2", "pullback", "htf")}

    adr = (((h[-20:] - l[-20:]) / c[-20:]) * 100).mean()
    avg_val_cr = val[-20:].mean() / 1e7
    liquid = avg_val_cr >= CFG["min_avg_value_cr"]
    if liquid and n >= 253:             # IBD-style weighted 3/6/9/12 month return
        r = lambda d: c[i] / c[i - d] - 1
        info["rs_raw"] = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
    if liquid and n > max(M["days"].values()):    # 1M / 3M / 6M returns, ranked later in main()
        info["perf"] = {k: c[i] / c[i - d] - 1 for k, d in M["days"].items()}
    if not liquid or adr < CFG["min_adr_pct"]:
        return info, {}

    s = pd.Series(c)
    ema11 = s.ewm(span=11, adjust=False).mean().to_numpy()
    ema21 = s.ewm(span=21, adjust=False).mean().to_numpy()
    ema10 = s.ewm(span=10, adjust=False).mean().to_numpy()
    ema20 = s.ewm(span=20, adjust=False).mean().to_numpy()
    sma50 = s.rolling(50).mean().to_numpy()

    T = CFG["ema_then"]
    j = i - T["days_ago"]
    then_ok = bool(c[j] > ema21[j])

    def passes(name):               # momentum gate AND the EMA 21 'then' check, where they apply
        return gate[name] and (then_ok or not (T["enabled"] and name in T["apply_to"]))

    row0 = g.iloc[-1]
    note = []
    if key in review_keys:
        note.append("unadjusted corporate action in history, check chart")
    if key in recent_keys:
        note.append("split/bonus adjusted in last 60 days")
    base = dict(
        key=key, symbol=row0["symbol"], exch=row0["exch"], name=row0["name"],
        close=num(c[i]), chg=num((c[i] / c[i - 1] - 1) * 100), adr=num(adr, 1),
        value_cr=num(avg_val_cr, 1), ema11=num(ema11[i]), ema21=num(ema21[i]),
        sma50=num(sma50[i]), adx=num(adx[i], 1), spark=[num(x) for x in c[-40:]], note="; ".join(note),
        **{f"ret{k}": (num(mom[k], 0) if mom else None) for k in ("1m", "3m", "6m")},
    )
    out = {}

    def finish(name, entry, stop, **extra):
        if entry <= stop:
            return
        risk = (entry - stop) / entry * 100
        if CFG["max_risk_pct"] and risk > CFG["max_risk_pct"]:
            return
        out[name] = {**base, "entry": num(entry), "stop": num(stop), "risk": num(risk, 1), **extra}

    # ---- 1. strong Stage 2 (rank filter is applied after all stocks are scored) ----
    if n >= 253 and passes("stage2"):
        st = CFG["stage2"]
        sma150 = s.rolling(150).mean().to_numpy()
        sma200 = s.rolling(200).mean().to_numpy()
        hi52, lo52 = h[-252:].max(), l[-252:].min()
        if (c[i] > sma150[i] and c[i] > sma200[i] and sma150[i] > sma200[i]
                and sma50[i] > sma150[i] and sma50[i] > sma200[i] and c[i] > sma50[i]
                and sma200[i] > sma200[i - st["slope_days"]]
                and c[i] >= lo52 * (1 + st["min_above_low_pct"] / 100)
                and c[i] >= hi52 * (1 - st["max_below_high_pct"] / 100)):
            out["stage2"] = {**base, "off_high": num((1 - c[i] / hi52) * 100, 1),
                             "above_low": num((c[i] / lo52 - 1) * 100, 0),
                             "sma200_slope": num((sma200[i] / sma200[i - st["slope_days"]] - 1) * 100, 1),
                             "signal": "Stage 2 uptrend"}

    # ---- 4. breakout: leader in a tight flag, buy the break of the flag high ----
    B = CFG["breakout"]
    if B["enabled"] and adr >= B["min_adr_pct"]:
        w = B["lookback_days"]
        pk = i - w + int(np.argmax(h[i - w:i]))          # flag high, before today
        top, flag_len = h[pk], i - pk
        runup = (top / l[max(0, pk - B["runup_days"]):pk + 1].min() - 1) * 100
        depth = (top - l[pk + 1:i + 1].min()) / top * 100 if flag_len >= 1 else 0.0

        def cons(e):
            """Is the flag ending at bar e tight? Returns (surfing the 10/20 EMA, [3 checks])."""
            rr = (h[:e + 1] - l[:e + 1]) / c[:e + 1] * 100
            ok = [bool(rr[-5:].mean() <= B["tight_ratio"] * rr[-30:].mean()),
                  bool(l[e - 4:e + 1].min() >= l[e - 9:e - 4].min()),
                  bool(v[e - 4:e + 1].mean() <= B["vol_dry_ratio"] * v[e - 54:e - 4].mean())]
            surf = bool(c[e] > ema20[e] and ema10[e] > ema20[e] > sma50[e])
            return surf, ok

        def brk(state, entry, stop, ok, vr):
            risk = (entry - stop) / entry * 100
            if risk / adr > B["max_risk_adr"]:
                return
            finish("breakout", entry, stop, state=state, runup=num(runup, 0), depth=num(depth, 1),
                   flag_days=int(flag_len), vol_ratio=num(vr, 1), risk_adr=num(risk / adr, 2),
                   checks=", ".join(nm for nm, k in zip(("tight range", "higher lows", "volume dry-up"), ok) if k),
                   signal="Closed above flag high on volume" if state == "Triggered"
                   else "Tight flag, trigger above its high")

        if flag_len >= B["flag_min_days"] and runup >= B["min_runup_pct"] and depth <= B["max_depth_pct"]:
            if c[i] <= top:                                    # still inside the flag
                surf, ok = cons(i)
                if surf and sum(ok) >= B["min_checks"] and c[i] >= top * (1 - B["max_below_pct"] / 100):
                    base_v = v[-55:-5].mean()
                    brk("Setup", top, l[-3:].min(), ok, v[-5:].mean() / base_v if base_v > 0 else 0)
            else:                                              # closed above the flag high today
                surf, ok = cons(i - 1)
                avg_v = v[i - 20:i].mean()
                vr = v[i] / avg_v if avg_v > 0 else 0
                pos = (c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 1.0
                if (surf and sum(ok) >= B["min_checks"] and vr >= B["break_vol_ratio"] and pos >= 0.5
                        and c[i] <= top * (1 + B["max_extended_pct"] / 100)):
                    brk("Triggered", top, l[i], ok, vr)

    # ---- 5. episodic pivot: gap-up on heavy volume after a quiet stretch ----
    E = CFG["ep"]
    if E["enabled"]:
        for k in range(i, i - E["lookback_days"], -1):         # most recent day first
            if k < 64:
                break
            avg_v = v[k - 20:k].mean()
            gap = (o[k] / c[k - 1] - 1) * 100
            if avg_v <= 0 or not (E["min_gap_pct"] <= gap <= E["max_gap_pct"]):
                continue
            vr = v[k] / avg_v
            pos = (c[k] - l[k]) / (h[k] - l[k]) if h[k] > l[k] else 1.0
            prior = (c[k - 1] / c[k - E["prior_days"]] - 1) * 100
            if (vr >= E["min_vol_ratio"] and pos >= E["min_close_pos"]
                    and abs(prior) <= E["max_prior_move_pct"] and c[i] >= o[k]):
                finish("ep", h[k:i + 1].max(), l[k], state="Today" if k == i else f"{i - k}d ago",
                       gap=num(gap, 1), vol_ratio=num(vr, 1), prior_3m=num(prior, 0), days_ago=int(i - k),
                       signal="Gap-up on heavy volume: buy above the EP-day high, stop at its low")
                break

    # pullback and flag also need the short-term trend stack
    if not (ema11[i] > ema21[i] > sma50[i] and c[i] > sma50[i]):
        return info, out

    # ---- 2. pullback: tightness only, close within +/-X% of a moving average ----
    p = CFG["pullback"]
    levels = {"EMA11": ema11[i], "EMA21": ema21[i], "SMA50": sma50[i]}
    best = min(((abs(c[i] / levels[k] - 1) * 100, k) for k in p["levels"]), key=lambda t: t[0])
    if best[0] <= p["tight_pct"] and passes("pullback"):
        k = best[1]
        finish("pullback", h[i], l[-5:].min(), level=k, dist=num((c[i] / levels[k] - 1) * 100, 2),
               signal=f"Within {p['tight_pct']:g}% of {k}")

    # ---- 3. high tight flag (relaxed) ----
    f = CFG["htf"]
    fmax = f["flag_max_days"]
    pk = i - fmax + int(np.argmax(h[i - fmax:i]))      # flag peak, before today
    flag_len = i - pk
    if f["flag_min_days"] <= flag_len <= fmax and passes("htf"):
        top = h[pk]
        gain = (top / l[max(0, pk - f["flagpole_days"]):pk + 1].min() - 1) * 100
        flag_low = l[pk + 1:i + 1].min()
        pb = (top - flag_low) / top * 100
        if gain >= f["min_gain_pct"] and pb <= f["max_pullback_pct"]:
            if c[i] > top:
                finish("htf", h[i], flag_low, state="Triggered", gain=num(gain, 0),
                       flag_days=int(flag_len), pullback=num(pb, 1), signal="Broke flag high")
            elif c[i] >= top * (1 - f["max_pullback_pct"] / 100):
                finish("htf", top, flag_low, state="Forming", gain=num(gain, 0),
                       flag_days=int(flag_len), pullback=num(pb, 1), signal="In flag, trigger above high")
    return info, out


def main():
    df = pd.read_csv(DATA / "prices_fyers.csv.gz").sort_values(["key", "date"])
    ev = pd.read_csv(DATA / "events.csv")
    dates = sorted(df["date"].unique())
    last = dates[-1]
    review_keys = set(ev.loc[ev["method"] == "review", "key"])
    recent_keys = set(ev.loc[(ev["method"] != "review") & (ev["date"] >= dates[-60]), "key"])

    hits = {"stage2": [], "pullback": [], "htf": [], "breakout": [], "ep": []}
    rs_raw, perf_raw = {}, {}
    total = traded = 0
    for key, g in df.groupby("key", sort=False):
        total += 1
        if g["date"].iloc[-1] != last:      # not traded on the latest day: stale or suspended
            continue
        traded += 1
        info, res = analyse(key, g, review_keys, recent_keys)
        if info["rs_raw"] is not None:
            rs_raw[key] = info["rs_raw"]
        if info["perf"] is not None:
            perf_raw[key] = info["perf"]
        for name, row in res.items():
            hits[name].append(row)

    # relative strength rank 1-99 among liquid stocks with a full year of history
    rank = (pd.Series(rs_raw).rank(pct=True) * 98 + 1).round().astype(int)
    keep = []
    for r in hits["stage2"]:
        r["rs"] = int(rank.get(r["key"], 0))
        if r["rs"] >= CFG["stage2"]["min_rs"]:
            keep.append(r)
    hits["stage2"] = sorted(keep, key=lambda r: -r["rs"])
    # momentum rank for the breakout scan: rank 1M, 3M and 6M returns separately among liquid
    # stocks, a stock's rank is its best of the three (it only has to be a top performer once)
    mrank = (pd.DataFrame(perf_raw).T.rank(pct=True) * 98 + 1).max(axis=1).round().astype(int) \
        if perf_raw else pd.Series(dtype=int)
    keep = []
    for r in hits["breakout"]:
        r["rank"] = int(mrank.get(r["key"], 0))
        if r["rank"] >= CFG["breakout"]["min_rank"]:
            keep.append(r)
    hits["breakout"] = sorted(keep, key=lambda r: (r["state"] != "Triggered", -r["rank"]))
    hits["ep"].sort(key=lambda r: (r["days_ago"], -r["vol_ratio"]))
    hits["pullback"].sort(key=lambda r: abs(r["dist"]))
    hits["htf"].sort(key=lambda r: (r["state"] != "Triggered", -r["gain"]))

    passed = len({r["key"] for rows in hits.values() for r in rows})
    out = {"asof": last, "generated": datetime.now().isoformat(timespec="seconds"),
           "config": CFG, "stats": {"stocks": total, "traded_last_day": traded,
                                    "ranked_for_rs": len(rs_raw), "with_a_signal": passed},
           "scans": hits}
    # Export compact chart history only for stocks that appear in a scanner result.
    # This keeps the dashboard chart data small while using the exact same EOD prices.
    chart_keys = {r["key"] for rows in hits.values() for r in rows}
    chart = {}
    for key in chart_keys:
        g = df[df["key"] == key].tail(252).copy()
        if g.empty:
            continue
        cc = g["close"].to_numpy(float)
        ss = pd.Series(cc)
        e11 = ss.ewm(span=11, adjust=False).mean().to_numpy()
        e21 = ss.ewm(span=21, adjust=False).mean().to_numpy()
        s50 = ss.rolling(50).mean().to_numpy()
        chart[key] = {
            "symbol": str(g.iloc[-1]["symbol"]), "exch": str(g.iloc[-1]["exch"]),
            "dates": g["date"].astype(str).tolist(),
            "open": [num(x) for x in g["open"]],
            "high": [num(x) for x in g["high"]],
            "low": [num(x) for x in g["low"]],
            "close": [num(x) for x in g["close"]],
            "volume": [num(x, 0) for x in g["volume"]],
            "ema11": [num(x) for x in e11],
            "ema21": [num(x) for x in e21],
            "sma50": [num(x) for x in s50],
        }
    (DATA / "chart_data.json").write_text(json.dumps(chart, separators=(",", ":")))

    (DATA / "results.json").write_text(json.dumps(out))

    print(f"As of {last}.  Stocks: {total:,}   traded on last day: {traded:,}   ranked for RS: {len(rs_raw):,}")
    for name, rows in hits.items():
        print(f"\n== {name.upper()}: {len(rows)} hits")
        for r in rows[:10]:
            flag = "  [!]" if r["note"] else ""
            if name == "stage2":
                print(f"  {r['exch']}:{r['symbol']:<12} close {r['close']:>9}  RS {r['rs']:>2}  "
                      f"1M {r['ret1m']}%  3M {r['ret3m']}%  6M {r['ret6m']}%  off high {r['off_high']}%{flag}")
            else:
                tag = r.get("state") or f"{r['level']} {r['dist']:+}%"
                print(f"  {r['exch']}:{r['symbol']:<12} close {r['close']:>9}  entry {r['entry']:>9}  "
                      f"stop {r['stop']:>9}  risk {r['risk']}%  ADR {r['adr']}%  {tag}{flag}")


if __name__ == "__main__":
    main()
