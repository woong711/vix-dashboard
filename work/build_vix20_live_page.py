from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "vix20_live_data"
OUT = ROOT / "outputs"
START_DATE = "2007-01-01"
BUY_COST = 0.001
SELL_COST = 0.001
THRESHOLD = -0.15
STRATEGY_VERSION = "2.5"
STRATEGY_NAME = "VIX 단독 2.5 QLD 전용"
SYMBOLS = ["QLD", "IAU", "IEF", "VWO", "TIP", "EFA", "^VIX", "QQQ"]
NAMES = {
    "QLD": "ProShares Ultra QQQ",
    "IAU": "iShares Gold Trust",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "TIP": "iShares TIPS Bond ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "^VIX": "CBOE Volatility Index",
    "QQQ": "Invesco QQQ Trust",
}
KEYS = {
    "QLD": "QLD",
    "IAU": "GOLD",
    "IEF": "TREASURY",
    "VWO": "EM",
    "TIP": "TIPS",
    "EFA": "DEV_EXUS",
    "^VIX": "VIX",
    "QQQ": "QQQ",
}


def yahoo_url(symbol: str) -> str:
    start = int(datetime(1998, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime.now(tz=timezone.utc).timestamp()) + 3 * 86400
    return (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
        f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
    )


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / f"{symbol.replace('^', '_')}.csv"
    if os.environ.get("VIX20_USE_CACHE") == "1" and cache.exists():
        return pd.read_csv(cache)
    req = Request(yahoo_url(symbol), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=45) as response:
        payload = json.loads(response.read())
    result = payload["chart"]["result"][0]
    quote_data = result["indicators"]["quote"][0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote_data["close"])
    rows = []
    for i, ts in enumerate(result["timestamp"]):
        close = quote_data["close"][i]
        open_ = quote_data["open"][i]
        high = quote_data["high"][i]
        low = quote_data["low"][i]
        adj = adjclose[i] if adjclose else close
        if close is None or open_ is None or high is None or low is None or adj is None:
            continue
        factor = adj / close if close else 1.0
        rows.append(
            {
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "adj_open": float(open_) * factor,
                "adj_high": float(high) * factor,
                "adj_low": float(low) * factor,
                "adj_close": float(adj),
            }
        )
    df = pd.DataFrame(rows).dropna().sort_values("date").drop_duplicates("date")
    df.to_csv(cache, index=False)
    return df


def prepare(df: pd.DataFrame, include_hilo: bool = False) -> dict:
    cols = ["date", "adj_open", "adj_close"] + (["adj_high", "adj_low"] if include_hilo else [])
    data = df[cols].dropna().sort_values("date").reset_index(drop=True)
    data["ts"] = pd.to_datetime(data["date"])
    rows = []
    for r in data.itertuples(index=False):
        row = {"date": str(r.date), "ts": r.ts, "open": float(r.adj_open), "close": float(r.adj_close)}
        if include_hilo:
            row["high"] = float(r.adj_high)
            row["low"] = float(r.adj_low)
        rows.append(row)
    by_date = {r["date"]: r for r in rows}
    last = {}
    for row in rows:
        last[row["date"][:7]] = row
    monthly = [{"month": m, "date": r["date"], "ts": r["ts"], "close": r["close"]} for m, r in sorted(last.items())]
    return {"rows": rows, "by_date": by_date, "dates": [r["date"] for r in rows], "monthly": monthly, "monthly_index": {r["month"]: i for i, r in enumerate(monthly)}}


def row_on_or_before(series: dict, date: str) -> dict | None:
    dates = series["dates"]
    if not dates or date < dates[0]:
        return None
    lo, hi = 0, len(dates) - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] <= date:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return series["rows"][best]


def month_range(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    y, m = sy, sm
    while y < ey or (y == ey and m <= em):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def monthly_momentum(series: dict, month: str, lookback: int) -> float | None:
    idx = series["monthly_index"].get(month)
    if idx is None or idx < lookback:
        return None
    return series["monthly"][idx]["close"] / series["monthly"][idx - lookback]["close"] - 1.0


def date_momentum(series: dict, signal_ts: pd.Timestamp, months: int) -> float | None:
    now = row_on_or_before(series, signal_ts.strftime("%Y-%m-%d"))
    past = row_on_or_before(series, (signal_ts - pd.DateOffset(months=months)).strftime("%Y-%m-%d"))
    if not now or not past:
        return None
    return now["close"] / past["close"] - 1.0


def vix_premium_to_ma20(series: dict, date: str) -> float | None:
    rows = series["VIX"]["rows"]
    idx = {row["date"]: i for i, row in enumerate(rows)}.get(date)
    if idx is None:
        row = row_on_or_before(series["VIX"], date)
        if not row:
            return None
        idx = {r["date"]: i for i, r in enumerate(rows)}.get(row["date"])
    if idx is None or idx < 19:
        return None
    ma20 = sum(r["close"] for r in rows[idx - 19 : idx + 1]) / 20
    return rows[idx]["close"] / ma20 - 1.0 if ma20 > 0 else None


def canary_info(series: dict, month: str) -> tuple[int, bool, list[dict]]:
    details = []
    for sym in ["EM", "TIPS", "DEV_EXUS"]:
        mom = monthly_momentum(series[sym], month, 6)
        risk = False if mom is None else mom < 0
        details.append({"asset": sym, "momentum": mom, "risk": risk})
    return sum(1 for d in details if d["risk"]), bool(details) and all(d["risk"] for d in details), details


def best_defensive(series: dict, signal_ts: pd.Timestamp) -> tuple[str, list[dict]]:
    scores = []
    for sym in ["GOLD", "TREASURY"]:
        mom = date_momentum(series[sym], signal_ts, 9)
        if mom is not None and math.isfinite(mom):
            scores.append({"asset": sym, "momentum": mom})
    scores.sort(key=lambda x: x["momentum"], reverse=True)
    return (scores[0]["asset"] if scores else "TREASURY"), scores


def next_trade_date(series: dict, after_date: str, target: str, previous: str | None, end_date: str) -> str | None:
    previous_dates = series[previous]["by_date"] if previous else {}
    for row in series["QLD"]["rows"]:
        date = row["date"]
        if date <= after_date or date > end_date:
            continue
        if date not in series[target]["by_date"]:
            continue
        if previous and date not in previous_dates:
            continue
        return date
    return None


def next_base_date(base_dates: list[str], after_date: str, end_date: str) -> str | None:
    for date in base_dates:
        if after_date < date <= end_date:
            return date
    return None


def weights_equal(a: dict[str, float] | None, b: dict[str, float]) -> bool:
    if a is None:
        return False
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) < 1e-10 for k in set(a) | set(b))


def run_vix11(series: dict, end_date: str, signal_end_month: str) -> dict:
    holdings = {}
    current_weights = None
    current_target = None
    vix_risk_state = False
    signals, trades, total_cost = [], [], 0.0
    first_month = series["QLD"]["monthly"][0]["month"]
    for month in month_range(first_month, signal_end_month):
        idx = series["QLD"]["monthly_index"].get(month)
        if idx is None:
            continue
        signal = series["QLD"]["monthly"][idx]
        signal_date = signal["date"]
        if signal_date > end_date:
            continue
        risk_count, all_canary_risk, canary_details = canary_info(series, month)
        vix_row = row_on_or_before(series["VIX"], signal_date)
        vix_close = vix_row["close"] if vix_row else None
        vix_premium = vix_premium_to_ma20(series, signal_date)
        if vix_premium is not None and vix_premium >= 0.30:
            vix_risk_state = True
        elif vix_premium is not None and vix_premium < 0.10:
            vix_risk_state = False
        risk = all_canary_risk or vix_risk_state
        defensive_target, defensive_scores = best_defensive(series, signal["ts"])
        target = defensive_target if risk else "QLD"
        trade_date = next_trade_date(series, signal_date, target, current_target, end_date)
        changed = False
        if trade_date:
            target_weights = {target: 1.0}
            if not weights_equal(current_weights, target_weights):
                before = 1.0 if not holdings else sum(shares * series[sym]["by_date"][trade_date]["open"] for sym, shares in holdings.items())
                symbols = set(holdings) | set(target_weights)
                old_values = {sym: holdings.get(sym, 0.0) * series[sym]["by_date"][trade_date]["open"] for sym in symbols}
                target_values = {sym: before * target_weights.get(sym, 0.0) for sym in symbols}
                cost = sum(max(old_values[sym] - target_values[sym], 0.0) for sym in symbols) * SELL_COST
                cost += sum(max(target_values[sym] - old_values[sym], 0.0) for sym in symbols) * BUY_COST
                after = before - cost
                total_cost += cost
                holdings = {target: after / series[target]["by_date"][trade_date]["open"]}
                current_weights = target_weights
                current_target = target
                changed = True
                trades.append({"month": month, "signalDate": signal_date, "tradeDate": trade_date, "target": target, "risk": risk, "canaryRiskCount": risk_count, "allCanaryRisk": all_canary_risk, "vixClose": vix_close, "vixPremiumToMa20": vix_premium, "navAfterCost": after})
        signals.append({"month": month, "signalDate": signal_date, "tradeDate": trade_date, "target": target, "changed": changed, "risk": risk, "vixRiskState": vix_risk_state, "allCanaryRisk": all_canary_risk, "canaryRiskCount": risk_count, "vixClose": vix_close, "vixPremiumToMa20": vix_premium, "canaries": canary_details, "defensiveScores": defensive_scores})
    return {"trades": trades, "signals": signals, "totalCost": total_cost}


def build_vix_component(series: dict, vix_result: dict, end_date: str) -> dict:
    rows = []
    trades = vix_result["trades"]
    base_rows = [r for r in series["QLD"]["rows"] if START_DATE <= r["date"] <= end_date]
    for i, trade in enumerate(trades):
        target = trade["target"]
        next_trade = trades[i + 1]["tradeDate"] if i + 1 < len(trades) else "9999-12-31"
        shares = trade["navAfterCost"] / series[target]["by_date"][trade["tradeDate"]]["open"]
        for base_row in base_rows:
            date = base_row["date"]
            if not (trade["tradeDate"] <= date < next_trade):
                continue
            price_row = series[target]["by_date"].get(date) or row_on_or_before(series[target], date)
            if not price_row:
                continue
            open_price = price_row.get("open", price_row["close"])
            rows.append({"date": date, "open": shares * open_price, "close": shares * price_row["close"], "holding": target})
    rows.sort(key=lambda r: r["date"])
    base_open = rows[0]["open"]
    for row in rows:
        row["open"] /= base_open
        row["close"] /= base_open
    return {"rows": rows, "by_date": {r["date"]: r for r in rows}, "dates": [r["date"] for r in rows]}


def build_asset_component(series: dict, symbol: str, dates: list[str]) -> dict:
    rows = []
    for date in dates:
        price_row = series[symbol]["by_date"].get(date) or row_on_or_before(series[symbol], date)
        if not price_row:
            continue
        rows.append({"date": date, "open": price_row.get("open", price_row["close"]), "close": price_row["close"]})
    base_open = rows[0]["open"]
    for row in rows:
        row["open"] /= base_open
        row["close"] /= base_open
    return {"rows": rows, "by_date": {r["date"]: r for r in rows}, "dates": [r["date"] for r in rows]}


def weekly_filter_signals(series: dict, base_dates: list[str], end_date: str) -> tuple[dict, list[dict]]:
    qqq = pd.DataFrame(series["QQQ"]["rows"])
    qqq["ts"] = pd.to_datetime(qqq["date"])
    qqq = qqq[(qqq["date"] >= START_DATE) & (qqq["date"] <= end_date)].sort_values("ts")
    weekly = (
        qqq.set_index("ts")
        .resample("W-FRI")
        .agg(date=("date", "last"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index(drop=True)
    )
    weekly["ma10"] = weekly["close"].rolling(10).mean()
    signals = {}
    states = []
    current_weight = 1.0
    for idx, row in enumerate(weekly.itertuples(index=False)):
        if idx < 19 or not math.isfinite(row.ma10):
            continue
        window = weekly.iloc[idx - 19 : idx + 1].reset_index(drop=True)
        peak_idx = int(window["high"].idxmax())
        peak_high = float(window.loc[peak_idx, "high"])
        trough_low = float(window.loc[peak_idx:, "low"].min())
        ordered_drawdown = trough_low / peak_high - 1.0
        below_ma = float(row.close) < float(row.ma10)
        entry = ordered_drawdown <= THRESHOLD and below_ma
        exit_signal = float(row.close) > float(row.ma10)
        target_weight = current_weight
        action = "hold"
        if current_weight >= 1.0 and entry:
            target_weight = 0.5
            action = "risk_on"
        elif current_weight < 1.0 and exit_signal:
            target_weight = 1.0
            action = "risk_off"
        state = {"signalDate": row.date, "qqqWeeklyClose": float(row.close), "qqqWeeklyMa10": float(row.ma10), "ordered20wDrawdown": ordered_drawdown, "belowMa10": below_ma, "entryCondition": entry, "exitCondition": exit_signal, "vixWeight": target_weight, "goldWeight": 1.0 - target_weight, "action": action}
        states.append(state)
        if abs(target_weight - current_weight) > 1e-12:
            trade_date = next_base_date(base_dates, row.date, end_date)
            if trade_date:
                signals[trade_date] = dict(state)
                signals[trade_date]["tradeDate"] = trade_date
                current_weight = target_weight
    return signals, states


def filter_weights(base_target: str, weekly_risk: bool) -> tuple[float, float]:
    applied = base_target == "QLD" and weekly_risk
    return (0.5, 0.5) if applied else (1.0, 0.0)


def qld_only_filter_signals(vix_component: dict, weekly_signals: dict, monthly_trades: list[dict]) -> tuple[dict, dict]:
    schedules, states = {}, {}
    weekly_risk = False
    weekly_signal_date = ""
    monthly_signal_date = ""
    monthly_idx = 0
    current_weight = 1.0
    for date in vix_component["dates"]:
        while monthly_idx < len(monthly_trades) and monthly_trades[monthly_idx]["tradeDate"] <= date:
            monthly_signal_date = monthly_trades[monthly_idx]["signalDate"]
            monthly_idx += 1
        if date in weekly_signals:
            weekly_risk = weekly_signals[date]["vixWeight"] < 1.0
            weekly_signal_date = weekly_signals[date]["signalDate"]
        base_target = vix_component["by_date"][date]["holding"]
        vix_weight, gold_weight = filter_weights(base_target, weekly_risk)
        states[date] = {"weeklyRiskState": weekly_risk, "weeklyFilterApplied": vix_weight < 1.0}
        if vix_weight != current_weight:
            if vix_weight < 1.0:
                reason = "weekly_defense_on" if date in weekly_signals else "monthly_qld_reentry"
            else:
                reason = "monthly_defense" if base_target != "QLD" else "weekly_defense_off"
            schedules[date] = {
                **weekly_signals.get(date, {}),
                "signalDate": max(monthly_signal_date, weekly_signal_date),
                "tradeDate": date,
                "monthlySignalDate": monthly_signal_date,
                "weeklySignalDate": weekly_signal_date,
                "baseTarget": base_target,
                "weeklyRiskState": weekly_risk,
                "reason": reason,
                "vixWeight": vix_weight,
                "goldWeight": gold_weight,
            }
            current_weight = vix_weight
    return schedules, states


def run_overlay(vix_component: dict, gold_component: dict, schedules: dict, internal_trade_count: int) -> dict:
    dates = [d for d in vix_component["dates"] if d in gold_component["by_date"]]
    weights = {"vix": 1.0, "gold": 0.0}
    first = dates[0]
    before = 1.0
    after = before - BUY_COST
    units = {"vix": after / vix_component["by_date"][first]["open"], "gold": 0.0}
    total_cost = BUY_COST
    trades = [{"tradeDate": first, "type": "initial", "vixWeight": 1.0, "goldWeight": 0.0, "navBeforeCost": before, "navAfterCost": after, "cost": BUY_COST}]
    daily = []
    for date in dates:
        if date in schedules:
            sig = schedules[date]
            target_weights = {"vix": sig["vixWeight"], "gold": sig["goldWeight"]}
            open_prices = {"vix": vix_component["by_date"][date]["open"], "gold": gold_component["by_date"][date]["open"]}
            nav_before = sum(units[k] * open_prices[k] for k in units)
            current_values = {k: units[k] * open_prices[k] for k in units}
            target_values = {k: nav_before * target_weights[k] for k in units}
            cost = sum(max(current_values[k] - target_values[k], 0.0) for k in units) * SELL_COST
            cost += sum(max(target_values[k] - current_values[k], 0.0) for k in units) * BUY_COST
            nav_after = nav_before - cost
            if cost > 1e-12:
                total_cost += cost
                weights = target_weights
                units = {k: nav_after * weights[k] / open_prices[k] for k in units}
                trades.append({"tradeDate": date, "type": "weekly_filter", "vixWeight": weights["vix"], "goldWeight": weights["gold"], "navBeforeCost": nav_before, "navAfterCost": nav_after, "cost": cost, **sig})
        close_prices = {"vix": vix_component["by_date"][date]["close"], "gold": gold_component["by_date"][date]["close"]}
        nav = sum(units[k] * close_prices[k] for k in units)
        daily.append({"date": date, "nav": nav, "vixWeight": weights["vix"], "goldWeight": weights["gold"], "vixHolding": vix_component["by_date"][date]["holding"], "holding": combined_holding(vix_component["by_date"][date]["holding"], weights["vix"], weights["gold"])})
    summary, monthly = summarize(daily, trades, internal_trade_count, total_cost)
    return {"summary": summary, "daily": daily, "monthly": monthly, "overlayTrades": trades, "totalCost": total_cost}


def combined_holding(vix_holding: str, vix_weight: float, gold_weight: float) -> str:
    parts = {}
    if vix_holding == "QLD":
        parts["QLD"] = parts.get("QLD", 0) + vix_weight
    elif vix_holding == "GOLD":
        parts["IAU"] = parts.get("IAU", 0) + vix_weight
    elif vix_holding == "TREASURY":
        parts["IEF"] = parts.get("IEF", 0) + vix_weight
    if gold_weight:
        parts["IAU"] = parts.get("IAU", 0) + gold_weight
    return " / ".join(f"{k} {v:.0%}" for k, v in parts.items() if v > 1e-9)


def summarize(daily: list[dict], overlay_trades: list[dict], internal_trade_count: int, total_cost: float) -> tuple[dict, list[dict]]:
    peak = -float("inf")
    prev = None
    for row in daily:
        row["dailyReturn"] = row["nav"] / prev - 1.0 if prev else None
        peak = max(peak, row["nav"])
        row["drawdown"] = row["nav"] / peak - 1.0
        prev = row["nav"]
    month_end = {}
    for row in daily:
        month_end[row["date"][:7]] = row
    monthly, yearly, last_nav, month_peak = [], {}, 1.0, -float("inf")
    for month, row in sorted(month_end.items()):
        ret = row["nav"] / last_nav - 1.0
        last_nav = row["nav"]
        month_peak = max(month_peak, row["nav"])
        monthly.append({"month": month, "date": row["date"], "nav": row["nav"], "return": ret, "drawdown": row["nav"] / month_peak - 1.0, "holding": row["holding"], "vixHolding": row["vixHolding"], "vixWeight": row["vixWeight"], "goldWeight": row["goldWeight"]})
        yearly[month[:4]] = yearly.get(month[:4], 1.0) * (1.0 + ret)
    returns = [r["dailyReturn"] for r in daily if r["dailyReturn"] is not None and math.isfinite(r["dailyReturn"])]
    avg = sum(returns) / len(returns)
    std = math.sqrt(sum((x - avg) ** 2 for x in returns) / (len(returns) - 1))
    years = (pd.Timestamp(daily[-1]["date"]) - pd.Timestamp(daily[0]["date"])).days / 365.25
    final = daily[-1]["nav"]
    cagr = final ** (1 / years) - 1.0
    vol = std * math.sqrt(252)
    mdd = min(daily, key=lambda r: r["drawdown"])
    return {
        "firstDate": daily[0]["date"],
        "lastDate": daily[-1]["date"],
        "finalNav": final,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": cagr / vol if vol else None,
        "maxDrawdown": mdd["drawdown"],
        "maxDrawdownDate": mdd["date"],
        "tradeEvents": len(overlay_trades) + internal_trade_count,
        "overlayRebalances": len(overlay_trades),
        "vixStrategyTrades": internal_trade_count,
        "totalCost": total_cost,
        "yearly": [{"year": y, "return": v - 1.0} for y, v in sorted(yearly.items())],
    }, monthly


def build_payload(raw: dict | None = None, as_of: pd.Timestamp | None = None) -> dict:
    if raw is None:
        raw = {symbol: fetch_yahoo(symbol) for symbol in SYMBOLS}
    series = {KEYS[symbol]: prepare(df, include_hilo=(symbol == "QQQ")) for symbol, df in raw.items()}
    end_date = min(series[k]["dates"][-1] for k in ["QLD", "GOLD", "TREASURY", "EM", "TIPS", "DEV_EXUS", "VIX", "QQQ"])
    today = pd.Timestamp.now(tz="Asia/Seoul") if as_of is None else as_of
    end_ts = pd.Timestamp(end_date)
    signal_end_month = (end_ts.replace(day=1) - pd.DateOffset(days=1)).strftime("%Y-%m") if end_ts.strftime("%Y-%m") == today.strftime("%Y-%m") else end_ts.strftime("%Y-%m")
    vix_result = run_vix11(series, end_date, signal_end_month)
    vix_component = build_vix_component(series, vix_result, end_date)
    gold_component = build_asset_component(series, "GOLD", vix_component["dates"])
    weekly_signals, weekly_states = weekly_filter_signals(series, vix_component["dates"], end_date)
    schedules, applied_states = qld_only_filter_signals(vix_component, weekly_signals, vix_result["trades"])
    overlay = run_overlay(vix_component, gold_component, schedules, len([t for t in vix_result["trades"] if START_DATE <= t["tradeDate"] <= end_date]))
    for row in overlay["daily"]:
        row.update(applied_states[row["date"]])
    latest_vix_signal = vix_result["signals"][-1]
    latest_weekly = weekly_states[-1] if weekly_states else None
    current_daily = overlay["daily"][-1]
    next_vix_holding = latest_vix_signal["target"]
    next_weekly_risk = latest_weekly["vixWeight"] < 1.0 if latest_weekly else current_daily["weeklyRiskState"]
    next_vix_weight, next_gold_weight = filter_weights(next_vix_holding, next_weekly_risk)
    next_signal_date = max(latest_vix_signal["signalDate"], latest_weekly["signalDate"] if latest_weekly else "")
    next_holding = combined_holding(next_vix_holding, next_vix_weight, next_gold_weight)
    coverage = [{"symbol": s, "name": NAMES[s], "first": str(raw[s]["date"].iloc[0]), "last": str(raw[s]["date"].iloc[-1]), "rows": int(len(raw[s]))} for s in SYMBOLS]
    return {
        "strategyVersion": STRATEGY_VERSION,
        "strategyName": STRATEGY_NAME,
        "generatedAt": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S KST"),
        "dataThrough": end_date,
        "signalThroughMonth": signal_end_month,
        "startDate": overlay["daily"][0]["date"],
        "costs": {"buy": BUY_COST, "sell": SELL_COST},
        "threshold": THRESHOLD,
        "summary": overlay["summary"],
        "daily": overlay["daily"],
        "monthly": overlay["monthly"],
        "yearly": overlay["summary"]["yearly"],
        "vixTrades": [t for t in vix_result["trades"] if START_DATE <= t["tradeDate"] <= end_date],
        "overlayTrades": overlay["overlayTrades"],
        "current": {
            "holding": current_daily["holding"],
            "vixWeight": current_daily["vixWeight"],
            "goldWeight": current_daily["goldWeight"],
            "vixHolding": current_daily["vixHolding"],
            "nextHolding": next_holding,
            "nextVixHolding": next_vix_holding,
            "nextVixWeight": next_vix_weight,
            "nextGoldWeight": next_gold_weight,
            "nextSignalDate": next_signal_date,
            "nextWeeklyRiskState": next_weekly_risk,
            "nextFilterApplied": next_vix_weight < 1.0,
            "weeklyRiskState": current_daily["weeklyRiskState"],
            "weeklyFilterApplied": current_daily["weeklyFilterApplied"],
            "nextNeedsAdjustment": next_holding != current_daily["holding"],
            "vixSignal": latest_vix_signal,
            "weeklyFilter": latest_weekly,
        },
        "coverage": coverage,
    }


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VIX 단독 2.5 QLD 전용</title>
<style>
:root{--bg:#f5f7fb;--panel:#fff;--ink:#142033;--muted:#66758c;--line:#dce3ee;--blue:#3154d4;--green:#12805c;--red:#c84646;--amber:#a46314;--soft:#eef2f7;--shadow:0 10px 24px rgba(20,32,51,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;line-height:1.45}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:20px 24px 14px}h1{margin:0;font-size:24px}.meta{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:8px;color:var(--muted);font-size:13px}
.shell{max-width:1580px;margin:0 auto;padding:18px 24px 36px;display:grid;grid-template-columns:360px minmax(0,1fr);gap:18px}.shell>main{min-width:0}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);overflow:hidden}.head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 16px;border-bottom:1px solid var(--soft)}.head h2{margin:0;font-size:15px}.body{padding:16px;min-width:0}.stack{display:grid;gap:13px}
.label{display:block;color:var(--muted);font-size:12px;font-weight:800;margin-bottom:5px}.big{font-size:28px;font-weight:950;line-height:1}.muted{color:var(--muted)}.pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 8px;font-size:12px;font-weight:900}.pill.ok{background:#e8f6ef;color:var(--green)}.pill.risk{background:#fdecec;color:var(--red)}.pill.neutral{background:#eef2ff;color:var(--blue)}
button{border:0;border-radius:6px;font-size:13px;font-weight:850;padding:9px 12px;cursor:pointer}.primary{background:var(--green);color:#fff;width:100%}.note{font-size:12px;color:var(--muted)}.status{font-size:12px;color:var(--muted);min-height:34px}.hold-row{display:grid;grid-template-columns:64px 1fr 44px;gap:8px;align-items:center;font-size:13px}.bar{height:8px;border-radius:999px;background:#e8edf5;overflow:hidden}.bar span{display:block;height:100%;background:var(--blue)}.coverage{display:grid;gap:7px}.coverage-row{display:grid;grid-template-columns:48px 1fr 82px;gap:8px;font-size:12px;color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:12px}.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;box-shadow:var(--shadow);min-height:84px}.kpi .name{color:var(--muted);font-size:12px;font-weight:850}.kpi .value{font-size:22px;font-weight:950;margin-top:8px}.kpi .hint{font-size:12px;color:var(--muted);margin-top:7px}
.grid2{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(360px,.75fr);gap:14px;margin-top:14px}.chart{width:100%;height:320px;display:block}.short{height:230px}.toolbar{display:flex;gap:8px;align-items:center}.mini-select{border:1px solid var(--line);border-radius:6px;padding:6px 8px;background:#fff}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 9px;border-bottom:1px solid var(--soft);text-align:right;white-space:nowrap}th{color:var(--muted);font-size:12px;font-weight:850;background:#fafbfd}td.text,th.text{text-align:left}.wrap{overflow:auto;max-height:360px}.seg{display:flex;gap:6px;flex-wrap:wrap}.seg button{background:#edf1f7;color:var(--muted)}.seg button.active{background:var(--blue);color:#fff}
.heat{display:grid;gap:3px;overflow-x:auto}.heat-row{display:grid;grid-template-columns:52px repeat(12,minmax(54px,1fr));gap:3px}.heat-cell{border-radius:4px;min-height:28px;display:flex;align-items:center;justify-content:center;font-weight:850;font-size:12px}.heat-label{color:var(--muted);font-size:12px;display:flex;align-items:center}
.tooltip{position:fixed;z-index:20;display:none;pointer-events:none;background:#111827;color:#fff;border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.45;box-shadow:0 10px 24px rgba(0,0,0,.22)}.tooltip b{display:block;margin-bottom:4px}
@media(max-width:1100px){.shell{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}@media(max-width:720px){.shell{padding:14px}.kpis{grid-template-columns:1fr}.heat-row{grid-template-columns:44px repeat(12,48px)}}
</style>
</head>
<body>
<header>
  <h1>VIX 단독 2.5 QLD 전용</h1>
  <div class="meta"><span id="generated"></span><span id="through"></span><span>월간 VIX 1.1 · QLD 보유 시 주간 방어</span></div>
</header>
<div class="shell">
  <aside class="panel">
    <div class="head"><h2>현재 상태</h2></div>
    <div class="body stack">
      <div><span class="label">현재 보유 비중</span><div id="currentHoldings"></div></div>
      <div><span class="label" id="nextHoldingLabel">다음 거래일 목표 보유</span><div id="nextHoldings"></div></div>
      <div><span class="label">주간 QQQ 필터</span><div id="weeklyFilter"></div></div>
      <div><span class="label">VIX 단독 1.1 내부 신호</span><div id="vixSignal"></div></div>
      <div>
        <button id="updateBtn" class="primary">최신 데이터 업데이트</button>
        <div id="updateStatus" class="status">로컬 서버에서 열면 버튼으로 최신 데이터를 받을 수 있습니다.</div>
      </div>
      <div><span class="label">데이터 범위</span><div id="coverage" class="coverage"></div></div>
    </div>
  </aside>
  <main>
    <section class="kpis" id="kpis"></section>
    <div class="grid2">
      <section class="panel"><div class="head"><h2>NAV</h2><div class="toolbar"><select id="yearSelect" class="mini-select"></select></div></div><div class="body"><canvas id="navChart" class="chart"></canvas></div></section>
      <section class="panel"><div class="head"><h2>Drawdown</h2><span class="muted">일별 기준</span></div><div class="body"><canvas id="ddChart" class="chart short"></canvas></div></section>
    </div>
    <div class="grid2">
      <section class="panel"><div class="head"><h2>거래내역</h2><div class="seg"><button id="showOverlay" class="active">2.5 비중조정</button><button id="showVix">VIX 1.1</button></div></div><div class="body"><div class="wrap"><table id="tradeTable"></table></div></div></section>
      <section class="panel"><div class="head"><h2>연도별 수익률</h2></div><div class="body"><div class="wrap"><table id="yearTable"></table></div></div></section>
    </div>
    <section class="panel" style="margin-top:14px"><div class="head"><h2>월별 수익률</h2></div><div class="body"><div id="heatmap" class="heat"></div></div></section>
    <section class="panel" style="margin-top:14px"><div class="head"><h2>월별 보유 자산</h2></div><div class="body"><div class="wrap"><table id="holdingTable"></table></div></div></section>
  </main>
</div>
<div id="tooltip" class="tooltip"></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const $ = id => document.getElementById(id);
const fmtPct=(v,d=2)=>Number.isFinite(v)?(v*100).toFixed(d)+'%':'-';
const fmtNum=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):'-';
const fmtDateKr=d=>{if(!d)return'-'; const [y,m,day]=d.split('-'); return `${Number(y)}년 ${Number(m)}월 ${Number(day)}일`;};
const assetLabel=s=>({GOLD:'IAU',TREASURY:'IEF',QLD:'QLD'}[s]||s||'-');
const reasonLabel=s=>({weekly_defense_on:'주간 방어',weekly_defense_off:'주간 복귀',monthly_defense:'월말 방어자산 전환',monthly_qld_reentry:'QLD 복귀 · 주간 방어 유지'}[s]||'초기 편입');
let tradeMode='overlay', year='all';
function render(){
  $('generated').textContent='생성: '+DATA.generatedAt;
  $('through').textContent='데이터 기준일: '+DATA.dataThrough;
  renderCurrent(); renderKpis(); renderYearOptions(); renderCharts(); renderTrades(); renderYears(); renderHeatmap(); renderHoldings();
  $('coverage').innerHTML=DATA.coverage.map(r=>`<div class="coverage-row"><b>${r.symbol}</b><span>${r.name}</span><span>${r.last}</span></div>`).join('');
}
function holdingRows(text){
  return (text || '-').split(' / ').filter(Boolean).map(x=>{const [a,b]=x.split(' '); const w=parseFloat(b)/100||0; return `<div class="hold-row"><b>${a}</b><div class="bar"><span style="width:${w*100}%"></span></div><span>${b||''}</span></div>`}).join('');
}
function renderCurrent(){
  $('currentHoldings').innerHTML = holdingRows(DATA.current.holding);
  $('nextHoldingLabel').textContent = `다음 거래일 목표 보유 (신호날짜: ${fmtDateKr(DATA.current.nextSignalDate)})`;
  const need=DATA.current.nextNeedsAdjustment;
  $('nextHoldings').innerHTML = `${holdingRows(DATA.current.nextHolding)}<div class="note"><span class="pill ${need?'risk':'ok'}">${need?'조정 필요':'변경 없음'}</span> 월말 목표 ${assetLabel(DATA.current.nextVixHolding)} · 추가 IAU ${fmtPct(DATA.current.nextGoldWeight,0)}</div>`;
  const f=DATA.current.weeklyFilter, risk=DATA.current.nextWeeklyRiskState, applied=DATA.current.nextFilterApplied;
  $('weeklyFilter').innerHTML=`<span class="pill ${risk?'risk':'ok'}">${risk?'QQQ 위험 ON':'QQQ 위험 OFF'}</span> <span class="pill ${applied?'risk':'neutral'}">${applied?'다음 목표에 50% 방어 적용':'다음 목표에 추가 방어 없음'}</span><div class="note">신호일 ${f.signalDate}, QQQ 종가 ${fmtNum(f.qqqWeeklyClose,2)}, 10주선 ${fmtNum(f.qqqWeeklyMa10,2)}</div><div class="note">20주 고점 이후 낙폭 ${fmtPct(f.ordered20wDrawdown)} / 기준 ${fmtPct(DATA.threshold,0)}</div>`;
  const v=DATA.current.vixSignal;
  $('vixSignal').innerHTML=`<span class="pill ${v.risk?'risk':'ok'}">${v.risk?'위험':'정상'}</span><div class="note">월간 신호 ${v.signalDate}, 목표 ${assetLabel(v.target)}</div><div class="note">VIX 20일 평균 대비 ${fmtPct(v.vixPremiumToMa20)} / 카나리아 위험 ${v.canaryRiskCount}개</div>`;
}
function renderKpis(){
  const s=DATA.summary;
  const rows=[['최종 NAV',fmtNum(s.finalNav,2)+'배','초기 1 기준'],['CAGR',fmtPct(s.cagr),'연복리'],['MDD',fmtPct(s.maxDrawdown),s.maxDrawdownDate],['변동성',fmtPct(s.volatility),'연율화'],['Sharpe',fmtNum(s.sharpe,2),'CAGR/변동성'],['거래횟수',String(s.tradeEvents),'월말 자산변경 + 2.5 비중조정']];
  $('kpis').innerHTML=rows.map(r=>`<div class="kpi"><div class="name">${r[0]}</div><div class="value">${r[1]}</div><div class="hint">${r[2]}</div></div>`).join('');
}
function renderYearOptions(){const years=DATA.yearly.map(r=>r.year); $('yearSelect').innerHTML=['all',...years].map(y=>`<option value="${y}">${y==='all'?'전체':y}</option>`).join(''); $('yearSelect').value=year;}
function drawLine(canvas, rows, field, color, maxOverride=null){
  const ctx=canvas.getContext('2d'), dpr=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect(); canvas.width=Math.floor(rect.width*dpr); canvas.height=Math.floor(rect.height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0);
  const w=rect.width,h=rect.height,p={l:48,r:14,t:16,b:28}; ctx.clearRect(0,0,w,h);
  const vals=rows.map(r=>r[field]); let min=Math.min(...vals), max=maxOverride??Math.max(...vals); if(min===max){min*=.95;max*=1.05}
  const x=i=>p.l+i*(w-p.l-p.r)/(rows.length-1), y=v=>p.t+(max-v)*(h-p.t-p.b)/(max-min);
  ctx.strokeStyle='#dce3ee'; ctx.fillStyle='#66758c'; ctx.font='12px Inter,sans-serif';
  for(let g=0;g<=4;g++){const yy=p.t+g*(h-p.t-p.b)/4; ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(w-p.r,yy);ctx.stroke(); const val=max-g*(max-min)/4; ctx.fillText(field==='drawdown'?fmtPct(val,0):fmtNum(val,1),6,yy+4)}
  ctx.beginPath(); rows.forEach((r,i)=>{if(i===0)ctx.moveTo(x(i),y(r[field])); else ctx.lineTo(x(i),y(r[field]));}); ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke(); canvas._chart={rows,field,p};
}
function renderCharts(){const rows=year==='all'?DATA.daily:DATA.daily.filter(r=>r.date.slice(0,4)===year); drawLine($('navChart'),rows.length>1?rows:DATA.daily,'nav','#3154d4'); drawLine($('ddChart'),DATA.daily,'drawdown','#c84646',0)}
function renderTrades(){
  if(tradeMode==='overlay') $('tradeTable').innerHTML='<thead><tr><th class="text">거래일</th><th>신호일</th><th>기본전략</th><th>추가 IAU</th><th class="text">변경 이유</th><th>비용</th></tr></thead><tbody>'+DATA.overlayTrades.map(t=>`<tr><td class="text">${t.tradeDate}</td><td>${t.signalDate||'-'}</td><td>${fmtPct(t.vixWeight,0)}</td><td>${fmtPct(t.goldWeight,0)}</td><td class="text">${reasonLabel(t.reason)}</td><td>${fmtNum(t.cost,4)}</td></tr>`).join('')+'</tbody>';
  else $('tradeTable').innerHTML='<thead><tr><th class="text">거래일</th><th>신호일</th><th>목표</th><th>위험</th><th>VIX</th><th>카나리아</th></tr></thead><tbody>'+DATA.vixTrades.map(t=>`<tr><td class="text">${t.tradeDate}</td><td>${t.signalDate}</td><td>${t.target}</td><td><span class="pill ${t.risk?'risk':'ok'}">${t.risk?'ON':'OFF'}</span></td><td>${fmtPct(t.vixPremiumToMa20)}</td><td>${t.canaryRiskCount}</td></tr>`).join('')+'</tbody>';
}
function renderYears(){ $('yearTable').innerHTML='<thead><tr><th class="text">연도</th><th>수익률</th></tr></thead><tbody>'+DATA.yearly.map(r=>`<tr><td class="text">${r.year}</td><td style="color:${r.return>=0?'var(--green)':'var(--red)'}">${fmtPct(r.return)}</td></tr>`).join('')+'</tbody>'; }
function heatColor(ret){const c=Math.max(-.18,Math.min(.18,ret||0)); if(c>=0){const a=Math.min(1,c/.1); return `rgba(${Math.round(230-145*a)},${Math.round(241-65*a)},${Math.round(234-126*a)},1)`} const a=Math.min(1,Math.abs(c)/.1); return `rgba(${Math.round(250-45*a)},${Math.round(232-132*a)},${Math.round(232-132*a)},1)`}
function renderHeatmap(){const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], years=[...new Set(DATA.monthly.map(r=>r.month.slice(0,4)))], by=Object.fromEntries(DATA.monthly.map(r=>[r.month,r])); let html=`<div class="heat-row"><div></div>${months.map(m=>`<div class="heat-label">${m}</div>`).join('')}</div>`; for(const y of years){html+=`<div class="heat-row"><div class="heat-label">${y}</div>`+months.map((m,i)=>{const key=`${y}-${String(i+1).padStart(2,'0')}`, r=by[key]; if(!r)return'<div></div>'; const fg=r.return>=.035||r.return<=-.035?'#fff':'#111827'; return `<div class="heat-cell" style="background:${heatColor(r.return)};color:${fg}" title="${key}">${(r.return*100).toFixed(1)}</div>`}).join('')+'</div>'} $('heatmap').innerHTML=html;}
function renderHoldings(){ $('holdingTable').innerHTML='<thead><tr><th class="text">월</th><th>NAV</th><th>수익률</th><th class="text">월말 보유</th><th>월말 기본목표</th><th>기본전략 / 추가 IAU</th></tr></thead><tbody>'+DATA.monthly.slice().reverse().map(r=>`<tr><td class="text">${r.month}</td><td>${fmtNum(r.nav,2)}</td><td style="color:${r.return>=0?'var(--green)':'var(--red)'}">${fmtPct(r.return)}</td><td class="text">${r.holding}</td><td>${assetLabel(r.vixHolding)}</td><td>${fmtPct(r.vixWeight,0)} / ${fmtPct(r.goldWeight,0)}</td></tr>`).join('')+'</tbody>'; }
function tooltip(e){const c=e.currentTarget._chart;if(!c)return;const rect=e.currentTarget.getBoundingClientRect(), n=c.rows.length, idx=Math.max(0,Math.min(n-1,Math.round((e.clientX-rect.left-c.p.l)*(n-1)/(rect.width-c.p.l-c.p.r)))), r=c.rows[idx], tip=$('tooltip'); tip.innerHTML=`<b>${r.date}</b>NAV ${fmtNum(r.nav,2)}<br>MDD ${fmtPct(r.drawdown)}<br>${r.holding}`; tip.style.display='block'; tip.style.left=e.clientX+14+'px'; tip.style.top=e.clientY+14+'px'}
async function updateData(){const st=$('updateStatus'); if(location.protocol==='file:'){st.textContent='업데이트 버튼은 로컬 서버에서 열었을 때 작동합니다. work/vix20_live_server.py를 실행한 뒤 http://127.0.0.1:8771/vix20_live_portfolio.html 로 열어주세요.'; st.style.color='var(--amber)'; return} $('updateBtn').disabled=true; st.textContent='최신 데이터를 받는 중입니다...'; st.style.color='var(--muted)'; try{const res=await fetch('/api/update',{method:'POST'}); const data=await res.json(); if(!res.ok||!data.ok) throw new Error(data.error||'업데이트 실패'); st.textContent=`업데이트 성공: 데이터 기준일 ${data.dataThrough}. 새로 여는 중입니다.`; st.style.color='var(--green)'; setTimeout(()=>location.href=location.pathname+'?updated='+Date.now(),700)}catch(e){st.textContent='업데이트 실패: '+e.message; st.style.color='var(--red)'; $('updateBtn').disabled=false}}
$('showOverlay').onclick=()=>{tradeMode='overlay';$('showOverlay').classList.add('active');$('showVix').classList.remove('active');renderTrades()}; $('showVix').onclick=()=>{tradeMode='vix';$('showVix').classList.add('active');$('showOverlay').classList.remove('active');renderTrades()};
$('yearSelect').onchange=e=>{year=e.target.value;renderCharts()}; $('updateBtn').onclick=updateData; for(const c of [$('navChart'),$('ddChart')]){c.addEventListener('mousemove',tooltip);c.addEventListener('mouseleave',()=>$('tooltip').style.display='none')} window.addEventListener('resize',renderCharts); render();
</script>
</body>
</html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
    output = OUT / "vix20_live_portfolio.html"
    output.write_text(html, encoding="utf-8")
    pd.DataFrame(payload["daily"]).to_csv(OUT / "vix20_live_daily_nav.csv", index=False)
    pd.DataFrame(payload["monthly"]).to_csv(OUT / "vix20_live_monthly.csv", index=False)
    pd.DataFrame(payload["overlayTrades"]).to_csv(OUT / "vix20_live_overlay_trades.csv", index=False)
    pd.DataFrame(payload["vixTrades"]).to_csv(OUT / "vix20_live_vix11_trades.csv", index=False)
    print(output)
    print(json.dumps(payload["current"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
