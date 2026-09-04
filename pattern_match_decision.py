# -*- coding: utf-8 -*-
"""投資判断スカウター専用の価格パターン照合エンジン。

日足と週足を別々の時系列として扱い、現在の形に似た過去局面を検索する。
照合後は、指定期間後のリターン分布をそのまま確率として集計する。
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TIMEFRAME_CONFIG = {
    "日足": {
        "resample": None,
        "horizon": 20,
        "checks": (5, 10, 20),
        "step": 5,
        "unit": "営業日",
        "spans": {"1ヶ月": 20, "3ヶ月": 60, "6ヶ月": 120, "1年": 240},
        "flat_bands": {5: 0.01, 10: 0.015, 20: 0.02},
    },
    "週足": {
        "resample": "W-FRI",
        "horizon": 12,
        "checks": (4, 8, 12),
        "step": 2,
        "unit": "週",
        "spans": {"6ヶ月": 26, "1年": 52, "2年": 104, "3年": 156},
        "flat_bands": {4: 0.02, 8: 0.03, 12: 0.04},
    },
}

# 旧コードとの互換性。新規コードは timeframe_config() を使う。
WINDOW_BY_SPAN = TIMEFRAME_CONFIG["日足"]["spans"]


def timeframe_config(timeframe: str) -> dict:
    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"未対応の足種です: {timeframe}")
    return TIMEFRAME_CONFIG[timeframe]


def prepare_close(data: pd.DataFrame | pd.Series, timeframe: str = "日足") -> pd.Series:
    """入力を照合用の終値系列に変換する。週足は金曜締めで再集計する。"""
    cfg = timeframe_config(timeframe)
    close = data if isinstance(data, pd.Series) else data["Close"]
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if cfg["resample"]:
        if not isinstance(close.index, pd.DatetimeIndex):
            raise ValueError("週足への変換には日付インデックスが必要です")
        close = close.resample(cfg["resample"]).last().dropna()
    return close


def _zscore(w: np.ndarray) -> np.ndarray | None:
    std = w.std()
    if std < 1e-9:
        return None
    return (w - w.mean()) / std


def build_library(
    window: int = 60,
    symbols: tuple[str, ...] | None = None,
    step: int | None = None,
    timeframe: str = "日足",
) -> dict:
    """過去データから、形状とその後のリターンをまとめた照合ライブラリを作る。"""
    cfg = timeframe_config(timeframe)
    horizon = cfg["horizon"]
    step = step or cfg["step"]
    X, fut, syms, dates, future_end_dates = [], [], [], [], []
    files = (
        sorted(PROCESSED_DIR / f"{s}.csv" for s in symbols)
        if symbols
        else sorted(PROCESSED_DIR.glob("*.csv"))
    )
    for csv in files:
        if not csv.exists():
            continue
        frame = pd.read_csv(csv, usecols=["Date", "Close"], parse_dates=["Date"])
        close = prepare_close(frame.set_index("Date")["Close"], timeframe)
        c = close.to_numpy(dtype=np.float32)
        d = close.index.strftime("%Y-%m-%d").to_numpy()
        for end in range(window, len(c) - horizon, step):
            z = _zscore(c[end - window:end])
            if z is None:
                continue
            X.append(z.astype(np.float32))
            fut.append(c[end:end + horizon] / c[end - 1] - 1.0)
            syms.append(csv.stem)
            dates.append(d[end - 1])
            future_end_dates.append(d[end + horizon - 1])

    if not X:
        return {
            "X": np.empty((0, window), dtype=np.float32),
            "fut": np.empty((0, horizon), dtype=np.float32),
            "syms": np.array([]),
            "dates": np.array([]),
            "future_end_dates": np.array([]),
            "window": window,
            "step": step,
            "timeframe": timeframe,
            "horizon": horizon,
            "checks": cfg["checks"],
            "unit": cfg["unit"],
        }

    return {
        "X": np.stack(X),
        "fut": np.stack(fut).astype(np.float32),
        "syms": np.array(syms),
        "dates": np.array(dates),
        "future_end_dates": np.array(future_end_dates),
        "window": window,
        "step": step,
        "timeframe": timeframe,
        "horizon": horizon,
        "checks": cfg["checks"],
        "unit": cfg["unit"],
    }


def _scenario_stats(returns: np.ndarray, flat_band: float) -> dict:
    probabilities = {
        "上昇": float((returns > flat_band).mean()),
        "横ばい": float(((returns >= -flat_band) & (returns <= flat_band)).mean()),
        "下落": float((returns < -flat_band).mean()),
    }
    most_likely = max(probabilities, key=probabilities.get)
    return {
        "probabilities": probabilities,
        "most_likely": most_likely,
        "most_likely_probability": probabilities[most_likely],
        "flat_band": flat_band,
    }


def find_similar(
    query: np.ndarray,
    lib: dict,
    k: int = 100,
    min_matches: int = 10,
    min_similarity: float = 0.55,
    before_date: str | pd.Timestamp | None = None,
    query_symbol: str | None = None,
    query_start_date: str | pd.Timestamp | None = None,
) -> dict | None:
    """query と似た過去局面を選び、その後の確率分布を返す。"""
    window = lib["window"]
    if len(lib["X"]) == 0:
        return None
    if len(query) != window:
        idx = np.linspace(0, len(query) - 1, window)
        query = np.interp(idx, np.arange(len(query)), query)
    zq = _zscore(np.asarray(query, dtype=np.float32))
    if zq is None:
        return None

    sims = lib["X"] @ zq / window
    eligible = np.ones(len(sims), dtype=bool)
    if before_date is not None:
        if "future_end_dates" not in lib:
            raise ValueError("過去検証には future_end_dates を含むライブラリが必要です")
        cutoff = np.datetime64(pd.Timestamp(before_date).date())
        eligible = lib["future_end_dates"].astype("datetime64[D]") < cutoff
    if query_symbol is not None and query_start_date is not None:
        if "future_end_dates" not in lib:
            raise ValueError("自己銘柄の重複除外には future_end_dates が必要です")
        query_start = np.datetime64(pd.Timestamp(query_start_date).date())
        same_symbol = lib["syms"] == query_symbol
        overlaps_query = lib["future_end_dates"].astype("datetime64[D]") >= query_start
        eligible &= ~(same_symbol & overlaps_query)
    ranking = np.where(eligible, sims, -np.inf)
    order = np.argsort(ranking)[::-1]
    chosen, used = [], {}
    min_gap = max(1, window // (2 * lib.get("step", 1)))
    for i in order:
        if not np.isfinite(ranking[i]) or sims[i] < min_similarity:
            break
        symbol = lib["syms"][i]
        if all(abs(i - j) >= min_gap for j in used.get(symbol, [])):
            chosen.append(i)
            used.setdefault(symbol, []).append(i)
        if len(chosen) >= k:
            break
    if len(chosen) < min_matches:
        return None

    fut = lib["fut"][chosen]
    paths = 1.0 + fut
    cfg = timeframe_config(lib.get("timeframe", "日足"))

    horizons = {}
    for h in lib.get("checks", cfg["checks"]):
        r = fut[:, h - 1]
        horizons[h] = {
            "p_up": float((r > 0).mean()),
            "median": float(np.median(r)),
            "mean": float(np.mean(r)),
            "q10": float(np.percentile(r, 10)),
            "q25": float(np.percentile(r, 25)),
            "q75": float(np.percentile(r, 75)),
            "q90": float(np.percentile(r, 90)),
            **_scenario_stats(r, cfg["flat_bands"][h]),
        }

    percentiles = {
        f"p{q}": np.percentile(paths, q, axis=0) for q in (10, 25, 50, 75, 90)
    }
    final_returns = fut[:, -1]
    matches = pd.DataFrame({
        "symbol": lib["syms"][chosen],
        "end_date": lib["dates"][chosen],
        "similarity": np.round(sims[chosen] * 100, 1),
        "forward_return_pct": np.round(final_returns * 100, 2),
    })

    return {
        "horizons": horizons,
        "paths_pct": percentiles,
        "forward_returns": final_returns,
        "ret20": final_returns,  # 旧UI互換
        "matches": matches,
        "n": len(chosen),
        "average_similarity": float(np.mean(sims[chosen])),
        "timeframe": lib.get("timeframe", "日足"),
        "unit": lib.get("unit", "営業日"),
        "horizon": lib.get("horizon", len(final_returns)),
    }
