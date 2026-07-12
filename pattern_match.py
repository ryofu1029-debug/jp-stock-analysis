# -*- coding: utf-8 -*-
"""パターン照合エンジン: 与えられた価格の「形」に似た過去の局面を
300社×5年の全履歴から探し、その後の値動きを確率・分布として集計する。

手法: 形状をzスコア正規化し、ピアソン相関で類似度を測る k近傍方式。
「過去に似た形のあと、20営業日でどう動いたか」の頻度分布を返す。
※ 過去の頻度であって将来の保証ではない。
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HORIZON = 20          # 何営業日先まで集計するか
STEP = 5              # 窓をずらす間隔（営業日）
HORIZON_CHECKS = (5, 10, 20)

# 画像のチャート期間 → 窓の長さ（営業日）
WINDOW_BY_SPAN = {"3ヶ月": 60, "6ヶ月": 120, "1年": 240}


def _zscore(w: np.ndarray) -> np.ndarray | None:
    std = w.std()
    if std < 1e-9:
        return None
    return (w - w.mean()) / std


def build_library(window: int = 60) -> dict:
    """全銘柄の過去データから照合用ライブラリ（窓と「その後」）を構築する。"""
    X, fut, syms, dates = [], [], [], []
    for csv in sorted(PROCESSED_DIR.glob("*.csv")):
        close = pd.read_csv(csv, usecols=["Date", "Close"], parse_dates=["Date"])
        c = close["Close"].to_numpy(dtype=np.float32)
        d = close["Date"].dt.strftime("%Y-%m-%d").to_numpy()
        for end in range(window, len(c) - HORIZON, STEP):
            z = _zscore(c[end - window:end])
            if z is None:
                continue
            X.append(z.astype(np.float32))
            fut.append(c[end:end + HORIZON] / c[end - 1] - 1.0)  # その後の累積リターン
            syms.append(csv.stem)
            dates.append(d[end - 1])
    return {
        "X": np.stack(X),
        "fut": np.stack(fut).astype(np.float32),
        "syms": np.array(syms),
        "dates": np.array(dates),
        "window": window,
    }


def find_similar(query: np.ndarray, lib: dict, k: int = 100) -> dict | None:
    """query の形に似た過去の局面 top-k を探し、その後の統計を返す。"""
    window = lib["window"]
    if len(query) != window:
        idx = np.linspace(0, len(query) - 1, window)
        query = np.interp(idx, np.arange(len(query)), query)
    zq = _zscore(np.asarray(query, dtype=np.float32))
    if zq is None:
        return None

    sims = lib["X"] @ zq / window  # 両方zスコア済みなので ≒ ピアソン相関

    # 類似度順に、同一銘柄の重なり合う窓を除外しながら k 件選ぶ
    order = np.argsort(sims)[::-1]
    chosen, used = [], {}
    min_gap = max(1, window // (2 * STEP))  # 窓インデックスの最小間隔
    for i in order:
        s = lib["syms"][i]
        if all(abs(i - j) >= min_gap for j in used.get(s, [])):
            chosen.append(i)
            used.setdefault(s, []).append(i)
        if len(chosen) >= k:
            break
    if len(chosen) < 10:
        return None

    fut = lib["fut"][chosen]          # (k, HORIZON) その後の累積リターン
    paths = 1.0 + fut                 # 窓の最終値=1とした相対価格

    horizons = {}
    for h in HORIZON_CHECKS:
        r = fut[:, h - 1]
        horizons[h] = {
            "p_up": float((r > 0).mean()),
            "median": float(np.median(r)),
            "q25": float(np.percentile(r, 25)),
            "q75": float(np.percentile(r, 75)),
        }

    percentiles = {f"p{q}": np.percentile(paths, q, axis=0) for q in (10, 25, 50, 75, 90)}

    matches = pd.DataFrame({
        "symbol": lib["syms"][chosen],
        "end_date": lib["dates"][chosen],
        "similarity": np.round(sims[chosen] * 100, 1),
        "ret_20d_pct": np.round(fut[:, -1] * 100, 2),
    })

    return {
        "horizons": horizons,
        "paths_pct": percentiles,
        "ret20": fut[:, -1],
        "matches": matches,
        "n": len(chosen),
    }
