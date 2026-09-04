# -*- coding: utf-8 -*-
"""時点を巻き戻して予測を採点するウォークフォワード検証。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine import build_decision
from pattern_match_decision import find_similar, timeframe_config


def _actual_scenario(value: float, flat_band: float) -> str:
    if value > flat_band:
        return "上昇"
    if value < -flat_band:
        return "下落"
    return "横ばい"


def walk_forward_validate(
    close: pd.Series,
    lib: dict,
    window: int,
    horizon: int,
    symbol: str | None = None,
    k: int = 100,
    max_points: int = 18,
) -> dict:
    """各検証時点より前に結果が確定した類似局面だけを使って採点する。"""
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if horizon not in lib["checks"]:
        raise ValueError(f"検証できない期間です: {horizon}")
    cfg = timeframe_config(lib["timeframe"])
    flat_band = cfg["flat_bands"][horizon]
    spacing = max(1, horizon)
    latest_end = len(close) - horizon
    earliest_end = max(window, latest_end - max_points * spacing)
    checkpoints = list(range(earliest_end, latest_end, spacing))[-max_points:]

    rows = []
    for end in checkpoints:
        as_of = close.index[end - 1]
        result = find_similar(
            close.iloc[end - window:end].to_numpy(),
            lib,
            k=k,
            min_matches=10,
            before_date=as_of,
            query_symbol=symbol,
            query_start_date=close.index[end - window],
        )
        if result is None:
            continue
        decision = build_decision(result, horizon)
        actual_return = float(close.iloc[end + horizon - 1] / close.iloc[end - 1] - 1)
        actual_scenario = _actual_scenario(actual_return, flat_band)
        rows.append({
            "as_of": as_of,
            "predicted_scenario": decision["most_likely"],
            "predicted_probability": decision["most_likely_probability"],
            "p_meaningful_up": decision["probabilities"]["上昇"],
            "decision": decision["label"],
            "score": decision["score"],
            "actual_scenario": actual_scenario,
            "actual_return": actual_return,
            "scenario_hit": decision["most_likely"] == actual_scenario,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rows": frame, "n": 0}

    actual_up = (frame["actual_scenario"] == "上昇").astype(float)
    buy_mask = frame["decision"].isin(["強気買い", "買い検討"])
    buy_returns = frame.loc[buy_mask, "actual_return"]
    return {
        "rows": frame,
        "n": len(frame),
        "scenario_accuracy": float(frame["scenario_hit"].mean()),
        "brier_up": float(np.mean((frame["p_meaningful_up"] - actual_up) ** 2)),
        "average_return": float(frame["actual_return"].mean()),
        "buy_count": int(buy_mask.sum()),
        "buy_average_return": float(buy_returns.mean()) if len(buy_returns) else None,
        "buy_win_rate": float((buy_returns > 0).mean()) if len(buy_returns) else None,
        "worst_return": float(frame["actual_return"].min()),
    }
