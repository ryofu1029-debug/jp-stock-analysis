# -*- coding: utf-8 -*-
"""類似パターンの分布を、説明可能な参考投資判断へ変換する。

売買を自動執行するモデルではない。確率、期待値、下振れ、サンプル品質を
同じ画面で比較できるようにするための透明なルールベース評価である。
"""
from __future__ import annotations

import numpy as np


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _confidence(n: int, average_similarity: float, spread: float) -> tuple[int, str]:
    """一致度・件数・結果のばらつきから、データ信頼度を0〜100で表す。"""
    similarity_score = _clip((average_similarity - 0.50) / 0.40, 0, 1) * 55
    sample_score = _clip(n / 100, 0, 1) * 30
    dispersion_score = (1 - _clip(spread / 0.35, 0, 1)) * 15
    score = int(round(similarity_score + sample_score + dispersion_score))
    label = "高" if score >= 72 else "中" if score >= 48 else "低"
    return score, label


def _decision_label(
    score: int, probabilities: dict, expected_return: float, median_return: float
) -> str:
    """スコアと意味のある値動きの確率が矛盾しないように判定する。"""
    up = probabilities["上昇"]
    down = probabilities["下落"]
    if score >= 70 and up >= 0.50 and expected_return > 0 and median_return > 0:
        return "強気買い"
    if score >= 58 and up > down and expected_return > 0 and median_return > 0:
        return "買い検討"
    if score < 30 and down >= 0.50 and expected_return < 0:
        return "弱気"
    if score < 42 and down > up and expected_return < 0:
        return "売却検討"
    return "様子見"


def build_decision(result: dict, horizon: int | None = None) -> dict:
    """find_similar() の結果から、表示用の確率・リスク・判断を作る。"""
    if not result or not result.get("horizons"):
        raise ValueError("判断に必要な類似パターン結果がありません")

    horizon = horizon or max(result["horizons"])
    stats = result["horizons"][horizon]
    returns = np.asarray(result["forward_returns"], dtype=float)
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    average_gain = float(positive.mean()) if len(positive) else 0.0
    average_loss = float(abs(negative.mean())) if len(negative) else 0.0
    reward_risk = average_gain / average_loss if average_loss > 1e-9 else 0.0
    spread = stats["q90"] - stats["q10"]
    confidence_score, confidence_label = _confidence(
        result["n"], result.get("average_similarity", 0.0), spread
    )

    # 50を中立とし、上昇頻度・期待値・損益比を加点/減点する。
    probability_edge = (
        stats["probabilities"]["上昇"] - stats["probabilities"]["下落"]
    ) * 45
    return_edge = _clip(stats["mean"] / max(abs(stats["q10"]), 0.01), -1, 1) * 15
    reward_edge = _clip(reward_risk - 1.0, -1, 1) * 10
    reliability_adjustment = (confidence_score - 50) * 0.10
    score = int(round(_clip(
        50 + probability_edge + return_edge + reward_edge + reliability_adjustment,
        0,
        100,
    )))
    # 平均が一部の大幅上昇だけに引っ張られている場合は、買いスコアを抑える。
    if stats["median"] <= 0 and stats["probabilities"]["上昇"] < 0.50:
        score = min(score, 57)
    label = _decision_label(
        score, stats["probabilities"], stats["mean"], stats["median"]
    )

    scenario = stats["most_likely"]
    scenario_probability = stats["most_likely_probability"]
    unit = result.get("unit", "営業日")
    if label in ("強気買い", "買い検討"):
        action = "上昇優位ですが、下振れ幅を確認しながら分割で検討する局面です。"
    elif label == "様子見":
        action = "方向感が十分に偏っていないため、新規判断は次のシグナルまで待つ局面です。"
    else:
        action = "下落リスクが優勢です。新規購入より、保有量と撤退条件の確認を優先する局面です。"

    return {
        "horizon": horizon,
        "horizon_label": f"{horizon}{unit}後",
        "most_likely": scenario,
        "most_likely_probability": scenario_probability,
        "probabilities": stats["probabilities"],
        "p_up": stats["p_up"],
        "median_return": stats["median"],
        "expected_return": stats["mean"],
        "downside": stats["q10"],
        "upside": stats["q90"],
        "average_gain": average_gain,
        "average_loss": average_loss,
        "reward_risk": reward_risk,
        "score": score,
        "label": label,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "sample_size": result["n"],
        "average_similarity": result.get("average_similarity", 0.0),
        "action": action,
        "flat_band": stats["flat_band"],
    }
