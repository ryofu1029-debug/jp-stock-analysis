# -*- coding: utf-8 -*-
"""「発動頻度」と「優位性」のトレードオフ表を作る。

RSI(21)<=20 は優位性が大きいが27年で759件しか出ない。しきい値を緩めれば
発動回数は増えるが優位性は薄れる。どこまで緩めれば実用的な頻度になり、
そのとき期待値がどれだけ残るのかを一覧にする。

訓練期間(1999-2012)と検証期間(2013-2026)を**並べて**表示する。
検証期間で最も良いものを選ぶのはデータスヌーピングなので、
「両方の期間で一貫してプラスか」を読むための表として使うこと。

使い方: python rule_tradeoff.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rule_optimize import COST, SPLIT, dedup, load, rsi

HOLD = 40                       # rule_optimize.py が採用した保有日数
RSI_GRID = [(p, t) for p in (14, 21) for t in (15, 20, 25, 30, 35, 40)]
DROP_GRID = [(w, d) for w in (5, 10, 15) for d in (0.10, 0.15, 0.20, 0.25)]


def evaluate() -> pd.DataFrame:
    series, _, _ = load()
    n_sym = len(series)
    sig: dict[tuple, dict[str, list[float]]] = {}
    base: dict[str, list[np.ndarray]] = {}
    years: dict[str, float] = {"訓練": 0.0, "検証": 0.0}

    for c, dates in series:
        n = len(c)
        fwd = np.full(n, np.nan)
        fwd[:n - HOLD] = c[HOLD:] / c[:n - HOLD] - 1.0
        is_tr = dates < SPLIT.to_datetime64()
        for era, m in (("訓練", is_tr), ("検証", ~is_tr)):
            years[era] += int(m.sum()) / 250.0
            base.setdefault(era, []).append(fwd[m & np.isfinite(fwd)][::5])

        for p, thr in RSI_GRID:
            r = rsi(c, p)
            cross = np.flatnonzero((r <= thr) & (np.roll(r, 1) > thr))
            for i in dedup(cross[cross > p]):
                if np.isfinite(fwd[i]):
                    era = "訓練" if is_tr[i] else "検証"
                    sig.setdefault((f"RSI{p}", f"≤{thr}"), {}).setdefault(era, []).append(fwd[i])

        for w, d in DROP_GRID:
            ratio = np.full(n, np.nan)
            ratio[w:] = c[w:] / c[:-w] - 1.0
            for i in dedup(np.flatnonzero(ratio <= -d)):
                if np.isfinite(fwd[i]):
                    era = "訓練" if is_tr[i] else "検証"
                    key = (f"急落{w}日", f"-{d:.0%}")
                    sig.setdefault(key, {}).setdefault(era, []).append(fwd[i])

    bstat = {e: (float(np.median(np.concatenate(v))),
                 float((np.concatenate(v) > 0).mean())) for e, v in base.items()}

    rows = []
    for (kind, thr), d in sig.items():
        r = {"種類": kind, "条件": thr}
        for era in ("訓練", "検証"):
            a = np.array(d.get(era, []))
            if len(a) < 30:
                r[f"{era}_超過"] = np.nan
                r[f"{era}_勝率差"] = np.nan
                continue
            bmed, bwin = bstat[era]
            r[f"{era}_超過"] = round((float(np.median(a)) - bmed) * 100, 2)
            r[f"{era}_勝率差"] = round((float((a > 0).mean()) - bwin) * 100, 1)
        n_all = len(d.get("訓練", [])) + len(d.get("検証", []))
        stock_years = years["訓練"] + years["検証"]   # 全銘柄合計の「銘柄×年」

        r["件数"] = n_all
        # 全銘柄を監視したとき1年に出るシグナル数 = 件数 ÷ (銘柄×年) × 銘柄数
        r["全体で年間"] = round(n_all / stock_years * n_sym, 1)
        r["1銘柄あたり何年に1回"] = round(stock_years / max(n_all, 1), 1)
        rows.append(r)

    df = pd.DataFrame(rows)
    df["コスト後(検証)"] = (df["検証_超過"] - COST * 100).round(2)
    df["両期間の最小"] = df[["訓練_超過", "検証_超過"]].min(axis=1)
    return df.sort_values("全体で年間")


def main() -> None:
    df = evaluate()
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 250)
    cols = ["種類", "条件", "件数", "全体で年間", "1銘柄あたり何年に1回", "訓練_超過", "検証_超過",
            "コスト後(検証)", "訓練_勝率差", "検証_勝率差", "両期間の最小"]
    print(f"=== 発動頻度と優位性のトレードオフ（保有{HOLD}営業日・中央値ベース） ===")
    print("※「全体で年間」= 587銘柄すべてを監視した場合に1年で出るシグナル数\n")
    print(df[cols].to_string(index=False))

    out = Path(__file__).parent / "data" / "rule_tradeoff.csv"
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{out.name} に保存")

    ok = df[(df["両期間の最小"] > 0)].sort_values("全体で年間", ascending=False)
    print(f"\n=== 訓練・検証の両方でプラスだった条件: {len(ok)}件 ===")
    print(ok[cols].to_string(index=False))


if __name__ == "__main__":
    main()
