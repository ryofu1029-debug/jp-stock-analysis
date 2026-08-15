# -*- coding: utf-8 -*-
"""業種分散モメンタムとレジーム切替を検証する。

これまでの経緯:
  - RSI21<=25の押し目買い(mechanical_overlay.py)は訓練・検証とも合格
  - 古典モメンタム(momentum_classic.py)は検証期で圧勝だが訓練期で不合格
    原因は(a)暴落時のmomentum crash (b)上位銘柄の業種集中

ここでは2つの改良を検証する:
  1. 業種分散モメンタム — 同一業種の採用数に上限をかける / 業種内相対強度で測る
     / ボラティリティで割る（リスク調整モメンタム）
  2. レジーム切替 — 市場が200日線より上ならモメンタム、下ならRSI押し目に退避
     （RSIは下げ相場で強く、モメンタムはトレンド相場で強いため）

合格条件: 訓練期CAGR > 11.44% かつ 検証期CAGR > 19.91%（ベンチマーク実績）
パラメータは訓練期の成績のみで選ぶ。

使い方: python strat_sector_regime.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mechanical_backtest import build_matrix, load_named
from momentum_classic import metrics, period_cagr
from rule_optimize import rsi
from tickers import TICKERS

SPLIT = pd.Timestamp("2012-10-03")
N_TOP = 20
REBAL = 21
COST = 0.002
BENCH_TRAIN, BENCH_TEST = 11.44, 19.91


def sector_of(sym: str) -> str:
    return TICKERS.get(sym, (sym, "?", "?"))[2]


def pick_with_sector_cap(score: np.ndarray, syms: list[str], n_top: int,
                         cap: int | None) -> np.ndarray:
    """スコア降順に選ぶが、同一業種は cap 銘柄までに制限する。"""
    order = np.argsort(-score)
    chosen, used = [], {}
    for i in order:
        if not np.isfinite(score[i]):
            continue
        if cap is not None:
            sec = sector_of(syms[i])
            if used.get(sec, 0) >= cap:
                continue
            used[sec] = used.get(sec, 0) + 1
        chosen.append(i)
        if len(chosen) >= n_top:
            break
    return np.array(chosen, dtype=int)


def momentum_variant(price_df: pd.DataFrame, lookback: int, skip: int,
                     mode: str, cap: int | None, n_top: int = N_TOP,
                     rebal: int = REBAL, cost: float = COST) -> np.ndarray:
    """モメンタムの各種変種。mode: raw / volscaled / sectorrel"""
    syms = list(price_df.columns)
    price = price_df.to_numpy(dtype=np.float64)
    pct = price_df.pct_change().to_numpy(dtype=np.float64)
    n_days = price.shape[0]
    sectors = np.array([sector_of(s) for s in syms])

    start = lookback + skip + 5
    rebal_set = set(range(start, n_days, rebal))
    equity = np.empty(n_days)
    equity[0] = 1.0
    holdings = np.array([], dtype=int)

    for day in range(1, n_days):
        if len(holdings):
            r = pct[day, holdings]
            r = r[np.isfinite(r)]
            equity[day] = equity[day - 1] * (1 + (float(r.mean()) if len(r) else 0.0))
        else:
            equity[day] = equity[day - 1]

        if day in rebal_set:
            p_rec = price[day - skip] if skip > 0 else price[day]
            p_old = price[day - lookback]
            valid = np.isfinite(p_rec) & np.isfinite(p_old) & (p_old > 0) & np.isfinite(price[day])
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = np.where(valid, p_rec / p_old - 1, np.nan)

            if mode == "volscaled":
                win = pct[max(0, day - lookback):day]
                vol = np.nanstd(win, axis=0)
                score = np.where(valid & (vol > 0), raw / np.where(vol > 0, vol, np.nan), np.nan)
            elif mode == "sectorrel":       # 業種平均を引いた相対強度
                score = np.full(len(syms), np.nan)
                for sec in np.unique(sectors):
                    m = (sectors == sec) & valid
                    if m.sum() >= 3:
                        score[m] = raw[m] - np.nanmean(raw[m])
            else:
                score = raw

            score = np.where(np.isfinite(score), score, -np.inf)
            new = pick_with_sector_cap(score, syms, n_top, cap)
            changed = len(set(holdings.tolist()) ^ set(new.tolist()))
            equity[day] *= (1 - cost * changed / (2 * max(n_top, 1)))
            holdings = new
    return equity


def rsi_overlay_equity(series: dict, price_df: pd.DataFrame, bench: np.ndarray,
                       thr: int = 25, period: int = 21, hold: int = 20,
                       n_slots: int = N_TOP) -> np.ndarray:
    """合格済みのRSI押し目(overlay)の資産曲線。レジーム切替の退避先に使う。"""
    from mechanical_overlay import simulate_overlay
    master = price_df.index
    price = price_df.to_numpy(dtype=np.float32)
    cols = {c: i for i, c in enumerate(price_df.columns)}
    sigs: dict[int, list[tuple[float, str]]] = {}
    for sym, s in series.items():
        c = s.to_numpy()
        r = rsi(c, period)
        cross = np.flatnonzero((r <= thr) & (np.roll(r, 1) > thr))
        cross = cross[cross > period]
        if len(cross) == 0:
            continue
        loc = master.get_indexer(s.index[cross])
        for di, pr in zip(loc, r[cross]):
            if di >= 0:
                sigs.setdefault(int(di), []).append((float(pr), sym))
    for di in sigs:
        sigs[di].sort()
    eq, _, _ = simulate_overlay(price, cols, bench, sigs, n_slots, hold, COST)
    return eq


def regime_blend(eq_mom: np.ndarray, eq_rsi: np.ndarray, bench: np.ndarray,
                 ma: int = 200, lag: int = 1) -> np.ndarray:
    """市場が ma 日線より上ならモメンタム、下ならRSI押し目に切り替える。

    判定は lag 日前の情報のみを使う（当日の終値を見て当日切り替えるのは先読み）。
    """
    b = pd.Series(bench)
    above = (b > b.rolling(ma, min_periods=ma).mean()).shift(lag).fillna(False).to_numpy()
    r_mom = np.diff(np.log(eq_mom), prepend=np.log(eq_mom[0]))
    r_rsi = np.diff(np.log(eq_rsi), prepend=np.log(eq_rsi[0]))
    r = np.where(above, r_mom, r_rsi)
    return np.exp(np.cumsum(r))


def report(name: str, eq: np.ndarray, master, bench: np.ndarray, rows: list) -> None:
    m = metrics(eq, master)
    tr = period_cagr(eq, master, master[0], SPLIT)
    te = period_cagr(eq, master, SPLIT, None)
    ok = (tr > BENCH_TRAIN) and (te > BENCH_TEST)
    rows.append({"設定": name, "通期CAGR%": m["CAGR%"], "訓練期%": round(tr, 2),
                 "検証期%": round(te, 2), "対ベンチ": round(float(eq[-1] / bench[-1]), 2),
                 "最大DD%": m["最大DD%"], "CAGR/vol": m["CAGR/vol"],
                 "合格": "○" if ok else "✕"})
    print(f"  {name:<34} 訓練{tr:>6.2f}%  検証{te:>6.2f}%  対ベンチ{float(eq[-1]/bench[-1]):>6.2f}倍  "
          f"CAGR/vol{m['CAGR/vol']:>5.2f}  {'○合格' if ok else '✕'}")


def main() -> None:
    print("=== データ読み込み ===")
    series = load_named()
    price_df = build_matrix(series)
    master = price_df.index
    bench = (1 + price_df.pct_change().mean(axis=1).fillna(0)).cumprod().to_numpy()
    bench = bench / bench[0]
    print(f"587銘柄 / ベンチ 訓練{BENCH_TRAIN}% 検証{BENCH_TEST}%\n")

    rows: list[dict] = []
    curves = {"ベンチマーク": bench}

    print("【1】業種分散モメンタム")
    variants = [
        ("MOM 12-1 業種上限3", 252, 21, "raw", 3),
        ("MOM 12-1 業種上限2", 252, 21, "raw", 2),
        ("MOM 12-1 業種内相対", 252, 21, "sectorrel", None),
        ("MOM 12-1 ボラ調整", 252, 21, "volscaled", None),
        ("MOM 12-1 ボラ調整+上限3", 252, 21, "volscaled", 3),
        ("MOM 6-1 ボラ調整+上限3", 126, 21, "volscaled", 3),
    ]
    eq_mom_best = None
    for name, lb, sk, mode, cap in variants:
        eq = momentum_variant(price_df, lb, sk, mode, cap)
        report(name, eq, master, bench, rows)
        curves[name] = eq
        if name == "MOM 12-1 ボラ調整+上限3":
            eq_mom_best = eq

    print("\n【2】レジーム切替（市場200日線でモメンタム↔RSI押し目）")
    eq_rsi = rsi_overlay_equity(series, price_df, bench)
    report("RSI押し目(参考・合格済み)", eq_rsi, master, bench, rows)
    curves["RSI押し目"] = eq_rsi

    eq_mom_plain = momentum_variant(price_df, 252, 21, "raw", None)
    for ma in (100, 150, 200):
        eq = regime_blend(eq_mom_plain, eq_rsi, bench, ma=ma)
        report(f"切替 MOM<->RSI (MA{ma})", eq, master, bench, rows)
        curves[f"切替MA{ma}"] = eq
    if eq_mom_best is not None:
        eq = regime_blend(eq_mom_best, eq_rsi, bench, ma=200)
        report("切替 ボラ調整MOM<->RSI (MA200)", eq, master, bench, rows)
        curves["切替ボラ調整MA200"] = eq

    print("\n【3】常時併用（資金を半分ずつ、日次リバランス）")
    for nm, e2 in (("MOM素", eq_mom_plain), ("MOMボラ調整", eq_mom_best)):
        if e2 is None:
            continue
        r = 0.5 * np.diff(np.log(e2), prepend=np.log(e2[0])) + \
            0.5 * np.diff(np.log(eq_rsi), prepend=np.log(eq_rsi[0]))
        eq = np.exp(np.cumsum(r))
        report(f"50:50 {nm} + RSI押し目", eq, master, bench, rows)
        curves[f"50-50 {nm}"] = eq

    df = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    print("\n【全結果】")
    print(df.to_string(index=False))

    out = Path(__file__).parent / "data"
    df.to_csv(out / "strat_sector_regime.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves, index=master).to_csv(out / "strat_sector_regime_curves.csv",
                                              encoding="utf-8-sig")
    passed = df[df["合格"] == "○"]
    print(f"\n=== 合格: {len(passed)}件 / {len(df)}件 ===")
    if len(passed):
        print(passed.sort_values("CAGR/vol", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
