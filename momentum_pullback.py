# -*- coding: utf-8 -*-
"""「上昇トレンドの銘柄が一時的に押した」ところを買うハイブリッド戦略。

これまでの検証で分かったこと:
  - 短期の急騰直後を買うのは軽く逆効果（pattern_verify.py の「15日で15%急騰」）
  - RSI21<=20/25の押し目は、アイドル中を現金でなくベンチマークに置く設計
    (mechanical_overlay.py) だと、リスクを増やさずベンチマークに勝てた

ここではRSIの押し目シグナルに「中長期の上昇トレンド」条件を重ねる:
  - PRICE>SMA200: 株価が200日線より上（最も単純な長期トレンド条件）
  - GOLDEN      : SMA50がSMA200より上（ゴールデンクロス状態、より強い確認）

「上昇トレンドの銘柄を買っている」という心理的な納得感を保ちながら、
実証済みの押し目タイミングの優位性を使えるかを検証する。

同じ資産曲線を1999-2026通しで1回だけ計算し、その中の1999-2012(訓練)区間・
2013-2026(検証)区間それぞれの年率リターンを見る（複利の連続性を保ったまま
期間ごとの実現リターンを測れる）。

最後に、この中で最も安定していた条件を「現在(最新データ)」に当てはめ、
条件に合う実在銘柄を一覧にする。

使い方: python momentum_pullback.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mechanical_backtest import COST, HOLD, N_SLOTS, build_matrix, load_named, metrics
from mechanical_overlay import simulate_overlay
from rule_optimize import rsi
from tickers import TICKERS

SPLIT = pd.Timestamp("2012-10-03")
RSI_PERIOD = 21
CONFIGS = [
    ("RSI21_20_単体", 20, None),
    ("RSI21_25_単体", 25, None),
    ("RSI21_20_PRICE>SMA200", 20, "PRICE"),
    ("RSI21_25_PRICE>SMA200", 25, "PRICE"),
    ("RSI21_20_GOLDEN", 20, "GOLDEN"),
    ("RSI21_25_GOLDEN", 25, "GOLDEN"),
]


def sma(c: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(c).rolling(period, min_periods=period).mean().to_numpy()


def trend_mask(c: np.ndarray, kind: str | None) -> np.ndarray:
    if kind is None:
        return np.ones(len(c), dtype=bool)
    s200 = sma(c, 200)
    if kind == "PRICE":
        return c > s200
    s50 = sma(c, 50)
    return s50 > s200  # GOLDEN


def gen_signals(sym: str, native: pd.Series, master: pd.DatetimeIndex,
                thr: int, trend_kind: str | None) -> list[tuple[int, float, str]]:
    c = native.to_numpy()
    n = len(c)
    r = rsi(c, RSI_PERIOD)
    tmask = trend_mask(c, trend_kind)
    cross = np.flatnonzero((r <= thr) & (np.roll(r, 1) > thr) & tmask)
    cross = cross[cross > max(RSI_PERIOD, 200)]
    if len(cross) == 0:
        return []
    loc = master.get_indexer(native.index[cross])
    return [(int(di), float(pr), sym) for di, pr in zip(loc, r[cross]) if di >= 0]


def period_cagr(equity: np.ndarray, dates: pd.DatetimeIndex, lo, hi) -> float:
    m = (dates >= lo) & (dates < hi) if hi else (dates >= lo)
    idx = np.flatnonzero(m)
    if len(idx) < 30:
        return np.nan
    years = (dates[idx[-1]] - dates[idx[0]]).days / 365.25
    if years <= 0:
        return np.nan
    return float((equity[idx[-1]] / equity[idx[0]]) ** (1 / years) - 1) * 100


def main() -> None:
    print("=== データ読み込み ===")
    series = load_named()
    price_df = build_matrix(series)
    master = price_df.index
    price = price_df.to_numpy(dtype=np.float32)
    cols = {c: i for i, c in enumerate(price_df.columns)}

    rets = price_df.pct_change()
    bench = (1 + rets.mean(axis=1).fillna(0)).cumprod().to_numpy()
    bench = bench / bench[0]
    bm = metrics(bench, master, np.full(len(master), N_SLOTS), N_SLOTS)
    b_tr = period_cagr(bench, master, master[0], SPLIT)
    b_te = period_cagr(bench, master, SPLIT, None)
    print(f"【ベンチマーク】通期CAGR {bm['CAGR%']}%  訓練期CAGR {b_tr:.2f}%  検証期CAGR {b_te:.2f}%  "
          f"最大DD {bm['最大DD%']}%\n")

    rows = []
    curves = {"ベンチマーク": bench}
    last_signals: dict[str, list[tuple[float, str]]] = {}

    for name, thr, trend_kind in CONFIGS:
        sigs: dict[int, list[tuple[float, str]]] = {}
        for sym, s in series.items():
            for di, pr, sy in gen_signals(sym, s, master, thr, trend_kind):
                sigs.setdefault(di, []).append((pr, sy))
        for di in sigs:
            sigs[di].sort()

        eq, nop, trades = simulate_overlay(price, cols, bench, sigs, N_SLOTS, HOLD, COST)
        m = metrics(eq, master, nop, N_SLOTS)
        tr_cagr = period_cagr(eq, master, master[0], SPLIT)
        te_cagr = period_cagr(eq, master, SPLIT, None)
        rows.append({
            "設定": name, "取引数": len(trades),
            "通期CAGR%": m["CAGR%"], "訓練期CAGR%": round(tr_cagr, 2), "検証期CAGR%": round(te_cagr, 2),
            "対ベンチ倍率": round(float(eq[-1] / bench[-1]), 2),
            "最大DD%": m["最大DD%"], "CAGR/vol": m["CAGR/vol"],
        })
        curves[name] = eq
        if sigs:
            last_day = max(sigs)
            print(f"  {name:<24} 通期{m['CAGR%']:>6.2f}%  訓練{tr_cagr:>6.2f}%  検証{te_cagr:>6.2f}%  "
                  f"対ベンチ{float(eq[-1]/bench[-1]):>6.2f}倍  最大DD{m['最大DD%']:>6.1f}%  "
                  f"CAGR/vol{m['CAGR/vol']:>5.2f}  取引{len(trades)}")

    df = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    print("\n【全設定比較】訓練・検証どちらもベンチマークを上回るものが本命")
    print(df.to_string(index=False))

    out = Path(__file__).parent / "data"
    df.to_csv(out / "momentum_pullback.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves, index=master).to_csv(out / "momentum_pullback_curves.csv", encoding="utf-8-sig")

    # ---- 訓練・検証どちらもベンチマークを上回り、かつ最も安定した設定を選ぶ ----
    ok = df[(df["訓練期CAGR%"] > b_tr) & (df["検証期CAGR%"] > b_te)].copy()
    if ok.empty:
        print("\n訓練・検証の両方でベンチマークを上回った設定はありませんでした。")
        return
    ok["安定度"] = ok[["訓練期CAGR%", "検証期CAGR%"]].sub([b_tr, b_te], axis=1).min(axis=1)
    winner = ok.sort_values("安定度", ascending=False).iloc[0]
    print(f"\n=== 採用: {winner['設定']}（訓練・検証とも上回り、差が最も安定） ===")

    # ---- 現在時点でのスクリーニング ----
    wname, wthr, wtrend = next((n, t, k) for n, t, k in CONFIGS if n == winner["設定"])
    print(f"\n=== 現在（最新データ）でこの条件に合う銘柄 ===")
    picks = []
    for sym, s in series.items():
        c = s.to_numpy()
        if len(c) < max(RSI_PERIOD, 200) + 5:
            continue
        r = rsi(c, RSI_PERIOD)[-1]
        tmask = trend_mask(c, wtrend)[-1]
        if np.isfinite(r) and r <= wthr and tmask:
            name_jp, group, sector = TICKERS.get(sym, (sym, "?", "?"))
            s200 = sma(c, 200)[-1]
            picks.append({
                "コード": sym, "銘柄名": name_jp, "業種": sector,
                "終値": round(float(c[-1]), 1), "RSI21": round(float(r), 1),
                "SMA200乖離%": round(float(c[-1] / s200 - 1) * 100, 1),
                "最新日": str(s.index[-1].date()),
            })
    if not picks:
        print("該当銘柄なし（この条件は稀にしか発動しません。日を改めて確認してください）。")
    else:
        pdf = pd.DataFrame(picks).sort_values("RSI21")
        print(pdf.to_string(index=False))
        pdf.to_csv(out / "momentum_pullback_picks.csv", index=False, encoding="utf-8-sig")
        print(f"\ndata/momentum_pullback_picks.csv に保存（{len(pdf)}銘柄）")


if __name__ == "__main__":
    main()
