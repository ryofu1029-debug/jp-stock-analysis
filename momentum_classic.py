# -*- coding: utf-8 -*-
"""古典的モメンタム投資（Jegadeesh & Titman 1993型）を単独で検証する。

RSIの押し目買い（平均回帰）とは仕組みが正反対: 「過去6-12ヶ月で強かった
銘柄は、その後もアウトパフォームしやすい」という中期の順張り効果を使う。
直近1ヶ月は除外する（短期は逆に反落しやすいという別の効果と混ざるため）。

毎月末（約21営業日ごと）に、直近L営業日〜直近skip営業日の騰落率で全銘柄を
ランキングし、上位N銘柄を均等保有。入れ替え時のみ売買コストがかかる。

RSIの押し目買いとは混ぜず、独立した戦略として検証する（別々に持てば
シグナルが出ない期間を補い合える可能性がある）。

使い方: python momentum_classic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mechanical_backtest import build_matrix, load_named
from tickers import TICKERS

SPLIT = pd.Timestamp("2012-10-03")
N_TOP = 20
REBAL_FREQ = 21          # 約1ヶ月ごとに入れ替え
COST = 0.002             # 往復コスト
FORMATIONS = [
    ("12-1ヶ月モメンタム(古典)", 252, 21),
    ("6-1ヶ月モメンタム", 126, 21),
    ("3-1ヶ月モメンタム", 63, 21),
    ("12-0ヶ月(直近含む・比較用)", 252, 0),
]


def momentum_backtest(price_df: pd.DataFrame, lookback: int, skip: int,
                      n_top: int = N_TOP, rebal_freq: int = REBAL_FREQ,
                      cost: float = COST):
    dates = price_df.index
    price = price_df.to_numpy(dtype=np.float64)
    pct = price_df.pct_change().to_numpy(dtype=np.float64)
    n_days, n_sym = price.shape

    start = lookback + skip
    rebal_set = set(range(start, n_days, rebal_freq))

    equity = np.empty(n_days)
    equity[0] = 1.0
    holdings = np.array([], dtype=int)
    n_hold_curve = np.zeros(n_days, dtype=int)

    for day in range(1, n_days):
        if len(holdings) > 0:
            r = pct[day, holdings]
            r = r[np.isfinite(r)]
            ret = float(r.mean()) if len(r) else 0.0
        else:
            ret = 0.0
        equity[day] = equity[day - 1] * (1 + ret)

        if day in rebal_set:
            p_recent = price[day - skip] if skip > 0 else price[day]
            p_old = price[day - lookback]
            valid = np.isfinite(p_recent) & np.isfinite(p_old) & (p_old > 0) & np.isfinite(price[day])
            score = np.where(valid, p_recent / np.where(p_old > 0, p_old, np.nan) - 1, -np.inf)
            order = np.argsort(-score)
            candidates = order[np.isfinite(score[order]) & (score[order] > -np.inf)]
            new_holdings = candidates[:n_top]

            changed = len(set(holdings.tolist()) ^ set(new_holdings.tolist()))
            turnover_frac = changed / (2 * max(n_top, 1))
            equity[day] *= (1 - cost * turnover_frac)
            holdings = new_holdings
        n_hold_curve[day] = len(holdings)

    return equity, n_hold_curve


def metrics(equity: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(equity)
    dd = float((equity / peak - 1).min())
    vol = float(np.std(np.diff(np.log(equity))) * np.sqrt(250))
    return {"CAGR%": round(cagr * 100, 2), "最終倍率": round(float(equity[-1] / equity[0]), 2),
            "最大DD%": round(dd * 100, 1), "年率vol%": round(vol * 100, 1),
            "CAGR/vol": round(cagr / vol, 2) if vol > 0 else np.nan}


def period_cagr(equity: np.ndarray, dates: pd.DatetimeIndex, lo, hi) -> float:
    m = (dates >= lo) & (dates < hi) if hi is not None else (dates >= lo)
    idx = np.flatnonzero(m)
    if len(idx) < 30:
        return np.nan
    years = (dates[idx[-1]] - dates[idx[0]]).days / 365.25
    return float((equity[idx[-1]] / equity[idx[0]]) ** (1 / years) - 1) * 100 if years > 0 else np.nan


def main() -> None:
    print("=== データ読み込み ===")
    series = load_named()
    price_df = build_matrix(series)
    master = price_df.index
    print(f"日本株 {len(series)}銘柄 / {master[0].date()} 〜 {master[-1].date()}")

    rets = price_df.pct_change()
    bench = (1 + rets.mean(axis=1).fillna(0)).cumprod().to_numpy()
    bench = bench / bench[0]
    bm = metrics(bench, master)
    b_tr, b_te = period_cagr(bench, master, master[0], SPLIT), period_cagr(bench, master, SPLIT, None)
    print(f"\n【ベンチマーク】通期CAGR {bm['CAGR%']}%  訓練期 {b_tr:.2f}%  検証期 {b_te:.2f}%  "
          f"最大DD {bm['最大DD%']}%\n")

    rows = []
    curves = {"ベンチマーク": bench}
    for name, lb, sk in FORMATIONS:
        eq, nhold = momentum_backtest(price_df, lb, sk)
        m = metrics(eq, master)
        tr, te = period_cagr(eq, master, master[0], SPLIT), period_cagr(eq, master, SPLIT, None)
        rows.append({"設定": name, "通期CAGR%": m["CAGR%"], "訓練期CAGR%": round(tr, 2),
                    "検証期CAGR%": round(te, 2), "対ベンチ倍率": round(float(eq[-1] / bench[-1]), 2),
                    "最大DD%": m["最大DD%"], "年率vol%": m["年率vol%"], "CAGR/vol": m["CAGR/vol"]})
        curves[name] = eq
        print(f"  {name:<26} 通期{m['CAGR%']:>6.2f}%  訓練{tr:>6.2f}%  検証{te:>6.2f}%  "
              f"対ベンチ{float(eq[-1]/bench[-1]):>6.2f}倍  最大DD{m['最大DD%']:>6.1f}%  CAGR/vol{m['CAGR/vol']:>5.2f}")

    df = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    print("\n【全設定比較】訓練・検証どちらもベンチマークを上回るものが本命")
    print(df.to_string(index=False))

    out = Path(__file__).parent / "data"
    df.to_csv(out / "momentum_classic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves, index=master).to_csv(out / "momentum_classic_curves.csv", encoding="utf-8-sig")

    ok = df[(df["訓練期CAGR%"] > b_tr) & (df["検証期CAGR%"] > b_te)].copy()
    if ok.empty:
        print("\n訓練・検証の両方でベンチマークを上回った設定はありませんでした。")
        winner_row = None
    else:
        ok["安定度"] = ok[["訓練期CAGR%", "検証期CAGR%"]].sub([b_tr, b_te], axis=1).min(axis=1)
        winner_row = ok.sort_values("安定度", ascending=False).iloc[0]
        print(f"\n=== 採用: {winner_row['設定']} ===")

    if winner_row is None:
        return
    wname, wlb, wsk = next((n, lb, sk) for n, lb, sk in FORMATIONS if n == winner_row["設定"])

    # ---- 現在時点の保有銘柄（最新のランキング）----
    print(f"\n=== 現在（最新データ）でこの条件を適用した場合の上位{N_TOP}銘柄 ===")
    price = price_df.to_numpy(dtype=np.float64)
    day = len(master) - 1
    p_recent = price[day - wsk] if wsk > 0 else price[day]
    p_old = price[day - wlb]
    valid = np.isfinite(p_recent) & np.isfinite(p_old) & (p_old > 0) & np.isfinite(price[day])
    score = np.where(valid, p_recent / np.where(p_old > 0, p_old, np.nan) - 1, -np.inf)
    order = np.argsort(-score)[:N_TOP]
    picks = []
    for i in order:
        sym = price_df.columns[i]
        name_jp, group, sector = TICKERS.get(sym, (sym, "?", "?"))
        picks.append({"コード": sym, "銘柄名": name_jp, "業種": sector,
                      "終値": round(float(price[day, i]), 1),
                      "モメンタム%": round(float(score[i]) * 100, 1)})
    pdf = pd.DataFrame(picks)
    print(pdf.to_string(index=False))
    pdf.to_csv(out / "momentum_classic_picks.csv", index=False, encoding="utf-8-sig")
    print(f"\ndata/momentum_classic_picks.csv に保存")


if __name__ == "__main__":
    main()
