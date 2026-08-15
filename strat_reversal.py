# -*- coding: utf-8 -*-
"""短期リバーサル（横断的な「売られすぎ」ランキング）を検証する。

これまでの検証で「売られすぎを買う」平均回帰効果が27年間一貫して機能する
ことは分かっている。ただし mechanical_overlay.py のRSIルールは
「RSIが閾値を下抜けたら買う」という**絶対的な閾値方式**なので、
シグナルが出ない日が多く、資金がベンチマークで待機する時間が長い。

ここでは同じ平均回帰効果を**横断的（クロスセクショナル）なランキング**に
置き換える。毎回必ずその日の「最も売られすぎている上位N銘柄」を買うので、
ポートフォリオは常に埋まり、平均回帰効果をフルに使える。

検証する売られすぎスコア（いずれも小さいほど「売られすぎ」= 買い）:
  RET     直近L営業日の騰落率（Lehmann 1990 / Jegadeesh 1990 の短期反転効果）
  RETVOL  上をボラティリティで割って正規化（変動の大きい銘柄への偏りを防ぐ）
  SMADEV  SMA(L)からの下方乖離率
  PCTB    ボリンジャー%B（SMA±2σ に対する位置。z-scoreの線形変換）
  RSI     RSI(L)の低い順（既存の絶対閾値ルールの直接的なランキング版）
  SECRET  業種平均を引いた相対騰落率（業種内リバーサル）

重要な設計上の注意:
  - 保有中の比率はリバランスまで**ドリフトさせる**（日次で均等に戻さない）。
    日次均等リバランスは、それ自体が平均回帰効果でリターンを水増しするため、
    リバーサル戦略の検証では自分に有利な数字が出てしまう。
  - 売買コストは往復0.2%を**片道回転率に比例**してリバランス日に差し引く。
    短期リバーサルは回転率が非常に高くなるので、コスト前後の両方を報告する。
  - 助走期間（先頭260営業日）はベンチマークを保有したものとして扱う。
    全設定で同じ扱いなので、設定間・対ベンチの比較が歪まない。

データスヌーピング対策として手順を固定する:
  1. 訓練期(1999-2012)のCAGR**だけ**でグリッド探索する
  2. 近傍（リバランス頻度違い）も同じくらい良い設定を選ぶ（一点突出を避ける）
  3. 選んだ1つを検証期(2013-2026)で**一度だけ**評価する

使い方: python strat_reversal.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mechanical_backtest import build_matrix, load_named
from momentum_classic import metrics, period_cagr
from tickers import TICKERS

SPLIT = pd.Timestamp("2012-10-03")
COST = 0.002          # 往復売買コスト（片道回転率に比例して控除）
N_TOP = 20            # グリッド探索中は固定
START_DAY = 260       # 助走期間（全設定共通。ここまではベンチマークを保有）
VOL_WIN = 60          # ボラティリティ正規化に使う窓

# (スコア種別, 表示名, 試すルックバック)
KINDS: list[tuple[str, str, tuple[int, ...]]] = [
    ("RET",    "短期リバーサル(騰落率)", (5, 10, 21, 63)),
    ("RETVOL", "ボラ調整リバーサル",     (5, 10, 21, 63)),
    ("SMADEV", "SMA乖離率",              (10, 20, 50)),
    ("PCTB",   "ボリンジャー%B",         (10, 20, 50)),
    ("RSI",    "RSIランキング",          (7, 14, 21)),
    ("SECRET", "業種内リバーサル",       (5, 10, 21, 63)),
]
FREQS = (5, 10, 21)               # リバランス間隔（営業日）: 週次 / 隔週 / 月次
N_TOP_SWEEP = (10, 20, 30, 50)    # 勝ち設定について訓練期のみで追加検討する保有銘柄数


def rsi_matrix(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """rule_optimize.rsi と同じ定義をDataFrame全体に一括適用する。"""
    d = price_df.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


def build_score(kind: str, lookback: int, price_df: pd.DataFrame,
                pct_df: pd.DataFrame, sector: pd.Series) -> np.ndarray:
    """売られすぎスコア行列を返す（小さいほど売られすぎ = 買い候補）。"""
    if kind == "RET":
        s = price_df / price_df.shift(lookback) - 1.0
    elif kind == "RETVOL":
        raw = price_df / price_df.shift(lookback) - 1.0
        vol = pct_df.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std() * np.sqrt(lookback)
        s = raw / vol.where(vol > 1e-6)
    elif kind == "SMADEV":
        ma = price_df.rolling(lookback, min_periods=lookback).mean()
        s = price_df / ma - 1.0
    elif kind == "PCTB":
        ma = price_df.rolling(lookback, min_periods=lookback).mean()
        sd = price_df.rolling(lookback, min_periods=lookback).std()
        s = (price_df - ma) / (2 * sd.where(sd > 1e-9))   # %B = 0.5 + s/2 の線形変換
    elif kind == "RSI":
        s = rsi_matrix(price_df, lookback)
    elif kind == "SECRET":
        raw = price_df / price_df.shift(lookback) - 1.0
        s = raw - raw.T.groupby(sector).transform("mean").T
    else:
        raise ValueError(kind)
    return s.to_numpy(dtype=np.float64)


def backtest(price: np.ndarray, pct: np.ndarray, bench_ret: np.ndarray,
             score: np.ndarray, n_top: int, freq: int,
             cost: float = COST, start_day: int = START_DAY
             ) -> tuple[np.ndarray, np.ndarray]:
    """スコア下位n_top銘柄を均等保有し、freq営業日ごとに入れ替える。

    戻り値: (資産曲線, リバランスごとの片道回転率)
    """
    n_days = price.shape[0]
    equity = np.empty(n_days)
    equity[0] = 1.0
    hold = np.array([], dtype=int)
    w = np.array([], dtype=np.float64)
    rebal = set(range(start_day, n_days, freq))
    turnovers: list[float] = []

    for day in range(1, n_days):
        if day < start_day:
            equity[day] = equity[day - 1] * (1 + bench_ret[day])
        elif len(hold):
            r = pct[day, hold]
            r = np.where(np.isfinite(r), r, 0.0)   # 欠損・上場廃止はリターン0で据え置き
            pr = float(w @ r)
            equity[day] = equity[day - 1] * (1 + pr)
            if 1 + pr > 1e-12:
                w = w * (1 + r) / (1 + pr)         # 比率はリバランスまでドリフトさせる
        else:
            equity[day] = equity[day - 1]

        if day in rebal:
            s = score[day]
            ok = np.isfinite(s) & np.isfinite(price[day]) & (price[day] > 0)
            cand = np.flatnonzero(ok)
            if len(cand) == 0:
                continue
            k = min(n_top, len(cand))
            sel = cand[np.argsort(s[cand], kind="stable")[:k]]
            new_w = np.full(k, 1.0 / k)

            old_map = dict(zip(hold.tolist(), w.tolist()))
            new_map = dict(zip(sel.tolist(), new_w.tolist()))
            diff = sum(abs(new_map.get(i, 0.0) - old_map.get(i, 0.0))
                       for i in set(old_map) | set(new_map))
            turn = diff / 2.0                       # 片道回転率
            equity[day] *= (1 - cost * turn)
            turnovers.append(turn)
            hold, w = sel, new_w

    return equity, np.array(turnovers)


def evaluate(equity: np.ndarray, master: pd.DatetimeIndex, bench: np.ndarray,
             turns: np.ndarray, freq: int) -> dict:
    m = metrics(equity, master)
    return {
        "通期CAGR%": m["CAGR%"],
        "訓練期CAGR%": round(period_cagr(equity, master, master[0], SPLIT), 2),
        "検証期CAGR%": round(period_cagr(equity, master, SPLIT, None), 2),
        "対ベンチ倍率": round(float(equity[-1] / bench[-1]), 2),
        "最大DD%": m["最大DD%"], "年率vol%": m["年率vol%"], "CAGR/vol": m["CAGR/vol"],
        "年回転率%": round(float(turns.mean()) * (250 / freq) * 100, 0) if len(turns) else 0.0,
    }


def main() -> None:
    print("=== データ読み込み ===")
    series = load_named()
    price_df = build_matrix(series)
    master = price_df.index
    print(f"日本株 {len(series)}銘柄 / {master[0].date()} 〜 {master[-1].date()} "
          f"({len(master):,}日)")

    pct_df = price_df.pct_change()
    price = price_df.to_numpy(dtype=np.float64)
    pct = pct_df.to_numpy(dtype=np.float64)
    bench_ret = pct_df.mean(axis=1).fillna(0).to_numpy()
    bench = np.cumprod(1 + bench_ret)
    bench = bench / bench[0]

    sector = pd.Series({s: TICKERS.get(s, (s, "?", "?"))[2] for s in price_df.columns})

    bm = metrics(bench, master)
    b_tr = period_cagr(bench, master, master[0], SPLIT)
    b_te = period_cagr(bench, master, SPLIT, None)
    print(f"\n【ベンチマーク】587銘柄均等分散バイアンドホールド")
    print(f"  通期CAGR {bm['CAGR%']}%  訓練期 {b_tr:.2f}%  検証期 {b_te:.2f}%  "
          f"最大DD {bm['最大DD%']}%  CAGR/vol {bm['CAGR/vol']}")
    print(f"  合格条件: 訓練期 > {b_tr:.2f}% かつ 検証期 > {b_te:.2f}%\n")

    # ---- 手順1: グリッド探索（訓練期CAGRだけを見る）----
    print(f"=== 手順1: グリッド探索（保有{N_TOP}銘柄固定・コスト込み）===")
    rows = []
    curves: dict[str, np.ndarray] = {"ベンチマーク": bench}
    for kind, label, lookbacks in KINDS:
        for lb in lookbacks:
            score = build_score(kind, lb, price_df, pct_df, sector)
            for freq in FREQS:
                eq, turns = backtest(price, pct, bench_ret, score, N_TOP, freq)
                eq0, _ = backtest(price, pct, bench_ret, score, N_TOP, freq, cost=0.0)
                name = f"{label}{lb}日/{freq}日毎"
                r = {"種別": kind, "表示名": label, "ルックバック": lb,
                     "リバランス": freq, "保有数": N_TOP, "設定": name}
                r.update(evaluate(eq, master, bench, turns, freq))
                r["コスト前_訓練期CAGR%"] = round(period_cagr(eq0, master, master[0], SPLIT), 2)
                r["コスト前_通期CAGR%"] = metrics(eq0, master)["CAGR%"]
                rows.append(r)
                curves[name] = eq
            del score

    grid = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 250)
    show = ["設定", "訓練期CAGR%", "コスト前_訓練期CAGR%", "年回転率%", "最大DD%", "CAGR/vol"]
    print("訓練期CAGR 上位12（※検証期の数字はまだ一切見ていない）")
    print(grid.sort_values("訓練期CAGR%", ascending=False).head(12)[show].to_string(index=False))

    # ---- 手順2: 近傍チェック（リバランス頻度を変えても優位が残るか）----
    print("\n=== 手順2: 近傍チェック（同じスコア・同じルックバックで頻度を変える）===")
    top = grid.sort_values("訓練期CAGR%", ascending=False).head(8)
    best = None
    for _, r in top.iterrows():
        fam = grid[(grid["種別"] == r["種別"]) & (grid["ルックバック"] == r["ルックバック"])]
        worst = float(fam["訓練期CAGR%"].min())
        print(f"  {r['表示名']}{r['ルックバック']}日: 頻度違いの訓練期CAGR = "
              f"{sorted(np.round(fam['訓練期CAGR%'].to_numpy(), 2).tolist())} → 最小 {worst:.2f}%")
        if best is None or worst > best[1]:
            best = (r, worst)
    chosen, fam_min = best
    print(f"\n→ スコア確定: {chosen['表示名']}{chosen['ルックバック']}日 "
          f"（頻度を変えた時の最小訓練期CAGRが {fam_min:.2f}% と最も安定）")

    # ---- 手順3: 保有銘柄数の検討（これも訓練期のみ）----
    print(f"\n=== 手順3: 保有銘柄数の検討（訓練期のみ・リバランス頻度は近傍最良を採用）===")
    score = build_score(chosen["種別"], int(chosen["ルックバック"]), price_df, pct_df, sector)
    fam = grid[(grid["種別"] == chosen["種別"]) & (grid["ルックバック"] == chosen["ルックバック"])]
    w_freq = int(fam.sort_values("訓練期CAGR%", ascending=False).iloc[0]["リバランス"])
    sweep = []
    for nt in N_TOP_SWEEP:
        eq, turns = backtest(price, pct, bench_ret, score, nt, w_freq)
        tr = period_cagr(eq, master, master[0], SPLIT)
        sweep.append((nt, tr, metrics(eq, master)["CAGR/vol"]))
        print(f"  保有{nt:>2}銘柄: 訓練期CAGR {tr:>6.2f}%  CAGR/vol {sweep[-1][2]:>5.2f}")
    w_ntop = max(sweep, key=lambda x: x[1])[0]
    print(f"\n→ 最終確定: {chosen['表示名']}{chosen['ルックバック']}日 / "
          f"{w_freq}日毎リバランス / {w_ntop}銘柄保有")
    print("   （ここまで検証期の数字は一切参照していない）")

    # ---- 手順4: 検証期で一度だけ評価 ----
    eq, turns = backtest(price, pct, bench_ret, score, w_ntop, w_freq)
    eq0, turns0 = backtest(price, pct, bench_ret, score, w_ntop, w_freq, cost=0.0)
    final_name = (f"【採用】{chosen['表示名']}{chosen['ルックバック']}日/"
                  f"{w_freq}日毎/{w_ntop}銘柄")
    fin = evaluate(eq, master, bench, turns, w_freq)
    fin0 = evaluate(eq0, master, bench, turns0, w_freq)
    curves[final_name] = eq

    print(f"\n=== 手順4: 検証期(2012-10-03〜)での成績 ※一度だけの評価 ===")
    print(f"  【コスト後】通期{fin['通期CAGR%']:>6.2f}%  訓練期{fin['訓練期CAGR%']:>6.2f}%  "
          f"検証期{fin['検証期CAGR%']:>6.2f}%  最大DD{fin['最大DD%']:>6.1f}%  "
          f"CAGR/vol {fin['CAGR/vol']:>5.2f}  対ベンチ{fin['対ベンチ倍率']:>7.2f}倍  "
          f"年回転率{fin['年回転率%']:>5.0f}%")
    print(f"  【コスト前】通期{fin0['通期CAGR%']:>6.2f}%  訓練期{fin0['訓練期CAGR%']:>6.2f}%  "
          f"検証期{fin0['検証期CAGR%']:>6.2f}%  最大DD{fin0['最大DD%']:>6.1f}%  "
          f"CAGR/vol {fin0['CAGR/vol']:>5.2f}")
    print(f"  ベンチマーク 通期{bm['CAGR%']:>6.2f}%  訓練期{b_tr:>6.2f}%  検証期{b_te:>6.2f}%  "
          f"最大DD{bm['最大DD%']:>6.1f}%  CAGR/vol {bm['CAGR/vol']:>5.2f}")

    passed = fin["訓練期CAGR%"] > b_tr and fin["検証期CAGR%"] > b_te
    strong = passed and fin["CAGR/vol"] >= 0.80
    verdict = "強い合格" if strong else ("合格" if passed else "不合格")
    print(f"\n  判定: {verdict} "
          f"(訓練 {fin['訓練期CAGR%']:.2f} vs {b_tr:.2f} / "
          f"検証 {fin['検証期CAGR%']:.2f} vs {b_te:.2f} / "
          f"CAGR/vol {fin['CAGR/vol']:.2f} vs 0.80)")

    # ---- 保存 ----
    out = Path(__file__).parent / "data"
    final_row = {"種別": chosen["種別"], "表示名": chosen["表示名"],
                 "ルックバック": chosen["ルックバック"], "リバランス": w_freq,
                 "保有数": w_ntop, "設定": final_name, **fin,
                 "コスト前_訓練期CAGR%": fin0["訓練期CAGR%"],
                 "コスト前_通期CAGR%": fin0["通期CAGR%"]}
    bench_row = {"設定": "ベンチマーク(587銘柄均等分散)", "通期CAGR%": bm["CAGR%"],
                 "訓練期CAGR%": round(b_tr, 2), "検証期CAGR%": round(b_te, 2),
                 "対ベンチ倍率": 1.0, "最大DD%": bm["最大DD%"], "年率vol%": bm["年率vol%"],
                 "CAGR/vol": bm["CAGR/vol"], "年回転率%": 0.0}
    res = pd.concat([pd.DataFrame([bench_row, final_row]), grid], ignore_index=True)
    res.to_csv(out / "strat_reversal.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves, index=master).to_csv(
        out / "strat_reversal_curves.csv", encoding="utf-8-sig")

    print("\n【全設定一覧（訓練期CAGR降順）】※検証期は参考表示、選択には未使用")
    cols = ["設定", "訓練期CAGR%", "検証期CAGR%", "通期CAGR%", "対ベンチ倍率",
            "最大DD%", "CAGR/vol", "年回転率%"]
    print(grid.sort_values("訓練期CAGR%", ascending=False)[cols].to_string(index=False))

    # ---- 現在（最新データ）での推奨銘柄 ----
    print(f"\n=== 現在（{master[-1].date()}）の推奨銘柄 上位{w_ntop} ===")
    day = len(master) - 1
    s = score[day]
    ok = np.isfinite(s) & np.isfinite(price[day]) & (price[day] > 0)
    cand = np.flatnonzero(ok)
    sel = cand[np.argsort(s[cand], kind="stable")[:w_ntop]]
    picks = []
    for rank, i in enumerate(sel, 1):
        sym = price_df.columns[i]
        name_jp, _, sec = TICKERS.get(sym, (sym, "?", "?"))
        picks.append({"順位": rank, "コード": sym, "銘柄名": name_jp, "業種": sec,
                      "終値": round(float(price[day, i]), 1),
                      "スコア": round(float(s[i]), 4)})
    pdf = pd.DataFrame(picks)
    print(pdf.to_string(index=False))
    pdf.to_csv(out / "strat_reversal_picks.csv", index=False, encoding="utf-8-sig")
    print(f"\ndata/strat_reversal.csv / strat_reversal_curves.csv / "
          f"strat_reversal_picks.csv に保存")


if __name__ == "__main__":
    main()
