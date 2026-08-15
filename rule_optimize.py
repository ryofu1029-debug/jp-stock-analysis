# -*- coding: utf-8 -*-
"""生き残った売られすぎルール（急落 / RSI）のパラメータを詰める。

pattern_verify.py で、31のチャートパターンは全滅し「15日で15%急落」と
「RSI30以下」の2つだけが4条件すべてで優位性を保った。ここではその2つの
しきい値・期間・保有日数を最適化する。

過剰適合を避けるため手順を固定する:
  1. 訓練期間（前半）**だけ**でグリッド探索する
  2. 上位のパラメータについて、近傍が同じくらい良いか（表面が滑らかか）を確認する
     → 一点だけ突出しているものは偶然なので採らない
  3. 選んだ1つを、検証期間（後半）で**一度だけ**評価する
  4. 売買コスト往復0.2%を引いた数字も出す

※ 銘柄リストは現存銘柄のみなので生存者バイアスが残る。数値は上限値として読む。

使い方: python rule_optimize.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

LONG_DIR = Path(os.environ.get(
    "JPSA_LONG_DIR",
    Path(os.environ.get("LOCALAPPDATA", ".")) / "jp-stock-analysis" / "long"))

SPLIT = pd.Timestamp("2012-10-03")   # pattern_verify.py と同じ分割点
MIN_ROWS = 750
MIN_GAP = 20            # 重なったシグナルを間引く間隔（営業日）
COST = 0.002            # 往復売買コスト
MIN_TRAIN_N = 300       # 訓練期間でこれ未満の検出数のパラメータは採用しない

DROPS = (0.10, 0.12, 0.15, 0.20, 0.25)      # 急落と見なす下落率
WINDOWS = (5, 10, 15, 20, 30)               # 何営業日での下落を見るか
RSI_THRS = (20, 25, 30, 35)                 # RSIのしきい値
RSI_PERIODS = (7, 14, 21)                   # RSIの期間
HOLDS = (5, 10, 20, 40, 60)                 # 保有営業日数


def rsi(c: np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(c)
    d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    return (100 - 100 / (1 + ag / al)).to_numpy()


def dedup(idx: np.ndarray, gap: int = MIN_GAP) -> np.ndarray:
    """近接したシグナルを間引く（同じ下げを何度も数えないため）。"""
    keep, last = [], -10**9
    for i in idx:
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def sanitize(c: np.ndarray) -> np.ndarray:
    """Yahoo由来の異常値（前後から桁違いに外れた値）を落とすマスクを返す。

    ＳＢＩ新生銀行(8303.T)に1株53,969,555,456円という値が実在し、平均リターンを
    破壊していた。前後11営業日の中央値から5倍以上外れた点は価格ではないと見なす。
    """
    s = pd.Series(c)
    med = s.rolling(11, center=True, min_periods=3).median()
    ok = np.isfinite(c) & (c > 0) & np.isfinite(med.to_numpy())
    ok &= (c <= med.to_numpy() * 5) & (c >= med.to_numpy() / 5)
    return ok


def longest_continuous(c: np.ndarray) -> slice:
    """日次で5倍以上跳ぶ不連続点で系列を切り、最も長い区間を返す。

    スパイク除去では取れない「水準の断絶」に対応する。例: 日本航空(9201.T)は
    2010年に上場廃止・2012年に再上場しており、価格系列が繋がっていない。
    """
    r = c[1:] / c[:-1]
    cuts = [0] + (np.flatnonzero((r > 5) | (r < 0.2)) + 1).tolist() + [len(c)]
    best = max(zip(cuts[:-1], cuts[1:]), key=lambda ab: ab[1] - ab[0])
    return slice(*best)


def load() -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
    out, dropped, split_syms = [], 0, 0
    for f in sorted(LONG_DIR.glob("*.T.csv")):
        d = pd.read_csv(f, usecols=["Date", "Close"], parse_dates=["Date"]).dropna()
        if len(d) < MIN_ROWS:
            continue
        c = d["Close"].to_numpy(dtype=float)
        ok = sanitize(c)
        if not ok.all():
            dropped += int((~ok).sum())
            c, d = c[ok], d[ok]
        if len(c) < MIN_ROWS:
            continue
        sl = longest_continuous(c)
        if sl.stop - sl.start < len(c):
            split_syms += 1
            c, d = c[sl], d.iloc[sl]
        if len(c) < MIN_ROWS:
            continue
        out.append((c, d["Date"].to_numpy()))
    return out, dropped, split_syms


def collect(series: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, dict]:
    """各パラメータのシグナル後リターンを、訓練/検証に分けて集める。"""
    res: dict[tuple, dict[str, list[float]]] = {}
    base: dict[tuple[str, int], list[float]] = {}

    for c, dates in series:
        n = len(c)
        is_train = dates < SPLIT.to_datetime64()
        fwd = {h: np.full(n, np.nan) for h in HOLDS}
        for h in HOLDS:
            fwd[h][:n - h] = c[h:] / c[:n - h] - 1.0

        for h in HOLDS:
            for era, m in (("訓練", is_train), ("検証", ~is_train)):
                v = fwd[h][m & np.isfinite(fwd[h])]
                base.setdefault((era, h), []).append(v[::5])

        # --- 急落ルール ---
        for w in WINDOWS:
            ratio = np.full(n, np.nan)
            ratio[w:] = c[w:] / c[:-w] - 1.0
            for dr in DROPS:
                sig = dedup(np.flatnonzero(ratio <= -dr))
                if len(sig) == 0:
                    continue
                for h in HOLDS:
                    for era, m in (("訓練", is_train), ("検証", ~is_train)):
                        s = sig[m[sig] & np.isfinite(fwd[h][sig])]
                        if len(s):
                            res.setdefault(("急落", w, dr, h), {}).setdefault(era, []).extend(
                                fwd[h][s].tolist())

        # --- RSIルール ---
        for p in RSI_PERIODS:
            r = rsi(c, p)
            for thr in RSI_THRS:
                cross = np.flatnonzero((r <= thr) & (np.roll(r, 1) > thr))
                sig = dedup(cross[cross > p])
                if len(sig) == 0:
                    continue
                for h in HOLDS:
                    for era, m in (("訓練", is_train), ("検証", ~is_train)):
                        s = sig[m[sig] & np.isfinite(fwd[h][sig])]
                        if len(s):
                            res.setdefault(("RSI", p, thr, h), {}).setdefault(era, []).extend(
                                fwd[h][s].tolist())

    # 平均は外れ値に弱いので、順位付けには中央値を使う（平均も参考に残す）
    stats = {}
    for k, v in base.items():
        a = np.concatenate(v)
        stats[k] = (float(a.mean()), float(np.median(a)), float((a > 0).mean()))
    return res, stats


def table(res: dict, base: dict, era: str) -> pd.DataFrame:
    rows = []
    for key, d in res.items():
        a = np.array(d.get(era, []))
        if len(a) == 0:
            continue
        kind, p1, p2, h = key
        bmean, bmed, bwin = base[(era, h)]
        med = float(np.median(a))
        rows.append({
            "種類": kind,
            "期間": p1,
            "しきい値": p2 if kind == "RSI" else f"{p2:.0%}",
            "保有": h,
            "件数": len(a),
            "中央値%": round(med * 100, 2),
            "超過%": round((med - bmed) * 100, 2),          # 中央値ベース（順位付けに使う）
            "コスト後%": round((med - bmed - COST) * 100, 2),
            "勝率%": round(float((a > 0).mean()) * 100, 1),
            "勝率差": round((float((a > 0).mean()) - bwin) * 100, 1),
            "平均超過%": round((a.mean() - bmean) * 100, 2),  # 参考（外れ値に弱い）
        })
    return pd.DataFrame(rows)


def main() -> None:
    series, dropped, split_syms = load()
    print(f"=== 日本株 {len(series)}銘柄 / 訓練:〜{SPLIT.date()} / 検証:{SPLIT.date()}〜 ===")
    print(f"異常値として除外した足: {dropped:,}本 / 不連続で切り詰めた銘柄: {split_syms}社")
    res, base = collect(series)
    print("ベースライン(全日・中央値%): " + " / ".join(
        f"{e}{h}日 {base[(e, h)][1]*100:+.2f}" for e in ("訓練", "検証") for h in (5, 20)))

    tr = table(res, base, "訓練")
    tr = tr[tr["件数"] >= MIN_TRAIN_N].sort_values("超過%", ascending=False)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 250)

    print("\n【手順1】訓練期間（1999-2012）のみでのグリッド探索・上位12")
    print(tr.head(12).to_string(index=False))

    # --- 手順2: 近傍が同じくらい良いか（表面の滑らかさ）を見る ---
    print("\n【手順2】上位候補の近傍チェック（保有日数を変えても優位性が残るか）")
    best_rows = []
    for _, r in tr.head(6).iterrows():
        kind, p1 = r["種類"], r["期間"]
        p2 = r["しきい値"] if kind == "RSI" else float(str(r["しきい値"]).rstrip("%")) / 100
        neigh = tr[(tr["種類"] == kind) & (tr["期間"] == p1) & (tr["しきい値"] == r["しきい値"])]
        vals = neigh["超過%"].to_numpy()
        print(f"  {kind} 期間{p1} しきい値{r['しきい値']}: "
              f"保有日数を変えた時の超過% = {np.round(vals, 2).tolist()} "
              f"→ 最小 {vals.min():.2f}")
        best_rows.append((r, vals.min()))

    # 近傍の最小値が最も高いものを採用（一点突出を避ける）
    chosen, worst = max(best_rows, key=lambda x: x[1])
    print(f"\n→ 採用: {chosen['種類']} 期間{chosen['期間']} "
          f"しきい値{chosen['しきい値']} 保有{chosen['保有']}日"
          f"（近傍の最小超過が {worst:.2f}% と最も安定）")

    # --- 手順3: 検証期間で一度だけ評価 ---
    te = table(res, base, "検証")
    key = ((chosen["種類"] == te["種類"]) & (chosen["期間"] == te["期間"])
           & (chosen["しきい値"] == te["しきい値"]) & (chosen["保有"] == te["保有"]))
    print("\n【手順3】検証期間（2013-2026）での成績 ※ここは一度だけ評価")
    print(te[key].to_string(index=False))

    print("\n参考: 検証期間での上位10（探索には使っていない）")
    print(te[te["件数"] >= MIN_TRAIN_N].sort_values("超過%", ascending=False)
          .head(10).to_string(index=False))

    out = Path(__file__).parent / "data"
    tr.to_csv(out / "rule_train.csv", index=False, encoding="utf-8-sig")
    te.to_csv(out / "rule_test.csv", index=False, encoding="utf-8-sig")
    print(f"\ndata/rule_train.csv, data/rule_test.csv に保存")


if __name__ == "__main__":
    main()
