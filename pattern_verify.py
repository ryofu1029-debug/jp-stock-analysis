# -*- coding: utf-8 -*-
"""上場来データで、パターンが「どの相場でも」機能するかを検証する。

pattern_stats.py は2021-2026年の上昇相場1レジームしか見ていないため、
「急落を買う」が効いて当然という弱点があった。ここでは長期データを使い、
次の3つを分けて評価する:

  1. レジーム別   … 市場平均が200日線の上か下かで「上げ相場/下げ相場」に分割。
                    下げ相場でも効くかが本当の試金石。
  2. 前半/後半    … 前半で見つけた優位性が後半でも残るか（多重比較で
                    たまたま良く見えただけのものを落とす）。
  3. 既存指標と比較 … RSI売られすぎ・BB下限タッチ・急落ルールを同じ土俵に載せ、
                    チャートパターンがそれらを超えるかを見る。

データは fetch_long.py が %LOCALAPPDATA%\\jp-stock-analysis\\long\\ に置いたもの。

使い方: python pattern_verify.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pattern_stats import BEARISH, detect

LONG_DIR = Path(os.environ.get(
    "JPSA_LONG_DIR",
    Path(os.environ.get("LOCALAPPDATA", ".")) / "jp-stock-analysis" / "long"))

HORIZON = 20        # 何営業日後を評価するか
BASE_STEP = 5       # ベースライン用に何日おきにサンプルするか
MIN_ROWS = 750      # 3年未満の銘柄は使わない
MIN_HITS = 100      # これ未満の検出数は結論を出さない
SPLIT_QUANTILE = 0.5  # 前半/後半の分割点（日付の中央値）


def load_jp() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(LONG_DIR.glob("*.T.csv")):
        d = pd.read_csv(f, usecols=["Date", "Close"], parse_dates=["Date"])
        d = d.dropna().set_index("Date")
        if len(d) >= MIN_ROWS:
            out[f.stem] = d
    return out


def market_regime(data: dict[str, pd.DataFrame]) -> pd.Series:
    """全銘柄の平均日次リターンから等ウェイト指数を作り、200日線の上下で判定する。"""
    rets = pd.DataFrame({s: d["Close"].pct_change() for s, d in data.items()})
    idx = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    return idx > idx.rolling(200, min_periods=100).mean()


def indicator_signals(c: np.ndarray) -> list[tuple[str, int]]:
    """比較用のベンチマークルール（チャートパターン以外の売られすぎ指標）。"""
    s = pd.Series(c)
    out = []

    delta = s.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rsi = (100 - 100 / (1 + ag / al)).to_numpy()

    mid = s.rolling(20).mean()
    sd = s.rolling(20).std()
    bb_lo = (mid - 2 * sd).to_numpy()
    bb_up = (mid + 2 * sd).to_numpy()

    for i in range(20, len(c)):
        if rsi[i] <= 30 and rsi[i - 1] > 30:
            out.append(("[指標]RSI30以下", i))
        if rsi[i] >= 70 and rsi[i - 1] < 70:
            out.append(("[指標]RSI70以上", i))
        if not np.isnan(bb_lo[i]) and c[i] <= bb_lo[i] and c[i - 1] > bb_lo[i - 1]:
            out.append(("[指標]BB下限タッチ", i))
        if not np.isnan(bb_up[i]) and c[i] >= bb_up[i] and c[i - 1] < bb_up[i - 1]:
            out.append(("[指標]BB上限タッチ", i))
        if i >= 15 and c[i] / c[i - 15] - 1 <= -0.15:
            out.append(("[指標]15日で15%急落", i))
        if i >= 15 and c[i] / c[i - 15] - 1 >= 0.15:
            out.append(("[指標]15日で15%急騰", i))
    return out


BEARISH_RULES = BEARISH | {"[指標]RSI70以上", "[指標]BB上限タッチ", "[指標]15日で15%急騰"}


def main() -> None:
    if not LONG_DIR.exists() or not any(LONG_DIR.glob("*.T.csv")):
        raise SystemExit(f"{LONG_DIR} にデータがありません。先に fetch_long.py を実行してください。")

    data = load_jp()
    print(f"=== 日本株 {len(data)}銘柄 / 延べ {sum(len(d) for d in data.values()):,}営業日 ===")
    regime = market_regime(data)
    split = regime.index[int(len(regime) * SPLIT_QUANTILE)]
    print(f"期間: {regime.index[0].date()} 〜 {regime.index[-1].date()}")
    print(f"前半/後半の分割点: {split.date()}")
    print(f"上げ相場の割合: {regime.mean() * 100:.1f}%\n")

    # (パターン, レジーム, 時期) ごとにリターンを溜める
    hits: dict[tuple[str, str, str], list[float]] = {}
    basev: dict[tuple[str, str], list[float]] = {}

    for sym, d in data.items():
        c = d["Close"].to_numpy(dtype=float)
        dates = d.index
        reg = regime.reindex(dates).to_numpy()

        def tag(i: int) -> tuple[str, str] | None:
            if reg[i] is None or (isinstance(reg[i], float) and np.isnan(reg[i])):
                return None
            return ("上げ相場" if reg[i] else "下げ相場",
                    "前半" if dates[i] < split else "後半")

        for i in range(0, len(c) - HORIZON, BASE_STEP):
            t = tag(i)
            if t:
                basev.setdefault(t, []).append(c[i + HORIZON] / c[i] - 1.0)

        last: dict[str, int] = {}
        for name, e in list(detect(c)) + indicator_signals(c):
            if e + HORIZON >= len(c) or e - last.get(name, -10**9) < 20:
                continue
            last[name] = e
            t = tag(e)
            if t:
                hits.setdefault((name, *t), []).append(c[e + HORIZON] / c[e] - 1.0)

    base_up = {k: float(np.mean(np.array(v) > 0)) for k, v in basev.items()}
    print("【レジーム別ベースライン】20営業日後に上昇していた確率")
    for k in sorted(base_up):
        print(f"  {k[0]}・{k[1]}: {base_up[k]*100:5.1f}%  (n={len(basev[k]):,})")

    names = sorted({k[0] for k in hits})
    rows = []
    for name in names:
        bear = name in BEARISH_RULES
        r = {"ルール": name, "方向": "下落" if bear else "上昇"}
        total = 0
        for regn in ("上げ相場", "下げ相場"):
            for era in ("前半", "後半"):
                a = np.array(hits.get((name, regn, era), []))
                col = f"{regn[:2]}{era}"
                if len(a) < MIN_HITS:
                    r[col] = np.nan
                    continue
                hit = float((a < 0).mean()) if bear else float((a > 0).mean())
                b = base_up[(regn, era)]
                bh = 1.0 - b if bear else b
                r[col] = round((hit - bh) * 100, 1)
                total += len(a)
        r["検出数"] = total
        vals = [r[c] for c in ("上げ前半", "上げ後半", "下げ前半", "下げ後半")
                if isinstance(r.get(c), float) and not np.isnan(r.get(c))]
        r["有効セル"] = len(vals)
        r["最小差"] = round(min(vals), 1) if vals else np.nan
        r["平均差"] = round(float(np.mean(vals)), 1) if vals else np.nan
        rows.append(r)

    df = pd.DataFrame(rows)
    cols = ["ルール", "方向", "検出数", "上げ前半", "上げ後半", "下げ前半", "下げ後半",
            "有効セル", "最小差", "平均差"]
    df = df[cols].sort_values("最小差", ascending=False)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 250)
    print("\n【4条件すべてでの優位性】数値=同条件ベースラインとの差(ポイント)")
    print("※ 最小差がプラス = 上げ相場でも下げ相場でも、前半でも後半でも効いた")
    print(df.to_string(index=False))

    out = Path(__file__).parent / "data" / "pattern_verify.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{out.name} に保存")

    good = df[(df["有効セル"] == 4) & (df["最小差"] > 0)]
    print(f"\n=== 4条件すべてでプラスだったルール: {len(good)}件 / {len(df)}件中 ===")
    if len(good):
        print(good[["ルール", "検出数", "最小差", "平均差"]].to_string(index=False))
    else:
        print("なし。どのルールも、相場環境か時期のどちらかで優位性が消える。")


if __name__ == "__main__":
    main()
