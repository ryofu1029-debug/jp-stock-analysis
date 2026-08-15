# -*- coding: utf-8 -*-
"""名前付きチャートパターンの「その後」を全銘柄×5年で数える検証スクリプト。

ダブルボトム等の古典的パターンを実際に検出し、検出後の値動きを集計する。
「上昇確率◯%」は単体では意味がない（相場全体が上げていれば何でも上がる）ので、
**同じ母集団の無条件の上昇確率（ベースライン）との差**で評価する。

先読みバイアスを避けるため、エントリー日は必ず「その時点で確定していた日」にする:
  - ジグザグの極値は、そこから pct 逆行して初めて確定する → 確定日を使う
  - 反転型はネックライン突破日を使う（教科書どおりの確認）

使い方:
    python pattern_stats.py            # 全銘柄
    python pattern_stats.py --jp       # 日本株のみ
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ZIGZAG_PCT = 0.05      # 極値と認めるのに必要な逆行率
LEVEL_TOL = 0.04       # 「同じ高さ」と見なす許容差
HORIZONS = (5, 10, 20)  # 何営業日後を見るか
NECK_LIMIT = 60        # ネックライン突破を待つ上限（営業日）
MIN_GAP = 20           # 同一銘柄・同一パターンの重複検出を間引く間隔

# パターンが予告する方向。弱気パターンは「下落してこそ的中」なので、
# 上昇確率をそのまま勝率にすると評価が逆になる。
BEARISH = {
    "ダブルトップ", "トリプルトップ", "ヘッド&ショルダートップ", "逆V字型トップ",
    "逆N字型", "ボックスダウン", "下降フラッグ", "下降ウェッジ", "下降ペナント",
    "下降逆ペナント", "下降トライアングル",
}


@dataclass
class Pivot:
    i: int        # 極値が起きた位置
    v: float      # 極値の価格
    kind: str     # 'H' or 'L'
    conf: int     # その極値が確定した位置（先読み防止用）


def zigzag(c: np.ndarray, pct: float = ZIGZAG_PCT) -> list[Pivot]:
    """しきい値ジグザグ。極値と「確定した日」をセットで返す。"""
    if len(c) < 3:
        return []
    piv: list[Pivot] = []
    d = 1 if c[1] >= c[0] else -1
    ei, ev = 0, float(c[0])
    for i in range(1, len(c)):
        v = float(c[i])
        if d == 1:
            if v > ev:
                ei, ev = i, v
            elif v <= ev * (1 - pct):
                piv.append(Pivot(ei, ev, "H", i))
                d, ei, ev = -1, i, v
        else:
            if v < ev:
                ei, ev = i, v
            elif v >= ev * (1 + pct):
                piv.append(Pivot(ei, ev, "L", i))
                d, ei, ev = 1, i, v
    return piv


def _close(a: float, b: float, tol: float = LEVEL_TOL) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def _neck_break(c: np.ndarray, start: int, level: float, up: bool) -> int | None:
    """start以降で level を突破した最初の位置。見つからなければ None。"""
    end = min(len(c), start + NECK_LIMIT)
    for i in range(start, end):
        if (up and c[i] > level) or (not up and c[i] < level):
            return i
    return None


def detect(c: np.ndarray) -> list[tuple[str, int]]:
    """(パターン名, エントリー位置) のリストを返す。"""
    piv = zigzag(c)
    out: list[tuple[str, int]] = []
    n = len(piv)

    for j in range(n):
        p = piv[j]

        # ---- 反転型: 3ピボット（ダブルトップ / ダブルボトム / N字 / 逆N字）----
        if j >= 2:
            a, b = piv[j - 2], piv[j - 1]
            if a.kind == "L" and b.kind == "H" and p.kind == "L":
                if _close(a.v, p.v):                       # 安値2つが同水準
                    e = _neck_break(c, p.conf, b.v, up=True)
                    if e is not None:
                        out.append(("ダブルボトム", e))
                elif p.v > a.v * (1 + LEVEL_TOL):          # 切り上がった安値
                    e = _neck_break(c, p.conf, b.v, up=True)
                    if e is not None:
                        out.append(("N字型", e))
            if a.kind == "H" and b.kind == "L" and p.kind == "H":
                if _close(a.v, p.v):
                    e = _neck_break(c, p.conf, b.v, up=False)
                    if e is not None:
                        out.append(("ダブルトップ", e))
                elif p.v < a.v * (1 - LEVEL_TOL):          # 切り下がった高値
                    e = _neck_break(c, p.conf, b.v, up=False)
                    if e is not None:
                        out.append(("逆N字型", e))

        # ---- 反転型: 5ピボット（トリプル / ヘッド&ショルダー）----
        if j >= 4:
            q = piv[j - 4:j + 1]
            kinds = "".join(x.kind for x in q)
            if kinds == "LHLHL":
                lows = [q[0].v, q[2].v, q[4].v]
                neck = max(q[1].v, q[3].v)
                if _close(lows[0], lows[1]) and _close(lows[1], lows[2]):
                    e = _neck_break(c, p.conf, neck, up=True)
                    if e is not None:
                        out.append(("トリプルボトム", e))
                elif lows[1] < min(lows[0], lows[2]) * (1 - LEVEL_TOL) and _close(lows[0], lows[2]):
                    e = _neck_break(c, p.conf, neck, up=True)
                    if e is not None:
                        out.append(("ヘッド&ショルダーボトム", e))
            if kinds == "HLHLH":
                highs = [q[0].v, q[2].v, q[4].v]
                neck = min(q[1].v, q[3].v)
                if _close(highs[0], highs[1]) and _close(highs[1], highs[2]):
                    e = _neck_break(c, p.conf, neck, up=False)
                    if e is not None:
                        out.append(("トリプルトップ", e))
                elif highs[1] > max(highs[0], highs[2]) * (1 + LEVEL_TOL) and _close(highs[0], highs[2]):
                    e = _neck_break(c, p.conf, neck, up=False)
                    if e is not None:
                        out.append(("ヘッド&ショルダートップ", e))

        # ---- 反転型: V字 / 逆V字（急落→急騰、またはその逆）----
        if j >= 1 and piv[j - 1].kind != p.kind:
            span = p.i - piv[j - 1].i
            move = abs(p.v / piv[j - 1].v - 1)
            if span <= 15 and move >= 0.15:   # 15営業日以内に15%以上の一直線
                name = "V字型ボトム" if p.kind == "L" else "逆V字型トップ"
                out.append((name, p.conf))

    out.extend(_detect_consolidation(c))
    return out


def _fit_slope(idx: np.ndarray, val: np.ndarray) -> float:
    """価格で正規化した傾き（1営業日あたりの変化率）。"""
    if len(idx) < 2:
        return 0.0
    s = np.polyfit(idx, val, 1)[0]
    return float(s / max(1e-9, val.mean()))


def _detect_consolidation(c: np.ndarray, pole: int = 20, win: int = 20) -> list[tuple[str, int]]:
    """ポール（急騰・急落）の後の保ち合いを、上下トレンドラインの傾きで分類する。"""
    out = []
    flat, step = 0.0015, 10   # 傾きが±0.15%/日以内なら「水平」とみなす
    for e in range(pole + win, len(c), step):
        p0, p1 = c[e - win - pole], c[e - win]
        pole_ret = p1 / p0 - 1
        if abs(pole_ret) < 0.15:
            continue
        seg = c[e - win:e]
        x = np.arange(win)
        # 上側=各点の直近高値、下側=直近安値のラインで近似
        up = pd.Series(seg).rolling(3, min_periods=1).max().to_numpy()
        lo = pd.Series(seg).rolling(3, min_periods=1).min().to_numpy()
        su, sl = _fit_slope(x, up), _fit_slope(x, lo)
        rng = (seg.max() - seg.min()) / seg.mean()
        if rng > 0.35:            # 保ち合いと呼べないほど動いていれば除外
            continue
        rising = pole_ret > 0
        pre = "上昇" if rising else "下降"
        if abs(su) < flat and abs(sl) < flat:
            name = "ボックスアップ" if rising else "ボックスダウン"
        elif su < -flat and sl > flat:
            name = f"{pre}ペナント"
        elif su > flat and sl < -flat:
            name = f"{pre}逆ペナント"
        elif abs(su) < flat and sl > flat:
            name = f"{pre}トライアングル"
        elif su < -flat and abs(sl) < flat:
            name = f"{pre}トライアングル"
        elif (su > flat and sl > flat) or (su < -flat and sl < -flat):
            same_dir = (su > 0) == rising
            name = f"{pre}ウェッジ" if not same_dir else f"{pre}フラッグ"
        else:
            continue
        out.append((name, e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jp", action="store_true", help="日本株(.T)のみ")
    args = ap.parse_args()

    files = sorted(PROCESSED_DIR.glob("*.T.csv" if args.jp else "*.csv"))
    print(f"=== 対象 {len(files)}銘柄 / ジグザグ{ZIGZAG_PCT:.0%} / 同水準許容{LEVEL_TOL:.0%} ===")

    hits: dict[str, list[list[float]]] = {}
    base: list[list[float]] = [[] for _ in HORIZONS]
    n_days = 0

    for f in files:
        c = pd.read_csv(f, usecols=["Close"])["Close"].dropna().to_numpy(dtype=float)
        if len(c) < 300:
            continue
        n_days += len(c)
        # ベースライン: 全時点の先行リターン（無条件）
        for k, h in enumerate(HORIZONS):
            if len(c) > h:
                base[k].append(c[h:] / c[:-h] - 1.0)

        last: dict[str, int] = {}
        for name, e in detect(c):
            if e - last.get(name, -10**9) < MIN_GAP:
                continue
            last[name] = e
            rec = hits.setdefault(name, [[] for _ in HORIZONS])
            for k, h in enumerate(HORIZONS):
                if e + h < len(c):
                    rec[k].append(float(c[e + h] / c[e] - 1.0))

    base_stats = []
    for k, h in enumerate(HORIZONS):
        b = np.concatenate(base[k])
        base_stats.append((float((b > 0).mean()), float(np.median(b))))

    print(f"延べ {n_days:,} 営業日\n")
    print("【ベースライン（無条件）】")
    for (h, (p, m)) in zip(HORIZONS, base_stats):
        print(f"  {h:>2}日後: 上昇確率 {p*100:5.1f}%  中央値 {m*100:+.2f}%")

    rows = []
    for name, rec in hits.items():
        bear = name in BEARISH
        r = {"パターン": name, "方向": "下落" if bear else "上昇", "検出数": len(rec[2])}
        for k, h in enumerate(HORIZONS):
            a = np.array(rec[k])
            if len(a) == 0:
                r[f"{h}日的中率"] = r[f"{h}日差"] = np.nan
                continue
            # 予告どおりに動いた割合と、同方向のベースライン
            hit = float((a < 0).mean()) if bear else float((a > 0).mean())
            bh = 1.0 - base_stats[k][0] if bear else base_stats[k][0]
            r[f"{h}日的中率"] = round(hit * 100, 1)
            r[f"{h}日差"] = round((hit - bh) * 100, 1)
            if h == 20:
                med = float(np.median(a))
                r["20日中央値%"] = round(med * 100, 2)
                # 弱気パターンは下落幅が大きいほど良いので符号を反転して比較
                edge = (base_stats[k][1] - med) if bear else (med - base_stats[k][1])
                r["20日超過%"] = round(edge * 100, 2)
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("20日差", ascending=False)
    df = df[df["検出数"] >= 30]
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print("\n【パターン別】的中率 = 予告どおりに動いた割合 / 差 = 同方向ベースラインとの差")
    print(df.to_string(index=False))
    df.to_csv(BASE_DIR / "data" / "pattern_stats.csv", index=False, encoding="utf-8-sig")
    print(f"\ndata/pattern_stats.csv に保存")


if __name__ == "__main__":
    main()
