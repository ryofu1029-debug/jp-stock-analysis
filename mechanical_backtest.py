# -*- coding: utf-8 -*-
"""「毎回機械的に買えば実際に儲かるか」を口座レベルでシミュレーションする。

rule_tradeoff.py までは「シグナル1回あたりの平均超過リターン」だけを見ていた。
しかしそれだけでは、シグナルが少ないルールが実際の運用でどうなるかが分からない。
シグナルが少ない=資金の大半が現金のまま眠る時間が長い=複利が効かない、という
「資金効率」の問題は1回あたりの統計には現れない。

ここではN個の口座枠を用意し、各枠が独立して
  「空いていれば、その日の候補の中で最も売られすぎている銘柄を機械的に買う
   → 40営業日後に機械的に売る」
を裁量なしで繰り返すシミュレーションを行い、実際の資産曲線・CAGR・最大
ドローダウンを、587銘柄均等分散バイアンドホールドと比較する。

裁量が入る余地はゼロ（銘柄名も見ない、優先順位はしきい値からの乖離だけで決まる）。
保有していない間の現金は0%（利息なし）としており、これが資金効率の悪いルールを
不利に見せる主因になる。

使い方: python mechanical_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rule_optimize import LONG_DIR, MIN_ROWS, longest_continuous, rsi, sanitize

HOLD = 40
COST = 0.002
N_SLOTS = 20

RULES = [
    ("RSI21_15", "RSI", 21, 15),
    ("RSI21_20", "RSI", 21, 20),
    ("RSI21_25", "RSI", 21, 25),
    ("RSI21_30", "RSI", 21, 30),
    ("急落5日-25%", "DROP", 5, 0.25),
    ("急落10日-20%", "DROP", 10, 0.20),
    ("急落10日-15%", "DROP", 10, 0.15),
]


def load_named() -> dict[str, pd.Series]:
    out = {}
    for f in sorted(LONG_DIR.glob("*.T.csv")):
        d = pd.read_csv(f, usecols=["Date", "Close"], parse_dates=["Date"]).dropna()
        if len(d) < MIN_ROWS:
            continue
        c = d["Close"].to_numpy(dtype=float)
        ok = sanitize(c)
        c, d = c[ok], d[ok]
        if len(c) < MIN_ROWS:
            continue
        sl = longest_continuous(c)
        c, d = c[sl], d.iloc[sl]
        if len(c) < MIN_ROWS:
            continue
        out[f.stem] = pd.Series(c, index=pd.DatetimeIndex(d["Date"].to_numpy()))
    return out


def build_matrix(series: dict[str, pd.Series]) -> pd.DataFrame:
    """全銘柄を共通の日付軸に揃える。短い空白(5日以内)は埋め、それ以外はNaNのまま
    （上場前・データ欠落は「買えない」を意味するのでNaNのままにする）。"""
    all_dates = sorted(set().union(*[s.index for s in series.values()]))
    idx = pd.DatetimeIndex(all_dates)
    df = pd.DataFrame({sym: s.reindex(idx) for sym, s in series.items()})
    return df.ffill(limit=5)


def gen_signals(sym: str, native: pd.Series, master: pd.DatetimeIndex,
                kind: str, p1, p2) -> list[tuple[int, float, str]]:
    """(master上のday_idx, 優先度[小さいほど優先], symbol) のリストを返す。"""
    c = native.to_numpy()
    n = len(c)
    out = []
    if kind == "RSI":
        r = rsi(c, p1)
        cross = np.flatnonzero((r <= p2) & (np.roll(r, 1) > p2))
        cross = cross[cross > p1]
        prio = r[cross]  # 低いほど優先
    else:  # DROP
        ratio = np.full(n, np.nan)
        ratio[p1:] = c[p1:] / c[:-p1] - 1.0
        cross = np.flatnonzero(ratio <= -p2)
        prio = ratio[cross]  # 小さい(マイナスが大きい)ほど優先
    if len(cross) == 0:
        return out
    loc = master.get_indexer(native.index[cross])
    for di, pr in zip(loc, prio):
        if di >= 0:
            out.append((int(di), float(pr), sym))
    return out


def simulate(price: np.ndarray, cols: dict[str, int],
            signals_by_day: dict[int, list[tuple[float, str]]],
            n_slots: int = N_SLOTS, hold: int = HOLD, cost: float = COST):
    n_days = price.shape[0]
    slot_cash = np.full(n_slots, 1.0 / n_slots)
    slot_sym = [None] * n_slots
    slot_entry_price = np.zeros(n_slots)
    slot_entry_equity = np.zeros(n_slots)
    slot_exit_day = np.full(n_slots, -1, dtype=int)
    held: set[str] = set()

    equity = np.zeros(n_days)
    n_open = np.zeros(n_days, dtype=int)
    trade_rets: list[float] = []

    for day in range(n_days):
        for k in range(n_slots):
            if slot_exit_day[k] == day:
                sym = slot_sym[k]
                px = price[day, cols[sym]]
                if not np.isfinite(px):
                    px = slot_entry_price[k]  # 売却日に価格がなければ前日値で据え置き
                ret = px / slot_entry_price[k] - 1 - cost
                slot_cash[k] = slot_entry_equity[k] * (1 + ret)
                trade_rets.append(ret)
                held.discard(sym)
                slot_sym[k] = None

        todays = signals_by_day.get(day)
        if todays:
            idle = [k for k in range(n_slots) if slot_sym[k] is None]
            si = 0
            for k in idle:
                while si < len(todays) and todays[si][1] in held:
                    si += 1
                if si >= len(todays):
                    break
                _, sym = todays[si]
                si += 1
                px = price[day, cols[sym]]
                if not np.isfinite(px) or px <= 0:
                    continue
                slot_sym[k] = sym
                slot_entry_price[k] = px
                slot_entry_equity[k] = slot_cash[k]
                slot_exit_day[k] = day + hold
                held.add(sym)

        total = 0.0
        cnt = 0
        for k in range(n_slots):
            if slot_sym[k] is not None:
                px = price[day, cols[slot_sym[k]]]
                total += slot_entry_equity[k] * (px / slot_entry_price[k]) if np.isfinite(px) \
                    else slot_cash[k]
                cnt += 1
            else:
                total += slot_cash[k]
        equity[day] = total
        n_open[day] = cnt

    return equity, n_open, np.array(trade_rets)


def metrics(equity: np.ndarray, dates: pd.DatetimeIndex, n_open: np.ndarray,
            n_slots: int) -> dict:
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    daily_ret = np.diff(np.log(equity))
    vol = float(np.std(daily_ret) * np.sqrt(250))
    return {
        "CAGR%": round(cagr * 100, 2),
        "最終倍率": round(float(equity[-1] / equity[0]), 2),
        "最大DD%": round(float(dd.min()) * 100, 1),
        "年率vol%": round(vol * 100, 1),
        "CAGR/vol": round(cagr / vol, 2) if vol > 0 else np.nan,
        "稼働率%": round(float(n_open.mean() / n_slots) * 100, 1),
    }


def main() -> None:
    print("=== データ読み込み ===")
    series = load_named()
    print(f"日本株 {len(series)}銘柄")

    price_df = build_matrix(series)
    master = price_df.index
    price = price_df.to_numpy(dtype=np.float32)
    cols = {c: i for i, c in enumerate(price_df.columns)}
    print(f"共通カレンダー: {master[0].date()} 〜 {master[-1].date()} ({len(master):,}日)")

    # ベンチマーク: 587銘柄均等分散バイアンドホールド
    rets = price_df.pct_change()
    bench = (1 + rets.mean(axis=1).fillna(0)).cumprod().to_numpy()
    bench = bench / bench[0]
    bm = metrics(bench, master, np.full(len(master), N_SLOTS), N_SLOTS)
    print(f"\n【ベンチマーク】587銘柄 均等分散バイアンドホールド")
    print(f"  CAGR {bm['CAGR%']}% / 最終倍率 {bm['最終倍率']}倍 / "
          f"最大DD {bm['最大DD%']}% / 年率vol {bm['年率vol%']}%")

    rows = [{"ルール": "ベンチマーク(バイアンドホールド)", **bm, "取引数": "-"}]
    curves = {"ベンチマーク": bench}

    for name, kind, p1, p2 in RULES:
        sigs: dict[int, list[tuple[float, str]]] = {}
        for sym, s in series.items():
            for di, pr, sy in gen_signals(sym, s, master, kind, p1, p2):
                sigs.setdefault(di, []).append((pr, sy))
        for di in sigs:
            sigs[di].sort()  # 優先度の小さい順

        eq, nop, trades = simulate(price, cols, sigs, N_SLOTS)
        m = metrics(eq, master, nop, N_SLOTS)
        m["ルール"] = name
        m["取引数"] = len(trades)
        m["1回平均%"] = round(float(trades.mean()) * 100, 2) if len(trades) else np.nan
        rows.append(m)
        curves[name] = eq
        print(f"  {name:<14} CAGR {m['CAGR%']:>6.2f}%  最終倍率 {m['最終倍率']:>6.2f}倍  "
              f"最大DD {m['最大DD%']:>6.1f}%  稼働率 {m['稼働率%']:>5.1f}%  取引数 {m['取引数']}")

    df = pd.DataFrame(rows)
    cols_order = ["ルール", "CAGR%", "最終倍率", "最大DD%", "年率vol%", "CAGR/vol",
                  "稼働率%", "取引数", "1回平均%"]
    df = df[[c for c in cols_order if c in df.columns]]
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    print(f"\n【全ルール比較（口座枠 {N_SLOTS}）】")
    print(df.to_string(index=False))

    out = Path(__file__).parent / "data"
    df.to_csv(out / "mechanical_backtest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves, index=master).to_csv(out / "mechanical_curves.csv", encoding="utf-8-sig")
    print(f"\ndata/mechanical_backtest.csv, data/mechanical_curves.csv に保存")


if __name__ == "__main__":
    main()
