# -*- coding: utf-8 -*-
"""mechanical_backtest.py の弱点（アイドル中は現金0%）を直した版。

mechanical_backtest.py では、RSI系ルールは1回あたりの優位性が本物なのに、
シグナル待ちの間ずっと現金(0%)で寝ているせいでベンチマークに惨敗していた。
これは「ルールに価値がない」のではなく「アイドル時間の設計が悪い」ことが
原因の可能性がある。

ここでは各口座枠を、シグナル待ちの間は「ベンチマーク（587銘柄均等分散）」に
投資したままにし、シグナルが出た瞬間だけベンチマークを売って個別銘柄に
乗り換え、保有後にまたベンチマークへ戻す設計にする。

これにより「そのルールの個別銘柄選択が、ただの市場エクスポージャーに対して
本当に上乗せの価値を生んでいるか」を測れる。cash版で負けていたルールが
overlay版で勝てば「アイドル設計の問題」、overlay版でも勝てなければ
「ルール自体に市場を超える価値がない」と判定できる。

使い方: python mechanical_overlay.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mechanical_backtest import (COST, HOLD, N_SLOTS, RULES, build_matrix,
                                 gen_signals, load_named, metrics)


def simulate_overlay(price: np.ndarray, cols: dict[str, int], bench: np.ndarray,
                     signals_by_day: dict[int, list[tuple[float, str]]],
                     n_slots: int = N_SLOTS, hold: int = HOLD, cost: float = COST):
    n_days = price.shape[0]
    slot_sym = [None] * n_slots
    slot_entry_price = np.zeros(n_slots)
    slot_entry_equity = np.zeros(n_slots)
    slot_exit_day = np.full(n_slots, -1, dtype=int)
    idle_base_bench = np.full(n_slots, bench[0])
    idle_base_equity = np.full(n_slots, 1.0 / n_slots)
    held: set[str] = set()

    equity = np.zeros(n_days)
    n_open = np.zeros(n_days, dtype=int)
    trade_rets: list[float] = []

    def idle_value(k: int, day: int) -> float:
        return idle_base_equity[k] * (bench[day] / idle_base_bench[k])

    for day in range(n_days):
        for k in range(n_slots):
            if slot_exit_day[k] == day:
                sym = slot_sym[k]
                px = price[day, cols[sym]]
                if not np.isfinite(px):
                    px = slot_entry_price[k]
                ret = px / slot_entry_price[k] - 1 - cost
                new_equity = slot_entry_equity[k] * (1 + ret)
                trade_rets.append(ret)
                held.discard(sym)
                slot_sym[k] = None
                idle_base_bench[k] = bench[day]     # ベンチマークへ乗り換え
                idle_base_equity[k] = new_equity

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
                slot_entry_equity[k] = idle_value(k, day)   # ベンチマークから乗り換え
                slot_exit_day[k] = day + hold
                held.add(sym)

        total = 0.0
        cnt = 0
        for k in range(n_slots):
            if slot_sym[k] is not None:
                px = price[day, cols[slot_sym[k]]]
                total += slot_entry_equity[k] * (px / slot_entry_price[k]) if np.isfinite(px) \
                    else idle_value(k, day)
                cnt += 1
            else:
                total += idle_value(k, day)
        equity[day] = total
        n_open[day] = cnt

    return equity, n_open, np.array(trade_rets)


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
    print(f"\n【ベンチマーク】CAGR {bm['CAGR%']}%  最終倍率 {bm['最終倍率']}倍  最大DD {bm['最大DD%']}%\n")

    rows = [{"ルール": "ベンチマーク", **bm, "取引数": "-", "対ベンチ倍率": 1.0}]
    for name, kind, p1, p2 in RULES:
        sigs: dict[int, list[tuple[float, str]]] = {}
        for sym, s in series.items():
            for di, pr, sy in gen_signals(sym, s, master, kind, p1, p2):
                sigs.setdefault(di, []).append((pr, sy))
        for di in sigs:
            sigs[di].sort()

        eq, nop, trades = simulate_overlay(price, cols, bench, sigs, N_SLOTS)
        m = metrics(eq, master, nop, N_SLOTS)
        m["ルール"] = name
        m["取引数"] = len(trades)
        m["対ベンチ倍率"] = round(float(eq[-1] / bench[-1]), 2)
        rows.append(m)
        print(f"  {name:<14} CAGR {m['CAGR%']:>6.2f}%  最終倍率 {m['最終倍率']:>7.2f}倍  "
              f"最大DD {m['最大DD%']:>6.1f}%  CAGR/vol {m['CAGR/vol']:>5.2f}  "
              f"対ベンチ {m['対ベンチ倍率']:>5.2f}倍")

    df = pd.DataFrame(rows)
    cols_order = ["ルール", "CAGR%", "最終倍率", "対ベンチ倍率", "最大DD%", "年率vol%",
                  "CAGR/vol", "稼働率%", "取引数"]
    df = df[[c for c in cols_order if c in df.columns]]
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    print(f"\n【アイドル中はベンチマークに投資する版・口座枠{N_SLOTS}】")
    print(df.to_string(index=False))

    out = Path(__file__).parent / "data" / "mechanical_overlay.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{out.name} に保存")


if __name__ == "__main__":
    main()
