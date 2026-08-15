# -*- coding: utf-8 -*-
"""検証用に「上場来すべて」の日足を取得する（period="max"）。

パターン検証の確度は銘柄数より**レジームの多様性**で決まる。5年分だと
2021-2026年の上昇相場しか経験しておらず、リーマンショックやチャイナショックの
ような局面を一度も試していないため、遡れるだけ遡ったデータを別途用意する。

保存先は data/ ではなく %LOCALAPPDATA%\\jp-stock-analysis\\long\\:
  - data/ はGit管理下かつStreamlit Cloudへデプロイされるので膨らませたくない
  - OneDrive配下だと数百MBの同期が走る
アプリ本体（app.py）はこのデータを使わない。検証スクリプト専用。

使い方:
    python fetch_long.py            # 全銘柄
    python fetch_long.py 7203.T     # 指定銘柄のみ
"""
import os
import sys
import time
from pathlib import Path

import ssl_setup  # noqa: F401  Avast対策。yfinance より先に import する
import pandas as pd
import yfinance as yf

from tickers import TICKERS, get_symbols

LONG_DIR = Path(os.environ.get(
    "JPSA_LONG_DIR",
    Path(os.environ.get("LOCALAPPDATA", ".")) / "jp-stock-analysis" / "long"))

PERIOD = "max"
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 2.0
MIN_ROWS = 500  # 検証に使うので最低2年程度は欲しい


def save_one(symbol: str, df: pd.DataFrame) -> bool:
    df = df.dropna(subset=["Close"])
    if len(df) < MIN_ROWS:
        return False
    df = df[["Open", "High", "Low", "Close", "Volume"]].round(2)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)  # 米国株はtz付きで返ることがある
    df.index.name = "Date"
    df.to_csv(LONG_DIR / f"{symbol}.csv", encoding="utf-8")
    return True


def fetch_batch(symbols: list[str]) -> tuple[list[str], list[str]]:
    ok, failed = [], []
    df = yf.download(symbols, period=PERIOD, interval="1d", group_by="ticker",
                     auto_adjust=True, progress=False, threads=True)
    for sym in symbols:
        try:
            sub = df[sym] if len(symbols) > 1 else df
            (ok if save_one(sym, sub.copy()) else failed).append(sym)
        except (KeyError, TypeError):
            failed.append(sym)
    return ok, failed


def main() -> None:
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sys.argv[1:] if len(sys.argv) > 1 else get_symbols()
    total = len(symbols)
    print(f"=== 上場来データ取得: {total}社 → {LONG_DIR} ===")

    done, all_failed = 0, []
    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        ok, failed = fetch_batch(batch)
        done += len(ok)
        all_failed.extend(failed)
        print(f"  [{min(i + BATCH_SIZE, total):>4}/{total}] 成功 {len(ok)} / 失敗 {len(failed)}",
              flush=True)
        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    still = []
    if all_failed:
        print(f"--- 失敗 {len(all_failed)}社 を個別リトライ ---", flush=True)
        for sym in all_failed:
            time.sleep(1.0)
            try:
                d = yf.Ticker(sym).history(period=PERIOD, interval="1d", auto_adjust=True)
                if save_one(sym, d):
                    done += 1
                    continue
            except Exception:
                pass
            still.append(sym)

    print(f"=== 完了: 成功 {done}/{total}社 / 失敗 {len(still)}社 ===")

    # どこまで遡れたかの分布を出す（検証設計の判断材料）
    starts = []
    for f in LONG_DIR.glob("*.csv"):
        d = pd.read_csv(f, usecols=["Date"], parse_dates=["Date"])
        if len(d):
            starts.append((f.stem, d["Date"].iloc[0], len(d)))
    s = pd.DataFrame(starts, columns=["symbol", "start", "rows"])
    s["year"] = s["start"].dt.year
    print(f"\n--- 遡れた年の分布（{len(s)}銘柄・延べ{s['rows'].sum():,}営業日） ---")
    print(s["year"].value_counts().sort_index().to_string())
    print("\n年別カバー銘柄数（その年より前から存在する銘柄）")
    for y in (1990, 2000, 2006, 2008, 2011, 2016, 2020):
        print(f"  {y}年以前から: {int((s['year'] <= y).sum()):>5}銘柄")


if __name__ == "__main__":
    main()
