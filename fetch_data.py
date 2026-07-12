# -*- coding: utf-8 -*-
"""フェーズ1: yfinance で日足データ（OHLCV・過去5年）を取得して CSV 保存する。

使い方:
    python fetch_data.py            # 全銘柄
    python fetch_data.py 7203.T     # 指定銘柄のみ再取得

- data/raw/{コード}.csv に保存（例: data/raw/7203.T.csv）
- バッチ取得（20社ずつ）+ バッチ間スリープでレート制限を回避
- 失敗した銘柄は1社ずつリトライし、最終的な失敗は data/failed.txt に記録
"""
import sys
import time
from pathlib import Path

import ssl_setup  # noqa: F401  Avast対策。yfinance より先に import する
import pandas as pd
import yfinance as yf

from tickers import TICKERS, get_symbols

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
FAILED_FILE = BASE_DIR / "data" / "failed.txt"

PERIOD = "5y"
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 2.0  # 秒


def save_one(symbol: str, df: pd.DataFrame) -> bool:
    """1銘柄分の DataFrame を検証して CSV 保存。行数が少なすぎる場合は失敗扱い。"""
    df = df.dropna(subset=["Close"])
    if len(df) < 100:  # 5年分なら約1200行あるはず。極端に少ないのはデータ不良
        return False
    df = df[["Open", "High", "Low", "Close", "Volume"]].round(2)
    df.index.name = "Date"
    df.to_csv(RAW_DIR / f"{symbol}.csv", encoding="utf-8")
    return True


def fetch_batch(symbols: list[str]) -> tuple[list[str], list[str]]:
    """複数銘柄を一括ダウンロード。(成功リスト, 失敗リスト) を返す。"""
    ok, failed = [], []
    df = yf.download(
        symbols,
        period=PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    for sym in symbols:
        try:
            sub = df[sym] if len(symbols) > 1 else df
            if save_one(sym, sub.copy()):
                ok.append(sym)
            else:
                failed.append(sym)
        except (KeyError, TypeError):
            failed.append(sym)
    return ok, failed


def retry_single(symbol: str) -> bool:
    """失敗銘柄を1社ずつリトライ。"""
    try:
        df = yf.Ticker(symbol).history(period=PERIOD, interval="1d", auto_adjust=True)
        return save_one(symbol, df)
    except Exception:
        return False


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    symbols = sys.argv[1:] if len(sys.argv) > 1 else get_symbols()
    total = len(symbols)
    print(f"=== データ取得開始: {total}社 × 過去{PERIOD} ===")

    done, all_failed = 0, []
    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        ok, failed = fetch_batch(batch)
        done += len(ok)
        all_failed.extend(failed)
        print(f"  [{min(i + BATCH_SIZE, total):>3}/{total}] 成功 {len(ok)} / 失敗 {len(failed)}")
        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    # 失敗分をリトライ
    still_failed = []
    if all_failed:
        print(f"--- 失敗 {len(all_failed)}社 を個別リトライ ---")
        for sym in all_failed:
            time.sleep(1.0)
            if retry_single(sym):
                done += 1
                print(f"  リトライ成功: {sym} ({TICKERS.get(sym, ('?',))[0]})")
            else:
                still_failed.append(sym)
                print(f"  リトライ失敗: {sym} ({TICKERS.get(sym, ('?',))[0]})")

    if len(sys.argv) <= 1:  # 全銘柄実行のときだけ記録（個別再取得で上書きしない）
        FAILED_FILE.write_text("\n".join(still_failed), encoding="utf-8")
    print(f"=== 完了: 成功 {done}/{total}社 / 失敗 {len(still_failed)}社 ===")
    if still_failed:
        print("失敗銘柄:", " ".join(still_failed))


if __name__ == "__main__":
    main()
