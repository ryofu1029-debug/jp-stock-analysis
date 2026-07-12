# -*- coding: utf-8 -*-
"""フェーズ2+3 実行: raw CSV を読み、指標と予測を付けて保存する。

- data/processed/{コード}.csv … 指標列を追加した全履歴
- data/summary.csv … 全銘柄の最新シグナル一覧（Streamlit のスクリーニングで使用）

使い方: python process_data.py
"""
from pathlib import Path

import pandas as pd

from indicators import add_all_indicators, latest_signals
from forecast import forecast_summary
from tickers import TICKERS

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SUMMARY_FILE = BASE_DIR / "data" / "summary.csv"


def process_one(csv_path: Path) -> dict | None:
    symbol = csv_path.stem
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    if len(df) < 80:
        return None

    df = add_all_indicators(df)
    df.round(3).to_csv(PROCESSED_DIR / f"{symbol}.csv", encoding="utf-8")

    name, group, sector = TICKERS.get(symbol, (symbol, "?", "?"))
    row = {"symbol": symbol, "name": name, "group": group, "sector": sector,
           "last_date": df.index[-1].date()}
    row.update(latest_signals(df))
    row.update(forecast_summary(df))
    return row


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 銘柄リスト更新で外れた銘柄の残骸CSVを掃除（パターン照合ライブラリの汚染防止）
    for d in (RAW_DIR, PROCESSED_DIR):
        for f in d.glob("*.csv"):
            if f.stem not in TICKERS:
                f.unlink()

    files = sorted(RAW_DIR.glob("*.csv"))
    print(f"=== 指標計算 + 予測: {len(files)}銘柄 ===")

    rows = []
    for i, f in enumerate(files, 1):
        row = process_one(f)
        if row:
            rows.append(row)
        if i % 20 == 0 or i == len(files):
            print(f"  [{i:>3}/{len(files)}] 処理済み")

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    print(f"=== 完了: {len(summary)}銘柄のサマリーを {SUMMARY_FILE.name} に保存 ===")

    # 簡易レポート
    if len(summary):
        print("\n--- シグナル分布 ---")
        print("RSI:", summary["rsi_label"].value_counts().to_dict())
        print("MACD:", summary["macd_label"].value_counts().to_dict())
        print("トレンド:", summary["trend"].value_counts().to_dict())


if __name__ == "__main__":
    main()
