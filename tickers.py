# -*- coding: utf-8 -*-
"""銘柄リスト。data/tickers.csv（build_tickers.py がJPX公式リストから生成）を読み込む。

銘柄を増やす手順:
    python build_tickers.py 100   # 600社 (TOPIX500 + Small1の先頭100)
    python build_tickers.py 500   # 1000社 (TOPIX1000相当)
    python fetch_data.py && python process_data.py

コード: (企業名, 区分, 業種)
区分: "large" = TOPIX Core30+Large70 / "mid" = TOPIX Mid400 / "small" = TOPIX Small
※手動選定していた頃の300社リストは tickers_manual_300.py に保存してある。
"""
import csv
from pathlib import Path

_CSV = Path(__file__).parent / "data" / "tickers.csv"

TICKERS: dict[str, tuple[str, str, str]] = {}
if _CSV.exists():
    with open(_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            TICKERS[row["symbol"]] = (row["name"], row["group"], row["sector"])
else:
    raise FileNotFoundError(
        f"{_CSV} がありません。先に `python build_tickers.py` を実行してください。")


def get_symbols(group: str | None = None) -> list[str]:
    """区分でフィルタしたティッカーのリストを返す。group=None なら全銘柄。"""
    if group is None:
        return list(TICKERS.keys())
    return [s for s, (_, g, _) in TICKERS.items() if g == group]


if __name__ == "__main__":
    counts = {}
    for _, (_, g, _) in TICKERS.items():
        counts[g] = counts.get(g, 0) + 1
    print(f"large: {counts.get('large', 0)}社 / mid: {counts.get('mid', 0)}社 / "
          f"small: {counts.get('small', 0)}社 / 合計: {len(TICKERS)}社")
