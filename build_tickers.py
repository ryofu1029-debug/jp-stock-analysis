# -*- coding: utf-8 -*-
"""JPX公式の上場銘柄一覧（data_j.xls）から data/tickers.csv を自動生成する。

手動リストは300社が限界なので、600社以降はTOPIXの規模区分で機械的に選定する:
  large = TOPIX Core30 + Large70（計100社）
  mid   = TOPIX Mid400（400社）
  small = TOPIX Small 1 のうちコード順の先頭 N社（既定100社 → 合計600社）

使い方:
    python build_tickers.py        # 600社（small=100）
    python build_tickers.py 500    # 1000社（small=500）
"""
import io
import sys
from pathlib import Path

import pandas as pd

import ssl_setup  # noqa: F401  Avast対策
import requests

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
OUT = Path(__file__).parent / "data" / "tickers.csv"

SIZE_TO_GROUP = {
    "TOPIX Core30": "large",
    "TOPIX Large70": "large",
    "TOPIX Mid400": "mid",
    "TOPIX Small 1": "small",
}


def main() -> None:
    n_small = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("JPX上場銘柄一覧をダウンロード中…")
    r = requests.get(JPX_URL, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content))

    # 内国株式のみ・TOPIX構成銘柄のみ
    df = df[df["市場・商品区分"].astype(str).str.contains("内国株式")]
    df = df[df["規模区分"].isin(SIZE_TO_GROUP)]
    df["group"] = df["規模区分"].map(SIZE_TO_GROUP)
    df["symbol"] = df["コード"].astype(str).str.strip() + ".T"
    df["name"] = df["銘柄名"].astype(str).str.strip()
    df["sector"] = df["33業種区分"].astype(str).str.strip()
    df = df.sort_values("symbol")  # コードは英字入り(130A等)があるため文字列で揃える

    parts = [
        df[df["group"] == "large"],
        df[df["group"] == "mid"],
        df[df["group"] == "small"].head(n_small),
    ]
    out = pd.concat(parts)[["symbol", "name", "group", "sector"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    counts = out["group"].value_counts()
    print(f"生成完了: {OUT.name} / large {counts.get('large', 0)} / "
          f"mid {counts.get('mid', 0)} / small {counts.get('small', 0)} / 合計 {len(out)}社")


if __name__ == "__main__":
    main()
