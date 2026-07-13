# -*- coding: utf-8 -*-
"""公式リストから data/tickers.csv を自動生成する。

日本株: JPX公式の上場銘柄一覧（data_j.xls）からTOPIX規模区分で選定
  large = TOPIX Core30 + Large70（計100社）
  mid   = TOPIX Mid400（400社）
  small = TOPIX Small 1 のうちコード順の先頭 N社（既定100社）
米国株: Wikipedia の S&P 500 構成銘柄一覧（約503社）
  us    = S&P 500 全構成銘柄

使い方:
    python build_tickers.py        # 日本株600社 + 米国株S&P500
    python build_tickers.py 500    # 日本株1000社 + 米国株S&P500
"""
import io
import sys
from pathlib import Path

import pandas as pd

import ssl_setup  # noqa: F401  Avast対策
import requests

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT = Path(__file__).parent / "data" / "tickers.csv"

SIZE_TO_GROUP = {
    "TOPIX Core30": "large",
    "TOPIX Large70": "large",
    "TOPIX Mid400": "mid",
    "TOPIX Small 1": "small",
}


def fetch_jp(n_small: int) -> pd.DataFrame:
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

    return pd.concat([
        df[df["group"] == "large"],
        df[df["group"] == "mid"],
        df[df["group"] == "small"].head(n_small),
    ])[["symbol", "name", "group", "sector"]]


def fetch_us() -> pd.DataFrame:
    print("S&P500構成銘柄一覧をダウンロード中…")
    r = requests.get(SP500_URL, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    return pd.DataFrame({
        # Yahoo表記に変換（BRK.B → BRK-B など）
        "symbol": t["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
        "name": t["Security"].astype(str).str.strip(),
        "group": "us",
        "sector": t["GICS Sector"].astype(str).str.strip(),
    }).sort_values("symbol")


def main() -> None:
    n_small = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    out = pd.concat([fetch_jp(n_small), fetch_us()], ignore_index=True)
    out = out.drop_duplicates(subset="symbol")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    counts = out["group"].value_counts()
    print(f"生成完了: {OUT.name} / large {counts.get('large', 0)} / "
          f"mid {counts.get('mid', 0)} / small {counts.get('small', 0)} / "
          f"us {counts.get('us', 0)} / 合計 {len(out)}社")


if __name__ == "__main__":
    main()
