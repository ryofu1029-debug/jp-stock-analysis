# -*- coding: utf-8 -*-
"""銘柄のインクリメンタル検索（正規化 + AND絞り込み）。

表記ゆれ（全角/半角・ひらがな/カタカナ・中黒・長音記号）を吸収したうえで、
コード・社名・業種を対象に複数キーワードのAND検索を行う。

依存は標準ライブラリ + pandas のみ（streamlit には依存しない）。
"""
import re
import unicodedata

import pandas as pd

# 中黒と長音記号は表記ゆれの温床なので落とす
# （例:「パン・パシフィック…」を「パンパシフィック」で引けるようにする）
_DROP = str.maketrans("", "", "・ー")

# 残すのは 数字・英小文字・カタカナ・漢字（々含む）だけ。
# 空白・括弧・ハイフン・ドット・記号類はすべてノイズとして除去する。
_KEEP = re.compile(r"[^0-9a-z々゠-ヿ㐀-䶿一-鿿]")

# 正規化済み検索対象のキャッシュ（キー: 元文字列群のハッシュ）
_CACHE: dict[int, list[str]] = {}
_CACHE_MAX = 8


def normalize(text: str) -> str:
    """検索用に表記ゆれを吸収した文字列へ変換する。

    NFKCで全角英数→半角・半角カナ→全角カナに揃え、ひらがな→カタカナ変換と
    小文字化を行い、空白・中黒・長音記号・括弧などノイズになる記号を除去する。
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)
    s = _KEEP.sub("", s.lower())
    return s.translate(_DROP)


def _raw_targets(df: pd.DataFrame) -> list[str]:
    """各行の検索対象文字列（symbol + コード部 + name + sector）を組み立てる。"""
    sym = df["symbol"].astype(str)
    parts = [sym, sym.str.replace(".T", "", regex=False)]
    for col in ("name", "sector"):
        if col in df.columns:
            parts.append(df[col].astype(str))
    joined = parts[0]
    for p in parts[1:]:
        joined = joined + " " + p
    return joined.tolist()


def _haystack(df: pd.DataFrame) -> list[str]:
    """正規化済みの検索対象を返す。同じ内容のDataFrameならキャッシュを使い回す。

    毎キーストロークで呼ばれるため、重い正規化は内容ハッシュ単位で1回だけ行う。
    """
    raw = _raw_targets(df)
    key = hash(tuple(raw))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    hay = [normalize(s) for s in raw]
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = hay
    return hay


def filter_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """クエリの全語にマッチする行だけを返す（空白区切りのAND検索）。

    クエリが空なら df をそのまま返す。
    """
    if not query or not query.strip():
        return df
    terms = [t for t in (normalize(w) for w in query.split()) if t]
    if not terms:
        return df
    hay = _haystack(df)
    mask = [all(t in h for t in terms) for h in hay]
    return df[pd.Series(mask, index=df.index)]
