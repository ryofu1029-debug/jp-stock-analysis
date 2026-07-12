# -*- coding: utf-8 -*-
"""チャート画像から価格の形状（1本の曲線）を抽出する。

スクリーンショットや写真のチャート（折れ線・ローソク足）を対象に、
各x座標で「チャート本体らしい画素」のy中央値を拾って曲線に変換する。
背景（白/黒）・グリッド線・軸まわりはある程度自動で除外するが完璧ではないので、
抽出結果を画面に表示してユーザーに目視確認してもらう前提の作り。
"""
import numpy as np
import pandas as pd
from PIL import Image

MAX_WIDTH = 800  # 処理を軽くするための上限


def extract_series_from_image(img: Image.Image, n_points: int = 60) -> np.ndarray | None:
    """画像から価格曲線を抽出し、n_points 点にリサンプルして返す。失敗時 None。"""
    im = img.convert("RGB")
    if im.width > MAX_WIDTH:
        im = im.resize((MAX_WIDTH, max(1, int(im.height * MAX_WIDTH / im.width))))
    a = np.asarray(im, dtype=np.float32)
    gray = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)  # 彩度っぽさ（色付き線・ローソクの検出用）

    bg_is_light = float(np.median(gray)) > 128
    if bg_is_light:
        fg = (sat > 40) | (gray < 70)   # 白背景: 色付き or 濃い画素が前景
    else:
        fg = (sat > 40) | (gray > 200)  # 黒背景: 色付き or 明るい画素が前景

    h, w = fg.shape
    # 軸ラベル・凡例が集まりやすい外周5%を捨てる
    mx, my = max(1, int(w * 0.05)), max(1, int(h * 0.05))
    fg[:my, :] = False
    fg[h - my:, :] = False
    fg[:, :mx] = False
    fg[:, w - mx:] = False

    counts = fg.sum(axis=0)
    ys = np.full(w, np.nan)
    for x in range(w):
        # 前景ゼロの列と、縦軸・縦線のような「縦に半分以上埋まる」列は無視
        if 0 < counts[x] < h * 0.5:
            ys[x] = float(np.median(np.where(fg[:, x])[0]))

    valid = ~np.isnan(ys)
    if valid.sum() < w * 0.3:  # 3割以上の列で線を拾えなければ抽出失敗
        return None

    xs = np.arange(w)
    ys = np.interp(xs, xs[valid], ys[valid])
    # ローソクのヒゲ・写真ノイズ対策の平滑化
    ys = pd.Series(ys).rolling(5, center=True, min_periods=1).median().to_numpy()

    vals = -ys  # 画像は下方向が正なので反転（上=高値）
    idx = np.linspace(0, len(vals) - 1, n_points)
    return np.interp(idx, np.arange(len(vals)), vals)
