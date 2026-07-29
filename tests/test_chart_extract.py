# -*- coding: utf-8 -*-
"""chart_extract の抽出精度を、PILで作った合成チャート画像で測る。

実写のスクリーンショットには正解データが無いので、既知の価格系列から
画像を描き、抽出結果と正解のピアソン相関を見る。旧実装（列ごとのy中央値）も
同梱して比較表を出す。

    python tests/test_chart_extract.py
    python tests/test_chart_extract.py --dump <出力先>   # 画像と重ね描きを保存
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chart_extract import extract_series_detailed, render_overlay  # noqa: E402

W, H = 900, 620
N = 120                 # 価格の本数
TARGET = 0.95           # 目標のピアソン相関


# ---------------------------------------------------------------- 旧実装

def _legacy_extract(img: Image.Image, n_points: int = 60) -> np.ndarray | None:
    """比較用: 作り替え前の実装（列ごとに前景画素のy中央値を取るだけ）。"""
    max_width = 800
    im = img.convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, max(1, int(im.height * max_width / im.width))))
    a = np.asarray(im, dtype=np.float32)
    gray = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)

    bg_is_light = float(np.median(gray)) > 128
    if bg_is_light:
        fg = (sat > 40) | (gray < 70)
    else:
        fg = (sat > 40) | (gray > 200)

    h, w = fg.shape
    mx, my = max(1, int(w * 0.05)), max(1, int(h * 0.05))
    fg[:my, :] = False
    fg[h - my:, :] = False
    fg[:, :mx] = False
    fg[:, w - mx:] = False

    counts = fg.sum(axis=0)
    ys = np.full(w, np.nan)
    for x in range(w):
        if 0 < counts[x] < h * 0.5:
            ys[x] = float(np.median(np.where(fg[:, x])[0]))

    valid = ~np.isnan(ys)
    if valid.sum() < w * 0.3:
        return None

    xs = np.arange(w)
    ys = np.interp(xs, xs[valid], ys[valid])
    ys = pd.Series(ys).rolling(5, center=True, min_periods=1).median().to_numpy()

    vals = -ys
    idx = np.linspace(0, len(vals) - 1, n_points)
    return np.interp(idx, np.arange(len(vals)), vals)


# ------------------------------------------------------------ 画像生成

def _font(size: int) -> ImageFont.ImageFont:
    for name in ("meiryo.ttc", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_prices(seed: int = 7, n: int = N) -> np.ndarray:
    """それらしいランダムウォークの終値系列。"""
    rng = np.random.default_rng(seed)
    walk = rng.normal(0.0, 1.0, n).cumsum()
    wave = 4.0 * np.sin(np.linspace(0, 5.0, n))
    return 2000.0 + 45.0 * (walk + wave + np.linspace(0, 5, n))


def _xy(prices: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    lo, hi = float(prices.min()), float(prices.max())
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(left, right, len(prices))
    ys = bottom - (prices - lo) / (hi - lo) * (bottom - top)
    return list(zip(xs.tolist(), ys.tolist()))


def _grid(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color) -> None:
    left, top, right, bottom = box
    for i in range(1, 5):
        y = top + (bottom - top) * i / 5
        d.line([(left, y), (right, y)], fill=color, width=1)
    for i in range(1, 6):
        x = left + (right - left) * i / 6
        d.line([(x, top), (x, bottom)], fill=color, width=1)


def _volume(d: ImageDraw.ImageDraw, prices: np.ndarray,
            box: tuple[int, int, int, int], seed: int = 3, colored: bool = False) -> None:
    """下部の出来高パネル（旧実装が一番引っ張られる要素）。"""
    left, top, right, bottom = box
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.15, 1.0, len(prices)) * (1 + 0.5 * np.sin(np.linspace(0, 9, len(prices))))
    vol = vol / vol.max()
    xs = np.linspace(left, right, len(prices))
    bw = max(2.0, (right - left) / len(prices) * 0.6)
    for i, x in enumerate(xs):
        y = bottom - vol[i] * (bottom - top)
        if colored:
            up = i == 0 or prices[i] >= prices[i - 1]
            fill = (214, 64, 69) if up else (46, 154, 92)
        else:
            fill = (150, 160, 180)
        d.rectangle([x - bw / 2, y, x + bw / 2, bottom], fill=fill)


def _labels(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
            prices: np.ndarray, color) -> None:
    """右側の価格目盛りと下部の日付ラベル。"""
    left, top, right, bottom = box
    f = _font(14)
    lo, hi = float(prices.min()), float(prices.max())
    for i in range(6):
        y = top + (bottom - top) * i / 5
        d.text((right + 6, y - 8), f"{hi - (hi - lo) * i / 5:,.0f}", fill=color, font=f)
    for i, lab in enumerate(["01/06", "02/03", "03/03", "04/01", "05/07", "06/02"]):
        x = left + (right - left) * i / 5
        d.text((x - 16, bottom + 8), lab, fill=color, font=f)


def _legend(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color) -> None:
    """チャート内に重なる凡例・銘柄名・透かし。"""
    left, top, right, bottom = box
    d.text((left + 10, top + 6), "7203 トヨタ自動車  日足", fill=color, font=_font(18))
    d.text((left + 10, top + 30), "MA5  MA25  MA75", fill=(230, 140, 40), font=_font(14))
    d.text((right - 190, bottom - 26), "(c) SAMPLE SECURITIES",
           fill=(170, 170, 170), font=_font(14))


def _candles(d: ImageDraw.ImageDraw, prices: np.ndarray,
             box: tuple[int, int, int, int], seed: int = 11) -> None:
    left, top, right, bottom = box
    rng = np.random.default_rng(seed)
    opens = np.concatenate(([prices[0]], prices[:-1]))
    span = float(prices.max() - prices.min())
    highs = np.maximum(opens, prices) + rng.uniform(0.01, 0.05, len(prices)) * span
    lows = np.minimum(opens, prices) - rng.uniform(0.01, 0.05, len(prices)) * span

    lo, hi = float(lows.min()), float(highs.max())
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def y_of(v):
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    xs = np.linspace(left + 6, right - 6, len(prices))
    bw = max(3.0, (right - left) / len(prices) * 0.65)
    for i, x in enumerate(xs):
        up = prices[i] >= opens[i]
        col = (214, 64, 69) if up else (46, 154, 92)
        d.line([(x, y_of(highs[i])), (x, y_of(lows[i]))], fill=col, width=1)
        y1, y2 = sorted((y_of(opens[i]), y_of(prices[i])))
        if y2 - y1 < 2:
            y2 = y1 + 2
        d.rectangle([x - bw / 2, y1, x + bw / 2, y2], fill=col)


# ------------------------------------------------------------ パターン

def pat_line_only(p):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 50, W - 80, H - 60)
    d.line(_xy(p, box), fill=(30, 110, 200), width=2, joint="curve")
    return img


def pat_line_volume(p):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 40, W - 80, 400)
    d.line(_xy(p, box), fill=(30, 110, 200), width=2, joint="curve")
    _volume(d, p, (70, 470, W - 80, H - 50))
    return img


def pat_line_text(p):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 50, W - 90, H - 70)
    d.line(_xy(p, box), fill=(30, 110, 200), width=2, joint="curve")
    _labels(d, box, p, (60, 60, 60))
    _legend(d, box, (20, 20, 20))
    return img


def pat_line_grid(p):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 50, W - 80, H - 60)
    _grid(d, box, (198, 206, 214))
    d.rectangle(list(box), outline=(150, 158, 168), width=1)
    d.line(_xy(p, box), fill=(30, 110, 200), width=2, joint="curve")
    return img


def pat_candle_volume(p):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 40, W - 90, 400)
    _grid(d, box, (210, 216, 224))
    _candles(d, p, box)
    _volume(d, p, (70, 470, W - 90, H - 50), colored=True)
    _labels(d, box, p, (60, 60, 60))
    return img


def pat_dark_line(p):
    img = Image.new("RGB", (W, H), (18, 21, 28))
    d = ImageDraw.Draw(img)
    box = (70, 50, W - 80, H - 60)
    _grid(d, box, (44, 50, 62))
    d.line(_xy(p, box), fill=(90, 200, 250), width=2, joint="curve")
    _labels(d, box, p, (190, 196, 206))
    return img


def pat_line_two_ma(p):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 50, W - 80, H - 60)
    s = pd.Series(p)
    ma5 = s.rolling(5, min_periods=1).mean().to_numpy()
    ma25 = s.rolling(25, min_periods=1).mean().to_numpy()
    lo, hi = float(p.min()), float(p.max())

    def clip(v):  # 価格と同じ縦軸に載せる
        return np.clip(v, lo, hi)

    d.line(_xy_shared(clip(ma25), p, box), fill=(150, 90, 200), width=2, joint="curve")
    d.line(_xy_shared(clip(ma5), p, box), fill=(235, 150, 40), width=2, joint="curve")
    d.line(_xy(p, box), fill=(30, 110, 200), width=2, joint="curve")
    return img


def pat_realistic(p):
    """全部入り: グリッド + 出来高 + 凡例 + 軸ラベル + 移動平均2本。"""
    img = Image.new("RGB", (W, H), (252, 252, 252))
    d = ImageDraw.Draw(img)
    box = (70, 40, W - 90, 400)
    _grid(d, box, (205, 212, 220))
    s = pd.Series(p)
    lo, hi = float(p.min()), float(p.max())
    for w_, col in ((25, (150, 90, 200)), (5, (235, 150, 40))):
        ma = np.clip(s.rolling(w_, min_periods=1).mean().to_numpy(), lo, hi)
        d.line(_xy_shared(ma, p, box), fill=col, width=2, joint="curve")
    d.line(_xy(p, box), fill=(30, 110, 200), width=2, joint="curve")
    _legend(d, box, (20, 20, 20))
    _labels(d, box, p, (60, 60, 60))
    _volume(d, p, (70, 470, W - 90, H - 50))
    return img


def _xy_shared(series: np.ndarray, ref: np.ndarray,
               box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """ref と同じ縦軸スケールで series を配置する（移動平均線用）。"""
    left, top, right, bottom = box
    lo, hi = float(ref.min()), float(ref.max())
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(left, right, len(series))
    ys = bottom - (series - lo) / (hi - lo) * (bottom - top)
    return list(zip(xs.tolist(), ys.tolist()))


PATTERNS = [
    ("1. 折れ線のみ（白背景）", pat_line_only),
    ("2. 折れ線 + 出来高パネル", pat_line_volume),
    ("3. 折れ線 + 凡例 + 軸ラベル", pat_line_text),
    ("4. 折れ線 + グリッド線", pat_line_grid),
    ("5. ローソク足 + 出来高", pat_candle_volume),
    ("6. 暗背景の折れ線", pat_dark_line),
    ("7. 折れ線 + 移動平均2本", pat_line_two_ma),
    ("8. 全部入り（実戦想定）", pat_realistic),
]


# ------------------------------------------------------ ストレステスト

def pat_photo(p):
    """スマホで撮った写真風: 傾き + 照明ムラ + JPEGノイズ。"""
    import io
    img = pat_realistic(p).rotate(1.2, resample=Image.BICUBIC, fillcolor=(250, 250, 250))
    a = np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    shade = 0.80 + 0.25 * (xx / a.shape[1]) + 0.10 * (yy / a.shape[0])
    a = np.clip(a * shade[..., None] + np.random.default_rng(1).normal(0, 4, a.shape), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, format="JPEG", quality=55)
    return Image.open(buf).convert("RGB")


def pat_hollow_candle(p):
    """中抜きローソク（陽線=白抜き）。実体が塗られていない表示。"""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 70, 40, W - 90, 400
    rng = np.random.default_rng(11)
    opens = np.concatenate(([p[0]], p[:-1]))
    span = float(p.max() - p.min())
    highs = np.maximum(opens, p) + rng.uniform(0.01, 0.05, len(p)) * span
    lows = np.minimum(opens, p) - rng.uniform(0.01, 0.05, len(p)) * span
    lo, hi = float(lows.min()), float(highs.max())
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def y_of(v):
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    xs = np.linspace(left + 6, right - 6, len(p))
    bw = max(3.0, (right - left) / len(p) * 0.65)
    for i, x in enumerate(xs):
        up = p[i] >= opens[i]
        col = (214, 64, 69) if up else (46, 154, 92)
        d.line([(x, y_of(highs[i])), (x, y_of(lows[i]))], fill=col, width=1)
        y1, y2 = sorted((y_of(opens[i]), y_of(p[i])))
        box = [x - bw / 2, y1, x + bw / 2, max(y2, y1 + 2)]
        if up:
            d.rectangle(box, outline=col, fill=(255, 255, 255), width=1)
        else:
            d.rectangle(box, fill=col)
    _volume(d, p, (70, 470, W - 90, H - 50), colored=True)
    return img


def pat_area(p):
    """面グラフ（線の下を塗りつぶす表示）。"""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 50, W - 80, H - 60)
    pts = _xy(p, box)
    d.polygon(pts + [(box[2], box[3]), (box[0], box[3])], fill=(198, 222, 246))
    d.line(pts, fill=(30, 110, 200), width=2, joint="curve")
    return img


def pat_rsi_panel(p):
    """折れ線 + 下部にRSIサブパネル（出来高ではない指標パネル）。"""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (70, 40, W - 80, 380)
    _grid(d, box, (205, 212, 220))
    d.line(_xy(p, box), fill=(30, 110, 200), width=2, joint="curve")
    s = pd.Series(p).diff()
    up = s.clip(lower=0).ewm(alpha=1 / 14).mean()
    dn = (-s.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi = (100 - 100 / (1 + up / dn)).fillna(50).to_numpy()
    rbox = (70, 450, W - 80, H - 50)
    d.rectangle(list(rbox), outline=(200, 200, 200), width=1)
    d.line(_xy_shared(rsi, np.array([0.0, 100.0]), rbox), fill=(190, 90, 190), width=2)
    return img


def pat_not_a_chart(p):
    """チャートではない画像（None を返してほしいケース）。"""
    rng = np.random.default_rng(5)
    a = rng.integers(90, 210, (H, W, 3)).astype(np.uint8)
    img = Image.fromarray(a)
    d = ImageDraw.Draw(img)
    for i in range(12):
        x, y = rng.integers(0, W - 200), rng.integers(0, H - 100)
        d.text((x, y), "メモ帳のスクリーンショット", fill=(20, 20, 20), font=_font(20))
    return img


STRESS = [
    ("写真風（傾き/照明/ノイズ）", pat_photo, True),
    ("中抜きローソク", pat_hollow_candle, True),
    ("面グラフ（塗りつぶし）", pat_area, True),
    ("折れ線 + RSIパネル", pat_rsi_panel, True),
    ("チャートでない画像", pat_not_a_chart, False),
]


# -------------------------------------------------------------- 評価

def _corr(a: np.ndarray | None, truth: np.ndarray) -> float:
    if a is None or len(a) != len(truth) or float(np.std(a)) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, truth)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=str, default=None, help="生成画像と重ね描きの保存先")
    ap.add_argument("--points", type=int, default=60)
    args = ap.parse_args()

    prices = make_prices()
    truth = np.interp(np.linspace(0, len(prices) - 1, args.points),
                      np.arange(len(prices)), prices)
    dump = Path(args.dump) if args.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for i, (name, fn) in enumerate(PATTERNS, start=1):
        img = fn(prices)
        t0 = time.perf_counter()
        detail = extract_series_detailed(img, n_points=args.points)
        dt = time.perf_counter() - t0
        new = detail["values"] if detail else None
        old = _legacy_extract(img, n_points=args.points)

        c_new, c_old = _corr(new, truth), _corr(old, truth)
        rows.append((name, c_old, c_new, detail, dt))
        if not (c_new >= TARGET):
            failures.append(name)

        if dump:
            img.save(dump / f"{i}_input.png")
            if detail:
                render_overlay(img, detail).save(dump / f"{i}_overlay.png")

    print()
    print("=" * 88)
    print("合成チャートでの抽出精度（正解系列とのピアソン相関。1.0が完全一致）")
    print("=" * 88)
    print(f"{'パターン':<30}{'旧実装':>9}{'新実装':>9}{'種別':>9}{'信頼度':>9}{'秒':>8}")
    print("-" * 88)
    for name, c_old, c_new, detail, dt in rows:
        kind = detail["kind"] if detail else "-"
        conf = f"{detail['confidence']:.2f}" if detail else "-"
        pad = 30 - sum(2 if ord(ch) > 0x2000 else 1 for ch in name)
        print(f"{name}{' ' * max(1, pad)}{c_old:>9.3f}{c_new:>9.3f}"
              f"{kind:>9}{conf:>9}{dt:>8.2f}")
    print("-" * 88)
    olds = np.array([r[1] for r in rows], dtype=float)
    news = np.array([r[2] for r in rows], dtype=float)
    print(f"{'平均':<26}{np.nanmean(olds):>9.3f}{np.nanmean(news):>9.3f}")
    print(f"{f'0.95以上の数 (n={len(rows)})':<24}"
          f"{int(np.nansum(olds >= TARGET)):>9}{int(np.nansum(news >= TARGET)):>9}")
    print("=" * 88)

    if failures:
        print(f"\n目標 {TARGET} 未達: " + " / ".join(failures))
        return 1
    print("\n全パターンで目標を達成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
