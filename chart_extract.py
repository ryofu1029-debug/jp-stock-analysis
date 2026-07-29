# -*- coding: utf-8 -*-
"""チャート画像から価格の形状（1本の曲線）を抽出する。

証券アプリのスクリーンショットやスマホで撮った写真を想定し、次の手順で
「価格パネルだけ」を取り出して1本の曲線に変換する。

1. 背景色を推定して前景マスクを作る
2. 画像の大部分に伸びる単色の直線（グリッド線・軸）を前景から除く
3. ごく小さい連結成分（ノイズ）を落とす
4. 前景が無い水平な帯で画像を上下のパネルに分割し、価格パネルだけを残す
   （出来高・RSI・MACD のサブパネルを捨てるのが目的）
5. パネル内の文字・凡例らしい塊を連結成分の形から落とす
6. 列ごとの候補（前景の連続区間の中心）を動的計画法でつなぎ、1本の曲線を追う

ローソク足は前景色が赤系/緑（青）系の2クラスタに割れることで判定し、
ヒゲを含む全体ではなく実体（横に太い部分）の中心を採る。

抽出結果は pattern_match 側で z スコア正規化されてから相関で照合されるので、
絶対値のスケールは不要で「形（相対的な上下）」だけ合っていればよい。
うまく追えなかったと判断した場合は None を返す（呼び出し側で再撮影を促す前提）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

MAX_WIDTH = 800          # 処理を軽くするための上限（横）
MAX_HEIGHT = 900         # 同上（縦。縦長のスマホ画面対策）
MIN_COVERAGE = 0.30      # これ未満しか追えなければ抽出失敗とみなす

_NEIGH8 = np.ones((3, 3), dtype=bool)
_HORIZ3 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)


# ---------------------------------------------------------------- 下ごしらえ

def _to_array(img: Image.Image) -> tuple[np.ndarray, float]:
    """RGB配列と「作業座標→元画像座標」の倍率を返す。"""
    im = img.convert("RGB")
    scale = min(1.0, MAX_WIDTH / im.width, MAX_HEIGHT / im.height)
    if scale < 1.0:
        im = im.resize((max(2, int(im.width * scale)), max(2, int(im.height * scale))),
                       Image.BILINEAR)
    return np.asarray(im, dtype=np.float32), img.width / im.width


def _runs(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1次元bool配列の連続True区間を (開始, 終了+1) の配列ペアで返す。"""
    d = np.flatnonzero(np.diff(np.concatenate(([0], b.astype(np.int8), [0]))))
    return d[::2], d[1::2]


def _foreground(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """背景色を推定し、そこから離れた画素を前景とするマスクを返す。

    背景は「最頻の色（16段階に量子化）」で推定する。白/黒どちらのテーマでも、
    また薄い色地の画像でも同じ理屈で扱える。前景率が極端になった場合は
    しきい値を振り直し、それでもだめなら局所コントラスト（写真の照明ムラ対策）で拾う。
    """
    q = (rgb / 16.0).astype(np.int32).clip(0, 15)
    key = (q[..., 0] * 16 + q[..., 1]) * 16 + q[..., 2]
    top = int(np.bincount(key.ravel(), minlength=16 ** 3).argmax())
    bg = rgb[key == top].reshape(-1, 3).mean(axis=0)

    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    for th in (60.0, 45.0, 32.0, 22.0, 85.0, 120.0):
        fg = dist > th
        if 0.004 <= float(fg.mean()) <= 0.30:
            return fg, bg

    gray = rgb.mean(axis=2)
    size = max(15, min(rgb.shape[0], rgb.shape[1]) // 8)
    fg = np.abs(gray - ndimage.uniform_filter(gray, size=size)) > 12
    return fg, bg


# ------------------------------------------------ グリッド線・文字の除去

def _strip_lines(rgb: np.ndarray, fg: np.ndarray, min_frac: float,
                 tol: float = 45.0, max_gap: int = 12) -> np.ndarray:
    """行方向に「幅の大部分にわたって伸びる単色の線」を前景から消す（破壊的）。

    ローソク足に何度も遮られるグリッド線も拾えるよう、連続性は
    「最頻色の画素が端から端まで、細かい途切れだけで並んでいるか」で見る。
    色が一致する画素しか消さないので、交差しているチャート本体は残る。
    列方向は転置して呼ぶこと。
    """
    w = fg.shape[1]
    need = min_frac * w
    for y in np.flatnonzero(fg.sum(axis=1) >= need):
        cols = np.flatnonzero(fg[y])
        px = rgb[y, cols]
        q = (px / 32.0).astype(np.int32).clip(0, 7)
        key = (q[:, 0] * 8 + q[:, 1]) * 8 + q[:, 2]
        mode = int(np.bincount(key, minlength=512).argmax())
        cen = px[key == mode].mean(axis=0)
        xs = cols[np.sqrt(((px - cen) ** 2).sum(axis=1)) <= tol]
        if len(xs) < need or xs[-1] - xs[0] < need:
            continue
        if float(np.percentile(np.diff(xs), 97)) > max_gap:
            continue  # 途切れが大きい＝直線ではない
        fg[y, xs] = False
    return fg


def _remove_grid(rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """水平・垂直のグリッド線と軸線を前景から除いたマスクを返す。"""
    fg = _strip_lines(rgb, fg.copy(), min_frac=0.50)
    fg_t = _strip_lines(np.ascontiguousarray(rgb.transpose(1, 0, 2)),
                        np.ascontiguousarray(fg.T), min_frac=0.65)
    return np.ascontiguousarray(fg_t.T)


def _drop_specks(fg: np.ndarray, min_area: int = 6) -> np.ndarray:
    """面積の小さすぎる連結成分（圧縮ノイズ・点）を落とす。"""
    lab, n = ndimage.label(fg, structure=_NEIGH8)
    if n == 0:
        return fg
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return fg & (sizes[lab] >= min_area)


def _drop_text_blobs(fg: np.ndarray, candle: bool) -> np.ndarray:
    """文字・凡例・マーカーらしい連結成分を落とす。

    折れ線モードでは「横に長く伸びていない成分」を捨てる（チャート本体は
    横に伸びた大きな成分になるはず）。ローソク足モードでは1本1本が独立した
    小さい成分なので、代わりに「縦長でない塊」＝文字だけを捨てる。
    """
    lab, n = ndimage.label(fg, structure=_NEIGH8)
    if n == 0:
        return fg
    h, w = fg.shape
    sizes = np.bincount(lab.ravel())
    drop = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        fill = sizes[i] / float(bh * bw)
        if candle:
            # 文字は横長〜正方形、ローソクはヒゲを含めて必ず縦長になる
            if bh < 1.2 * bw and bw < 0.20 * w:
                drop[i] = True
            continue
        if bw < max(10, 0.05 * w) and bh < 0.20 * h:
            drop[i] = True                      # 短い断片＝文字・目盛り
        elif fill > 0.40 and bw < 0.25 * w and bh < 0.12 * h:
            drop[i] = True                      # 詰まった小さい塊＝文字
    drop[0] = True
    return fg & ~drop[lab]


# ---------------------------------------------------------- パネル分割

def _find_panels(fg: np.ndarray) -> list[tuple[int, int]]:
    """前景がほぼ無い水平な帯を区切りに、画像を上下のパネル（行区間）へ分ける。"""
    h, w = fg.shape
    min_gap = max(4, int(h * 0.02))
    min_h = max(8, int(h * 0.05))
    s, e = _runs(fg.any(axis=1))
    if len(s) == 0:
        return []
    bands = [[int(s[0]), int(e[0])]]
    for a, b in zip(s[1:], e[1:]):
        if a - bands[-1][1] < min_gap:
            bands[-1][1] = int(b)   # 隙間が狭ければ同じパネル
        else:
            bands.append([int(a), int(b)])
    return [(a, b) for a, b in bands if b - a >= min_h]


def _pick_price_panel(fg: np.ndarray, panels: list[tuple[int, int]]
                      ) -> tuple[int, int, bool] | None:
    """価格パネル（通常は一番上で一番背の高いパネル）を選ぶ。

    戻り値は (top, bottom, 下端塗りつぶし型か)。出来高パネルのように
    「ほぼ全列が帯の下端に接している」ものは価格パネルとして不利に採点する。
    面グラフ（価格の下を塗るタイプ）も同じ形になるため、除外はせず
    フラグだけ立てて後段で上端を追うようにする。
    """
    scored = []
    for top, bot in panels:
        sub = fg[top:bot]
        cols = sub.any(axis=0)
        if float(cols.mean()) < 0.35:
            continue  # 横に広がっていない＝凡例やラベルの塊
        touch = float((sub[-3:].any(axis=0) & cols).sum()) / max(1, int(cols.sum()))
        filled = touch > 0.70
        scored.append((top, bot, (bot - top) * (0.55 if filled else 1.0), filled))
    if not scored:
        return None
    best = max(s[2] for s in scored)
    for top, bot, score, filled in scored:      # 上にあるものを優先
        if score >= 0.75 * best:
            return top, bot, filled
    return None


# ------------------------------------------------------------ 曲線追跡

def _trace_curve(mask: np.ndarray, thick_penalty: float = 0.5,
                 top_edge: bool = False, jump_w: float = 1.0,
                 max_runs: int = 8) -> tuple[np.ndarray, np.ndarray] | None:
    """列ごとの候補を動的計画法（Viterbi的な経路探索）で繋ぎ1本の曲線を返す。

    候補は各列の前景の連続区間の中心（top_edge=True なら上端）。
    列間のジャンプ量にペナルティを与えるため、凡例の文字や別の線に
    引っ張られにくい。戻り値は (観測できた列, そのy) の組。
    """
    h, _ = mask.shape
    colsum = mask.sum(axis=0)
    cols = np.flatnonzero((colsum > 0) & (colsum <= 0.60 * h))
    if len(cols) < 4:
        return None

    xs: list[int] = []
    cand_y: list[np.ndarray] = []
    cand_len: list[np.ndarray] = []
    for x in cols:
        s, e = _runs(mask[:, x])
        lens = (e - s).astype(np.float64)
        keep = np.argsort(lens)[::-1][:max_runs]
        s, e, lens = s[keep], e[keep], lens[keep]
        xs.append(int(x))
        cand_y.append(s.astype(np.float64) if top_edge else (s + e - 1) / 2.0)
        cand_len.append(lens)

    typ = float(np.median(np.concatenate(cand_len)))
    thick = max(5.0, 2.5 * typ)   # これより太い区間は文字の塊などを疑う
    emit = [thick_penalty * np.maximum(0.0, L - thick) for L in cand_len]

    cost = emit[0].copy()
    backs: list[np.ndarray] = []
    for t in range(1, len(xs)):
        gap = max(1, xs[t] - xs[t - 1])
        d = np.abs(cand_y[t][:, None] - cand_y[t - 1][None, :])
        tot = cost[None, :] + jump_w * d / np.sqrt(gap)
        j = tot.argmin(axis=1)
        cost = tot[np.arange(len(j)), j] + emit[t]
        backs.append(j)

    ys = np.empty(len(xs))
    i = int(cost.argmin())
    for t in range(len(xs) - 1, -1, -1):
        ys[t] = cand_y[t][i]
        if t:
            i = int(backs[t - 1][i])
    return np.asarray(xs, dtype=np.int64), ys


# ------------------------------------------------------ 色クラスタ／ローソク

def _color_groups(rgb: np.ndarray, mask: np.ndarray, bg: np.ndarray | None = None,
                  max_groups: int = 4, min_share: float = 0.10,
                  min_sep: float = 70.0, max_assign: float = 100.0
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """前景を色でクラスタ分けし [(代表色, マスク), ...] を返す（1色なら空）。

    アンチエイリアスの中間色に代表色を持っていかれないよう、色を数えるのは
    背景から十分離れた「芯」の画素だけにする。どの代表色からも遠い画素は
    どのグループにも入れない。
    """
    pix = rgb[mask]
    n = len(pix)
    if n < 50:
        return []
    core = pix
    if bg is not None:
        d = np.sqrt(((pix - bg) ** 2).sum(axis=1))
        sel = d >= 0.55 * float(np.percentile(d, 90))
        if int(sel.sum()) >= 50:
            core = pix[sel]

    q = (core / 48.0).astype(np.int32).clip(0, 5)
    key = (q[:, 0] * 6 + q[:, 1]) * 6 + q[:, 2]
    counts = np.bincount(key, minlength=216)
    centers: list[np.ndarray] = []
    for k in np.argsort(counts)[::-1]:
        if counts[k] < min_share * len(core):
            break
        c = core[key == k].mean(axis=0)
        if all(float(np.linalg.norm(c - o)) >= min_sep for o in centers):
            centers.append(c)
        if len(centers) >= max_groups:
            break
    if len(centers) < 2:
        return []

    cen = np.stack(centers)
    dist = np.linalg.norm(pix[:, None, :] - cen[None, :, :], axis=2)
    lab = dist.argmin(axis=1)
    lab[dist.min(axis=1) > max_assign] = -1
    flat = np.flatnonzero(mask.ravel())
    out = []
    for i in range(len(cen)):
        m = np.zeros(mask.size, dtype=bool)
        m[flat[lab == i]] = True
        out.append((cen[i], m.reshape(mask.shape)))
    return out


def _is_warm(c: np.ndarray) -> bool:
    """赤系か。"""
    return bool(c[0] >= c[1] + 25 and c[0] >= c[2] + 25)


def _is_cool(c: np.ndarray) -> bool:
    """緑系または青系か。"""
    return bool(c[1] >= c[0] + 25 or c[2] >= c[0] + 25)


def _detect_candles(rgb: np.ndarray, groups: list[tuple[np.ndarray, np.ndarray]]
                    ) -> np.ndarray | None:
    """ローソク足なら赤系＋緑（青）系を合成したマスクを、違えば None を返す。"""
    warm = [m for c, m in groups if _is_warm(c)]
    cool = [m for c, m in groups if _is_cool(c)]
    if not warm or not cool:
        return None

    out = np.zeros(rgb.shape[:2], dtype=bool)
    for m in warm + cool:
        out |= m
    out &= (rgb.max(axis=2) - rgb.min(axis=2)) > 30.0   # 灰色の文字・目盛りを除く
    if not out.any():
        return None

    h = out.shape[0]
    cols = out.any(axis=0)
    first = np.argmax(out, axis=0)
    last = h - 1 - np.argmax(out[::-1], axis=0)
    ext = (last - first)[cols] + 1.0
    dens = out.sum(axis=0)[cols] / ext
    # ローソクは高値〜安値へ縦に伸び、かつその区間がほぼ埋まっている。
    # 色違いの線が複数本あるだけの折れ線チャートとはここで分かれる。
    if float(np.median(ext)) < max(5.0, 0.015 * h) or float(np.median(dens)) < 0.50:
        return None
    return out


def _body_mask(mask: np.ndarray) -> np.ndarray:
    """ローソク足の実体（横に太い部分）だけを残したマスクを返す。

    行方向にだけ連結した区間の長さ＝その画素の「横の太さ」。
    ヒゲは1〜2px、実体はローソク幅そのものになるので閾値で分けられる。
    実体をうまく取れない（中抜きのローソク等）場合は元のマスクを返す。
    """
    closed = ndimage.binary_closing(mask, structure=_HORIZ3)
    lab, n = ndimage.label(closed, structure=_HORIZ3)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    runlen = sizes[lab]
    vals = runlen[closed]
    bw = float(np.percentile(vals, 80))
    if bw < 3.0:
        return mask
    body = closed & (runlen >= max(2.0, 0.6 * bw))
    if body.any(axis=0).sum() < 0.6 * mask.any(axis=0).sum():
        return mask
    return body


# ------------------------------------------------------------------ 本体

def _pick_trace(rgb: np.ndarray, mask: np.ndarray, panel_w: int,
                bg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, str] | None:
    """価格パネルのマスクから最も「本命らしい」1本の曲線を追う。"""
    candle = _detect_candles(rgb, _color_groups(rgb, mask, bg))
    if candle is not None:
        body = _body_mask(_drop_text_blobs(candle, candle=True))
        tr = _trace_curve(body, thick_penalty=0.0)
        if tr is not None:
            return tr[0], tr[1], "candle"

    clean = _drop_text_blobs(mask, candle=False)
    if not clean.any():
        clean = mask

    # 移動平均線などで色が分かれている場合は、線ごとに追って本命を選ぶ。
    # 横に広く追えたものを優先し、同程度なら上下によく動く方＝実際の株価の線を採る
    # （移動平均は定義上、元の株価より必ず滑らかになる）。
    best = None
    best_key = (-1, -1.0)
    for _, gm in _color_groups(rgb, clean, bg):
        tr = _trace_curve(gm)
        if tr is None:
            continue
        cov = len(tr[0]) / float(panel_w)
        tv = float(np.abs(np.diff(tr[1])).sum()) / max(1, len(tr[1]))
        key = (round(cov * 20), tv)
        if cov >= 0.50 and key > best_key:
            best, best_key = tr, key
    if best is None:
        best = _trace_curve(clean)
    if best is None:
        return None
    return best[0], best[1], "line"


def extract_series_detailed(img: Image.Image, n_points: int = 60) -> dict | None:
    """抽出の詳細を返す。失敗時 None。

    戻り値:
        values     : n_points 点にリサンプルした系列（上が高値。スケールは無意味）
        ys_px      : 元画像座標での曲線のy（panel_box の左端〜右端に等間隔で対応）
        panel_box  : (left, top, right, bottom) 元画像座標での価格パネル
        kind       : "line" | "candle"
        confidence : 0〜1 の信頼度（横に追えた割合と曲線の素直さから算出）
    """
    rgb, scale = _to_array(img)
    h, w = rgb.shape[:2]

    fg, bg = _foreground(rgb)
    if not fg.any():
        return None
    fg = _drop_specks(_remove_grid(rgb, fg))
    if not fg.any():
        return None

    panels = _find_panels(fg)
    if not panels:
        return None
    picked = _pick_price_panel(fg, panels)
    if picked is None:
        return None
    top, bot, filled = picked

    sub = fg[top:bot]
    cols = np.flatnonzero(sub.any(axis=0))
    x_lo, x_hi = int(cols[0]), int(cols[-1])
    sub = sub[:, x_lo:x_hi + 1]
    sub_rgb = rgb[top:bot, x_lo:x_hi + 1]
    panel_w = sub.shape[1]

    if filled:
        # 面グラフ（価格の下を塗りつぶす表示）は上端が価格
        tr = _trace_curve(sub, top_edge=True)
        traced = (tr[0], tr[1], "line") if tr else None
    else:
        traced = _pick_trace(sub_rgb, sub, panel_w, bg)
    if traced is None:
        return None
    xs, ys, kind = traced

    x0, x1 = int(xs[0]), int(xs[-1])
    span = x1 - x0 + 1
    if span < 8:
        return None
    # ローソク足の隙間の列など「そもそも前景が無い列」は減点しない
    avail = max(1, int(sub.any(axis=0)[x0:x1 + 1].sum()))
    fill_ratio = min(1.0, len(xs) / float(avail))
    coverage = (span / float(panel_w)) * fill_ratio
    if coverage < MIN_COVERAGE:
        return None

    grid = np.arange(x0, x1 + 1)
    ys_full = np.interp(grid, xs, ys)
    ys_full = pd.Series(ys_full).rolling(3, center=True, min_periods=1).median().to_numpy()

    # 隣り合う観測点が素直に繋がっているほど信頼できるとみなす
    if len(xs) > 1:
        step = np.abs(np.diff(ys)) / np.maximum(1.0, np.diff(xs))
        continuity = float((step <= max(2.0, 0.06 * (bot - top))).mean())
    else:
        continuity = 0.0
    confidence = float(np.clip(coverage * (0.4 + 0.6 * continuity), 0.0, 1.0))

    vals = -ys_full  # 画像は下方向が正なので反転（上=高値）
    idx = np.linspace(0, len(vals) - 1, n_points)
    values = np.interp(idx, np.arange(len(vals)), vals)

    return {
        "values": values,
        "ys_px": (ys_full + top) * scale,
        "panel_box": (int((x_lo + x0) * scale), int(top * scale),
                      int((x_lo + x1) * scale), int(bot * scale)),
        "kind": kind,
        "confidence": confidence,
    }


def extract_series_from_image(img: Image.Image, n_points: int = 60) -> np.ndarray | None:
    """画像から価格曲線を抽出し、n_points 点にリサンプルして返す。失敗時 None。"""
    detail = extract_series_detailed(img, n_points=n_points)
    return None if detail is None else detail["values"]


def render_overlay(img: Image.Image, detail: dict) -> Image.Image:
    """元画像に「検出した価格パネルの枠」と「抽出した曲線」を描き重ねて返す。

    ユーザーが抽出結果を目視確認するためのもの。detail は
    extract_series_detailed() の戻り値をそのまま渡す。
    """
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    left, top, right, bottom = detail["panel_box"]
    lw = max(2, out.width // 400)
    draw.rectangle([left, top, right, bottom], outline=(0, 170, 255), width=lw)

    ys = np.asarray(detail["ys_px"], dtype=float)
    if len(ys) >= 2:
        xs = np.linspace(left, right, len(ys))
        pts = [(float(x), float(np.clip(y, 0, out.height - 1)))
               for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(pts) >= 2:
            # 明背景・暗背景どちらでも見えるよう白で縁取ってから本線を引く
            draw.line(pts, fill=(255, 255, 255), width=lw * 3, joint="curve")
            draw.line(pts, fill=(255, 60, 0), width=lw, joint="curve")
    return out
