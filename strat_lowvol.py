# -*- coding: utf-8 -*-
"""低ボラティリティ／リスクベース戦略を単独で検証する。

学術的に最も再現性が高いとされるアノマリーの1つが「低ボラティリティ効果」
(Baker, Bradley & Wurgler 2011 等)。ボラティリティの低い銘柄は、リスクが
低いにもかかわらずリターンで劣らない（しばしば上回る）というもので、絶対
リターンよりも**リスク調整後リターン(CAGR/vol)で優位**が出やすい。

使えるのは日足終値だけなので、価格から計算できるリスク尺度だけを使う:
  vol  … 過去N日の日次リターン標準偏差（古典的な低ボラ）
  dsd  … ダウンサイド・デビエーション（下落した日だけの変動）
  mdd  … 過去N日の最大ドローダウン
加重は均等(equal)とボラティリティ逆数(invvol = リスクパリティ的)の2種類。
さらに「低ボラ上位グループの中で相対的に強いもの」を選ぶ二軸版も試す。

過剰適合を避けるための手順（rule_optimize.py と同じ思想）:
  1. パラメータは**訓練期(〜2012-10-03)のCAGRだけ**で選ぶ
  2. 近傍のパラメータも同程度に良いか（表面が滑らかか）を確認する
  3. 選んだ1つを検証期(2012-10-03〜)で**一度だけ**評価する
  4. 入れ替え時に往復0.2%の売買コストを回転率に比例させて引く

ルックバック期間分のウォームアップ中は「ベンチマーク（全銘柄均等分散）を
保有」とする。そうしないとルックバックが長い設定だけ現金のまま眠る期間が
長くなり、設定間の比較が不公平になるため。

※ 銘柄リストは現存銘柄のみなので生存者バイアスが残る。数値は上限値として読む。

使い方: python strat_lowvol.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mechanical_backtest import build_matrix, load_named
from momentum_classic import metrics, period_cagr
from tickers import TICKERS

SPLIT = pd.Timestamp("2012-10-03")
COST = 0.002              # 入れ替え時の往復コスト（回転率に比例）
MIN_VOL = 1e-4            # 日次vol がこれ未満は「値が動いていない＝実質売買不能」として除外
MOM_SKIP = 21             # 二軸版のモメンタム計算で直近1ヶ月を除く
N_PICKS = 20              # 現在の推奨銘柄の表示件数


# ---------------------------------------------------------------------------
# バックテスト本体
# ---------------------------------------------------------------------------
def _risk_scores(pct: np.ndarray, price: np.ndarray, day: int, lookback: int,
                 rank: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(スコア[小さいほど良い], ボラティリティ, 有効フラグ) を返す。"""
    w0 = day - lookback + 1
    win = pct[w0:day + 1]                     # lookback 行 × 全銘柄
    pwin = price[w0:day + 1]
    ok = (np.isfinite(win).all(axis=0) & np.isfinite(pwin).all(axis=0)
          & np.isfinite(price[day]) & (price[day] > 0))
    safe = np.where(ok, win, 0.0)

    vol = safe.std(axis=0)
    if rank == "vol":
        score = vol
    elif rank == "dsd":
        down = np.minimum(safe, 0.0)
        score = np.sqrt((down ** 2).mean(axis=0))
    elif rank == "mdd":
        pw = np.where(ok, pwin, 1.0)
        peak = np.maximum.accumulate(pw, axis=0)
        score = -(pw / peak - 1.0).min(axis=0)   # 正の値（大きいほどDDが深い）
    else:
        raise ValueError(f"unknown rank: {rank}")

    ok &= vol > MIN_VOL
    return np.where(ok, score, np.inf), vol, ok


def _momentum(price: np.ndarray, day: int, mom_lookback: int,
              skip: int = MOM_SKIP) -> np.ndarray:
    p_recent = price[day - skip]
    p_old = price[day - mom_lookback]
    valid = np.isfinite(p_recent) & np.isfinite(p_old) & (p_old > 0)
    return np.where(valid, p_recent / np.where(p_old > 0, p_old, np.nan) - 1.0, -np.inf)


def backtest(price_df: pd.DataFrame, *, lookback: int, n_top: int, rebal_freq: int,
             rank: str = "vol", weight: str = "equal", prefilter: int = 0,
             mom_lookback: int = 252, cost: float = COST) -> np.ndarray:
    """低ボラ系戦略の資産曲線を返す。

    lookback   : リスク尺度のルックバック営業日数
    n_top      : 保有銘柄数
    rebal_freq : リバランス間隔（営業日）
    rank       : "vol" | "dsd" | "mdd"
    weight     : "equal"（均等） | "invvol"（ボラ逆数＝リスクパリティ的）
    prefilter  : >0 なら「低リスク上位prefilter銘柄」の中からモメンタム上位n_topを取る二軸版
    """
    price = price_df.to_numpy(dtype=np.float64)
    pct = price_df.pct_change().to_numpy(dtype=np.float64)
    n_days, n_sym = price.shape

    warm = lookback + (MOM_SKIP if prefilter else 0)
    if prefilter:
        warm = max(warm, mom_lookback + 1)
    start = max(warm + 1, 2)
    rebal_set = set(range(start, n_days, rebal_freq))

    equity = np.empty(n_days)
    equity[0] = 1.0
    hold = np.array([], dtype=int)
    w_h = np.array([], dtype=float)
    warmup = True

    for day in range(1, n_days):
        r = pct[day]
        if warmup:
            rr = r[np.isfinite(r)]
            ret = float(rr.mean()) if len(rr) else 0.0
        else:
            rh = np.where(np.isfinite(r[hold]), r[hold], 0.0)
            ret = float(w_h @ rh)
            w_h = w_h * (1.0 + rh)
            s = w_h.sum()
            w_h = w_h / s if s > 0 else np.full(len(hold), 1.0 / len(hold))
        equity[day] = equity[day - 1] * (1.0 + ret)

        if day not in rebal_set:
            continue

        score, vol, ok = _risk_scores(pct, price, day, lookback, rank)
        cand = np.flatnonzero(ok)
        if len(cand) < n_top:
            continue
        order = cand[np.argsort(score[cand], kind="stable")]

        if prefilter:
            pool = order[:min(prefilter, len(order))]
            mom = _momentum(price, day, mom_lookback)
            pool = pool[np.argsort(-mom[pool], kind="stable")]
            new_hold = pool[:n_top]
        else:
            new_hold = order[:n_top]

        if weight == "invvol":
            iv = 1.0 / np.maximum(vol[new_hold], MIN_VOL)
            new_w = iv / iv.sum()
        else:
            new_w = np.full(len(new_hold), 1.0 / len(new_hold))

        # 回転率 = 入れ替わった資産割合（ウォームアップ明けは全建て替え扱い）
        if warmup:
            turnover = 1.0
        else:
            old = np.zeros(n_sym)
            old[hold] = w_h
            new = np.zeros(n_sym)
            new[new_hold] = new_w
            turnover = 0.5 * float(np.abs(new - old).sum())
        equity[day] *= (1.0 - cost * turnover)

        hold, w_h, warmup = new_hold, new_w, False

    return equity


# ---------------------------------------------------------------------------
# 探索するパラメータ（訓練期だけで選ぶ）
# ---------------------------------------------------------------------------
def build_grid() -> list[dict]:
    grid: list[dict] = []
    for rank in ("vol", "dsd", "mdd"):
        for lookback in (60, 120, 252):
            for n_top in (20, 30, 50):
                for weight in ("equal", "invvol"):
                    for rebal in (21, 63):
                        grid.append({"rank": rank, "lookback": lookback, "n_top": n_top,
                                     "weight": weight, "rebal_freq": rebal, "prefilter": 0})
    # 低ボラ × モメンタムの二軸版
    for lookback in (120, 252):
        for prefilter in (100, 150):
            for n_top in (20, 30):
                for weight in ("equal", "invvol"):
                    grid.append({"rank": "vol", "lookback": lookback, "n_top": n_top,
                                 "weight": weight, "rebal_freq": 21, "prefilter": prefilter})
    return grid


def label(cfg: dict) -> str:
    base = {"vol": "低ボラ", "dsd": "低下方リスク", "mdd": "低最大DD"}[cfg["rank"]]
    if cfg["prefilter"]:
        base = f"低ボラ{cfg['prefilter']}×モメンタム"
    w = "均等" if cfg["weight"] == "equal" else "ボラ逆数"
    return f"{base} L{cfg['lookback']} N{cfg['n_top']} {w} R{cfg['rebal_freq']}"


def neighbors(cfg: dict, grid: list[dict]) -> list[dict]:
    """1つのパラメータだけを隣の値にずらした設定（表面の滑らかさ確認用）。"""
    out = []
    for g in grid:
        diff = sum(1 for k in ("rank", "lookback", "n_top", "weight", "rebal_freq", "prefilter")
                   if g[k] != cfg[k])
        if diff == 1:
            out.append(g)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    print("=== データ読み込み ===")
    series = load_named()
    price_df = build_matrix(series)
    master = price_df.index
    print(f"日本株 {len(series)}銘柄 / {master[0].date()} 〜 {master[-1].date()} ({len(master):,}日)")

    bench = (1 + price_df.pct_change().mean(axis=1).fillna(0)).cumprod().to_numpy()
    bench = bench / bench[0]
    bm = metrics(bench, master)
    b_tr = period_cagr(bench, master, master[0], SPLIT)
    b_te = period_cagr(bench, master, SPLIT, None)
    print(f"\n【ベンチマーク】通期CAGR {bm['CAGR%']}%  訓練期 {b_tr:.2f}%  検証期 {b_te:.2f}%  "
          f"最大DD {bm['最大DD%']}%  CAGR/vol {bm['CAGR/vol']}\n")

    grid = build_grid()
    print(f"=== 訓練期(〜{SPLIT.date()})のみでグリッド探索: {len(grid)}設定 ===")
    rows = []
    equities: dict[str, np.ndarray] = {}
    for cfg in grid:
        eq = backtest(price_df, **cfg)
        equities[label(cfg)] = eq
        m = metrics(eq, master)
        rows.append({"設定": label(cfg), **cfg,
                     "訓練期CAGR%": round(period_cagr(eq, master, master[0], SPLIT), 2),
                     "検証期CAGR%": round(period_cagr(eq, master, SPLIT, None), 2),
                     "通期CAGR%": m["CAGR%"], "最終倍率": m["最終倍率"],
                     "最大DD%": m["最大DD%"], "年率vol%": m["年率vol%"],
                     "CAGR/vol": m["CAGR/vol"],
                     "対ベンチ倍率": round(float(eq[-1] / bench[-1]), 2)})
    df = pd.DataFrame(rows)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 240)

    # ---- 訓練期だけで並べる（検証期の列はここでは見ない）----
    train_view = df.sort_values("訓練期CAGR%", ascending=False)
    print("\n【訓練期CAGRの上位15設定】※選択はこの列だけで行う")
    print(train_view.head(15)[["設定", "訓練期CAGR%", "最大DD%", "CAGR/vol"]].to_string(index=False))

    # ---- 近傍の滑らかさを見て採用を1つ決める ----
    best = train_view.iloc[0]
    best_cfg = next(g for g in grid if label(g) == best["設定"])
    nb = neighbors(best_cfg, grid)
    nb_tr = [float(df.loc[df["設定"] == label(g), "訓練期CAGR%"].iloc[0]) for g in nb]
    print(f"\n訓練期1位: {best['設定']}  訓練期CAGR {best['訓練期CAGR%']}%")
    print(f"  近傍{len(nb)}設定の訓練期CAGR: 中央値 {np.median(nb_tr):.2f}% / "
          f"最小 {min(nb_tr):.2f}% / 最大 {max(nb_tr):.2f}%")
    if np.median(nb_tr) < b_tr:
        print("  ※近傍の中央値がベンチマークを下回る＝一点だけ突出した偶然の可能性あり")
    print(f"\n=== 採用（訓練期のみで決定）: {best['設定']} ===")

    # ---- 採用した1設定を検証期で1度だけ評価する ----
    eq = equities[best["設定"]]
    m = metrics(eq, master)
    tr = period_cagr(eq, master, master[0], SPLIT)
    te = period_cagr(eq, master, SPLIT, None)
    print(f"  訓練期CAGR {tr:.2f}%  (ベンチ {b_tr:.2f}%)")
    print(f"  検証期CAGR {te:.2f}%  (ベンチ {b_te:.2f}%)")
    print(f"  通期CAGR {m['CAGR%']}%  最大DD {m['最大DD%']}%  年率vol {m['年率vol%']}%  "
          f"CAGR/vol {m['CAGR/vol']} (ベンチ {bm['CAGR/vol']})  "
          f"対ベンチ倍率 {float(eq[-1] / bench[-1]):.2f}倍")
    passed = (tr > b_tr) and (te > b_te)
    strong = passed and (m["CAGR/vol"] >= 0.80)
    print(f"  判定: {'強い合格' if strong else ('合格' if passed else '不合格')}"
          f"（訓練期>{b_tr:.2f}% かつ 検証期>{b_te:.2f}%）")
    if not passed and m["CAGR/vol"] > bm["CAGR/vol"]:
        print(f"  ※絶対リターンでは不合格だが、リスク効率は優位: "
              f"CAGR/vol {m['CAGR/vol']} > ベンチ {bm['CAGR/vol']}")

    out = Path(__file__).parent / "data"
    df.insert(0, "ベンチ訓練期CAGR%", round(b_tr, 2))
    df.insert(1, "ベンチ検証期CAGR%", round(b_te, 2))
    df["採用"] = np.where(df["設定"] == best["設定"], "○", "")
    df.to_csv(out / "strat_lowvol.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"ベンチマーク": bench, best["設定"]: eq},
                 index=master).to_csv(out / "strat_lowvol_curves.csv", encoding="utf-8-sig")

    # ---- 現在（最新データ）での推奨銘柄 ----
    print(f"\n=== 現在（{master[-1].date()}）でこの条件を適用した場合の上位{N_PICKS}銘柄 ===")
    price = price_df.to_numpy(dtype=np.float64)
    day = len(master) - 1
    score, vol, ok = _risk_scores(price_df.pct_change().to_numpy(dtype=np.float64),
                                  price, day, best_cfg["lookback"], best_cfg["rank"])
    cand = np.flatnonzero(ok)
    order = cand[np.argsort(score[cand], kind="stable")]
    if best_cfg["prefilter"]:
        pool = order[:min(best_cfg["prefilter"], len(order))]
        mom = _momentum(price, day, 252)
        order = pool[np.argsort(-mom[pool], kind="stable")]
    sel = order[:best_cfg["n_top"]]
    if best_cfg["weight"] == "invvol":
        iv = 1.0 / np.maximum(vol[sel], MIN_VOL)
        wts = iv / iv.sum()
    else:
        wts = np.full(len(sel), 1.0 / len(sel))

    # mdd は「期間中の最大下落率」、vol/dsd は日次なので年率換算して表示する
    score_col = {"vol": "年率vol%", "dsd": "年率下方偏差%", "mdd": f"最大DD%({best_cfg['lookback']}日)"}[
        best_cfg["rank"]]
    scale = 100.0 if best_cfg["rank"] == "mdd" else float(np.sqrt(250)) * 100.0
    picks = []
    for i, w in zip(sel[:N_PICKS], wts[:N_PICKS]):
        sym = price_df.columns[i]
        name_jp, _group, sector = TICKERS.get(sym, (sym, "?", "?"))
        picks.append({"コード": sym, "銘柄名": name_jp, "業種": sector,
                      "終値": round(float(price[day, i]), 1),
                      score_col: round(float(score[i]) * scale, 1),
                      "配分%": round(float(w) * 100, 2)})
    pdf = pd.DataFrame(picks)
    print(pdf.to_string(index=False))
    pdf.to_csv(out / "strat_lowvol_picks.csv", index=False, encoding="utf-8-sig")
    print("\ndata/strat_lowvol.csv, strat_lowvol_curves.csv, strat_lowvol_picks.csv に保存")


if __name__ == "__main__":
    main()
