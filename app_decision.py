# -*- coding: utf-8 -*-
"""確率を起点に投資判断までつなげる独立版 Streamlit UI。"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

from chart_extract import extract_series_detailed, render_overlay
from decision_engine import build_decision
from indicators import add_all_indicators
from pattern_match_decision import (
    TIMEFRAME_CONFIG,
    build_library,
    find_similar,
    prepare_close,
    timeframe_config,
)
from search import filter_df
from tickers import TICKERS
from walkforward import walk_forward_validate

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SUMMARY_FILE = BASE_DIR / "data" / "summary.csv"
SEARCH_PLACEHOLDER = "コード・社名・業種（例: 6857 / アドバンテスト / 半導体）"

st.set_page_config(page_title="AI株式判断スカウター", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6eaf0;
        border-radius: 14px; padding: 15px 16px; box-shadow: 0 2px 8px rgba(25,42,70,.04);}
    [data-testid="stMetricLabel"] {font-weight: 650; color: #536174;}
    .decision-card {border: 1px solid #dbe5f4; border-left: 6px solid #1769e0;
        border-radius: 14px; padding: 18px 22px; background: #f7faff; margin: 8px 0 18px;}
    .decision-title {font-size: 1.55rem; font-weight: 750; color: #10213b;}
    .decision-note {color: #40516a; margin-top: 6px;}
    .eyebrow {font-size: .82rem; font-weight: 750; color: #1769e0; letter-spacing: .04em;}
    .small-muted {color: #69778a; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_FILE)
    if "bb_signal" in summary.columns:
        summary["bb_signal"] = summary["bb_signal"].fillna("")
    return summary


@st.cache_data(ttl=3600)
def load_stock(symbol: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{symbol}.csv", index_col="Date", parse_dates=True)


@st.cache_resource(show_spinner="過去の類似パターンを準備しています…")
def get_library(timeframe: str, window: int) -> dict:
    return build_library(window=window, timeframe=timeframe)


@st.cache_data(show_spinner="過去時点に巻き戻して予測を採点しています…")
def get_walkforward(symbol: str, timeframe: str, window: int, horizon: int, k: int) -> dict:
    close = prepare_close(load_stock(symbol), timeframe)
    return walk_forward_validate(
        close, get_library(timeframe, window), window=window, horizon=horizon, symbol=symbol, k=k
    )


def _check_password() -> bool:
    try:
        password = st.secrets["APP_PASSWORD"]
    except Exception:
        return True
    if st.session_state.get("auth_ok"):
        return True
    entered = st.text_input("パスワード", type="password")
    if entered == password:
        st.session_state["auth_ok"] = True
        st.rerun()
    elif entered:
        st.error("パスワードが違います")
    return False


def fmt_price(value: float, symbol: str) -> str:
    return f"¥{value:,.0f}" if symbol.endswith(".T") else f"${value:,.2f}"


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "日足":
        return df.copy()
    columns = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    weekly = df[list(columns)].resample("W-FRI").agg(columns).dropna(subset=["Close"])
    return add_all_indicators(weekly)


def probability_figure(decision: dict) -> go.Figure:
    labels = ["下落", "横ばい", "上昇"]
    colors = ["#e34f4f", "#a7b0bd", "#16885f"]
    values = [decision["probabilities"][label] * 100 for label in labels]
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=235,
        margin=dict(t=10, b=20, l=15, r=45),
        xaxis=dict(range=[0, max(100, max(values) + 12)], title="確率", ticksuffix="%"),
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def fan_figure(result: dict) -> go.Figure:
    paths = result["paths_pct"]
    x = np.arange(1, len(paths["p50"]) + 1)
    fig = go.Figure()
    for high, low, color, label in (
        ("p90", "p10", "rgba(23,105,224,.12)", "10〜90%"),
        ("p75", "p25", "rgba(23,105,224,.25)", "25〜75%"),
    ):
        fig.add_trace(go.Scatter(x=x, y=(paths[high] - 1) * 100, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=(paths[low] - 1) * 100, line=dict(width=0),
                                 fill="tonexty", fillcolor=color, name=label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=(paths["p50"] - 1) * 100,
                             line=dict(color="#1769e0", width=3), name="中央値"))
    fig.add_hline(y=0, line_dash="dot", line_color="#8994a3")
    fig.update_layout(
        title=f"類似局面のその後{result['horizon']}{result['unit']}",
        xaxis_title=result["unit"],
        yaxis_title="現在からの変化率",
        yaxis_ticksuffix="%",
        height=390,
        margin=dict(t=55, b=35, l=10, r=10),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def price_figure(df: pd.DataFrame, timeframe: str, period_points: int) -> go.Figure:
    frame = resample_ohlcv(df, timeframe).iloc[-period_points:]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
        name="株価", increasing_line_color="#d94747", decreasing_line_color="#3474cc",
    ))
    for column, color in (("SMA20", "#e39a28"), ("SMA50", "#1769e0")):
        if column in frame:
            fig.add_trace(go.Scatter(x=frame.index, y=frame[column], name=column,
                                     line=dict(color=color, width=1.5)))
    fig.update_layout(
        height=440,
        xaxis_rangeslider_visible=False,
        margin=dict(t=20, b=25, l=10, r=10),
        legend=dict(orientation="h", y=1.04),
        yaxis_title="株価",
    )
    return fig


def comparison_figure(query: pd.Series, result: dict, window: int) -> go.Figure:
    top = result["matches"].head(3)
    cols = 1 + len(top)
    titles = ["現在"] + [
        f"{TICKERS.get(row.symbol, (row.symbol,))[0]}<br>{row.end_date} / {row.forward_return_pct:+.1f}%"
        for row in top.itertuples()
    ]
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    current = query.iloc[-window:].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(y=(current / current[0] - 1) * 100, line=dict(color="#1769e0", width=3),
                             showlegend=False), row=1, col=1)
    for column, row in enumerate(top.itertuples(), start=2):
        frame = load_stock(row.symbol)
        close = prepare_close(frame, result["timeframe"])
        end = pd.Timestamp(row.end_date)
        past = close.loc[:end].iloc[-window:]
        future = close.loc[close.index > end].iloc[:result["horizon"]]
        if len(past) < 2:
            continue
        past_pct = (past / past.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(y=past_pct, line=dict(color="#1769e0", width=2.2),
                                 showlegend=False), row=1, col=column)
        if len(future):
            future_pct = (future / past.iloc[0] - 1) * 100
            x_future = np.arange(len(past), len(past) + len(future))
            fig.add_trace(go.Scatter(x=x_future, y=future_pct, line=dict(color="#16a06d", width=2, dash="dash"),
                                     showlegend=False), row=1, col=column)
    fig.update_layout(height=330, margin=dict(t=70, b=20, l=10, r=10), hovermode="x unified")
    fig.update_yaxes(ticksuffix="%")
    return fig


def render_decision(decision: dict) -> None:
    scenario_color = {"上昇": "#16885f", "横ばい": "#6d7888", "下落": "#d94747"}[decision["most_likely"]]
    st.markdown('<div class="eyebrow">PROBABILITY FIRST</div>', unsafe_allow_html=True)
    st.subheader(f"最も可能性が高い展開：{decision['most_likely']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{decision['horizon_label']}の最有力", decision["most_likely"],
              f"確率 {decision['most_likely_probability'] * 100:.0f}%", delta_color="off")
    c2.metric("リターン中央値", f"{decision['median_return'] * 100:+.1f}%")
    c3.metric("想定下値（10%点）", f"{decision['downside'] * 100:+.1f}%")
    c4.metric("想定上値（90%点）", f"{decision['upside'] * 100:+.1f}%")
    c5.metric("標本信頼度", decision["confidence_label"], f"{decision['confidence_score']}/100", delta_color="off")
    st.markdown(
        f"""
        <div class="decision-card">
          <div class="small-muted">参考投資判断　スコア {decision['score']}/100</div>
          <div class="decision-title">{decision['label']}</div>
          <div class="decision-note">{decision['action']}</div>
          <div class="decision-note">期待リターン <b>{decision['expected_return'] * 100:+.1f}%</b>　
          リスクリワード <b>{decision['reward_risk']:.2f}倍</b>　
          類似局面 <b>{decision['sample_size']}件</b>　平均類似度 <b>{decision['average_similarity'] * 100:.0f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"横ばいは±{decision['flat_band'] * 100:.1f}%以内として集計。表示色: {scenario_color}")


def render_analysis(symbol: str, timeframe: str, span: str, horizon: int, k: int) -> None:
    cfg = timeframe_config(timeframe)
    window = cfg["spans"][span]
    df = load_stock(symbol)
    query = prepare_close(df, timeframe)
    if len(query) < window:
        st.error("選択期間に必要な株価データが足りません。分析期間を短くしてください。")
        return
    result = find_similar(
        query.iloc[-window:].to_numpy(),
        get_library(timeframe, window),
        k=k,
        query_symbol=symbol,
        query_start_date=query.index[-window],
    )
    if result is None:
        st.error("基準以上に似た過去局面を十分に見つけられませんでした。分析期間を変えてください。")
        return
    decision = build_decision(result, horizon)
    render_decision(decision)

    overview, similar, technical, validation = st.tabs(["判断の要点", "類似チャート", "テクニカル", "検証・根拠"])
    with overview:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### シナリオ別確率")
            st.plotly_chart(probability_figure(decision), width="stretch")
        with right:
            st.markdown("#### 判断の読み方")
            st.write(
                f"過去の類似局面では、{decision['horizon_label']}に株価が"
                f"**{decision['most_likely']}**となるケースが最も多く、確率は"
                f"**{decision['most_likely_probability'] * 100:.0f}%**でした。"
            )
            st.write(
                f"一方、悪い方から10%に位置するケースでは**{decision['downside'] * 100:+.1f}%**。"
                "この下振れに耐えられるかを、購入量と撤退条件に反映します。"
            )
        display_points = 120 if timeframe == "日足" else 104
        st.plotly_chart(price_figure(df, timeframe, display_points), width="stretch")

    with similar:
        st.markdown("#### 現在と最も似ていた過去局面")
        st.caption("青線が照合した形、緑の破線がその後の実際の値動きです。")
        st.plotly_chart(comparison_figure(query, result, window), width="stretch")
        st.plotly_chart(fan_figure(result), width="stretch")
        table = result["matches"].head(15).copy()
        table.insert(1, "企業名", [TICKERS.get(s, (s,))[0] for s in table["symbol"]])
        table.columns = ["コード", "企業名", "類似日", "類似度%", f"{result['horizon']}{result['unit']}後%"]
        st.dataframe(table, width="stretch", hide_index=True)

    with technical:
        frame = resample_ohlcv(df, timeframe)
        row = frame.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("RSI(14)", f"{row['RSI']:.1f}")
        c2.metric("MACD", "上向き" if row["MACD"] > row["MACD_signal"] else "下向き")
        c3.metric("SMA20比", f"{(row['Close'] / row['SMA20'] - 1) * 100:+.1f}%")
        st.plotly_chart(price_figure(df, timeframe, 240 if timeframe == "日足" else 156), width="stretch")
        st.caption("テクニカル指標は判断の補助情報です。最終スコアは類似局面の確率分布を中心に算出しています。")

    with validation:
        st.markdown("#### 今回の判断に使った根拠")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("類似局面", f"{result['n']}件")
        v2.metric("平均類似度", f"{result['average_similarity'] * 100:.0f}%")
        v3.metric("期待リターン", f"{decision['expected_return'] * 100:+.1f}%")
        v4.metric("勝率", f"{decision['p_up'] * 100:.0f}%")
        st.markdown("#### 時間を巻き戻した予測テスト")
        st.caption("各検証時点より前に結果が判明していたデータだけを使用します。未来のデータは検索候補から除外します。")
        if st.button("この銘柄で過去検証を実行", type="primary"):
            validation_result = get_walkforward(symbol, timeframe, window, horizon, k)
            if validation_result["n"] == 0:
                st.warning("検証に必要な過去データを十分に確保できませんでした。")
            else:
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("検証回数", f"{validation_result['n']}回")
                w2.metric("最有力シナリオ的中率", f"{validation_result['scenario_accuracy'] * 100:.0f}%")
                if validation_result["buy_average_return"] is None:
                    w3.metric("買い判定後の平均", "該当なし")
                    w4.metric("買い判定後の勝率", "該当なし")
                else:
                    w3.metric("買い判定後の平均", f"{validation_result['buy_average_return'] * 100:+.1f}%")
                    w4.metric("買い判定後の勝率", f"{validation_result['buy_win_rate'] * 100:.0f}%")
                st.caption(
                    f"確率誤差（Brier score） {validation_result['brier_up']:.3f}。0に近いほど良好。"
                    f"検証中の最大下落 {validation_result['worst_return'] * 100:+.1f}%"
                )
                history = validation_result["rows"].copy()
                history["as_of"] = history["as_of"].dt.strftime("%Y-%m-%d")
                history["predicted_probability"] = (history["predicted_probability"] * 100).round(0).astype(int).astype(str) + "%"
                history["actual_return"] = (history["actual_return"] * 100).round(1).map(lambda x: f"{x:+.1f}%")
                history = history[["as_of", "predicted_scenario", "predicted_probability", "decision",
                                   "actual_scenario", "actual_return", "scenario_hit"]]
                history.columns = ["予測日", "最有力予測", "予測確率", "参考判断", "実際", "実績リターン", "的中"]
                st.dataframe(history, width="stretch", hide_index=True)


def render_image_analysis(timeframe: str, span: str, horizon: int, k: int) -> None:
    cfg = timeframe_config(timeframe)
    window = cfg["spans"][span]
    st.subheader("画像チャートから確率分析")
    uploaded = st.file_uploader("チャート画像（PNG / JPG）", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("ローソク足または折れ線チャートをアップロードしてください。日足・週足は左側で指定します。")
        return
    image = Image.open(uploaded)
    detail = extract_series_detailed(image, n_points=window)
    if detail is None:
        st.image(image, caption="アップロード画像", width="stretch")
        st.error("価格の形を読み取れませんでした。チャート部分だけにトリミングして再度お試しください。")
        return
    left, right = st.columns(2)
    left.image(render_overlay(image, detail), caption="赤線が読み取った価格形状", width="stretch")
    query = detail["values"]
    fig = go.Figure(go.Scatter(y=query, line=dict(color="#1769e0", width=3)))
    fig.update_layout(height=300, title="抽出した形", xaxis_visible=False, yaxis_visible=False)
    right.plotly_chart(fig, width="stretch")
    right.progress(detail["confidence"], text=f"画像読み取り信頼度 {detail['confidence'] * 100:.0f}%")
    result = find_similar(query, get_library(timeframe, window), k=k)
    if result is None:
        st.error("似た局面を十分に見つけられませんでした。足種または分析期間を確認してください。")
        return
    render_decision(build_decision(result, horizon))
    st.plotly_chart(fan_figure(result), width="stretch")


def main() -> None:
    if not _check_password():
        return
    st.title("AI株式判断スカウター")
    st.caption("過去の類似局面を確率化し、リスクと期待値から参考投資判断につなげます。データ: Yahoo Finance（遅延・欠損の可能性あり）")
    if not SUMMARY_FILE.exists():
        st.error("データがありません。先に `python fetch_data.py` と `python process_data.py` を実行してください。")
        return

    summary = load_summary()
    with st.sidebar:
        st.header("分析条件")
        mode = st.radio("入力", ["銘柄を選ぶ", "画像をアップロード"], horizontal=True)
        timeframe = st.radio("足種", list(TIMEFRAME_CONFIG), horizontal=True)
        cfg = timeframe_config(timeframe)
        span = st.select_slider("形を比べる期間", list(cfg["spans"]), value=list(cfg["spans"])[1])
        horizon = st.select_slider(
            "投資判断の期間",
            list(cfg["checks"]),
            value=cfg["checks"][-1],
            format_func=lambda value: f"{value}{cfg['unit']}後",
        )
        k = st.slider("参照する類似局面", 30, 200, 100, 10)
        st.caption("日足と週足は、それぞれ同じ足種の過去データだけで比較します。")

    if mode == "画像をアップロード":
        render_image_analysis(timeframe, span, horizon, k)
        return

    with st.sidebar:
        query_text = st.text_input("銘柄検索", placeholder=SEARCH_PLACEHOLDER)
        pool = filter_df(summary, query_text)
        if pool.empty:
            st.warning("該当する銘柄がありません")
            return
        options = (pool["symbol"] + "  " + pool["name"]).tolist()
        default_index = next((i for i, option in enumerate(options) if option.startswith("6857.T")), 0)
        selected = st.selectbox("銘柄", options, index=default_index)
        symbol = selected.split()[0]

    name, _, sector = TICKERS.get(symbol, (symbol, "?", "?"))
    current = float(load_stock(symbol)["Close"].dropna().iloc[-1])
    st.markdown(f"### {name}（{symbol}）")
    st.caption(f"{sector}　現在値 {fmt_price(current, symbol)}　{timeframe}")
    render_analysis(symbol, timeframe, span, horizon, k)
    st.divider()
    st.caption(
        "本ツールは教育・調査目的です。表示する判断は過去データに基づく参考評価であり、"
        "将来の成果を保証せず、個別の投資助言や売買の勧誘を行うものではありません。"
    )


if __name__ == "__main__":
    main()
