# -*- coding: utf-8 -*-
"""フェーズ4: Streamlit UI（銘柄選択 → チャート・指標・予測の表示）。

起動: streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

from chart_extract import extract_series_from_image
from forecast import forecast_next_week
from pattern_match import WINDOW_BY_SPAN, build_library, find_similar
from tickers import TICKERS

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SUMMARY_FILE = BASE_DIR / "data" / "summary.csv"

st.set_page_config(page_title="日本株テクニカル分析", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600)
def load_summary() -> pd.DataFrame:
    s = pd.read_csv(SUMMARY_FILE)
    if "bb_signal" in s.columns:
        s["bb_signal"] = s["bb_signal"].fillna("")  # 空欄はCSV経由でNaNになるため
    return s


@st.cache_data(ttl=3600)
def load_stock(symbol: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{symbol}.csv", index_col="Date", parse_dates=True)


@st.cache_resource(show_spinner="過去パターンのライブラリを構築中…")
def get_library(window: int) -> dict:
    return build_library(window)


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(symbol: str) -> pd.DataFrame:
    """ARIMA込みで1銘柄0.7秒ほどかかるため、銘柄ごとにキャッシュする。"""
    return forecast_next_week(load_stock(symbol))


def page_pattern(summary: pd.DataFrame) -> None:
    """パターン照合ページ: チャートの形に似た過去局面を探し、その後の推移を確率で示す。"""
    st.subheader("🔮 パターン照合 — 「この形のあと、過去はどう動いたか」")
    st.caption("入力したチャートの形に似ている局面を過去5年×300社の全履歴から探し、"
               "その後20営業日の値動きを頻度分布として集計します。過去の頻度であり、将来の保証ではありません。")

    c1, c2 = st.columns([1, 1])
    with c1:
        src = st.radio("パターンの入力方法", ["📷 チャート画像をアップロード", "📊 銘柄から選ぶ"], horizontal=True)
    with c2:
        span = st.select_slider("チャートのおおよその期間", list(WINDOW_BY_SPAN.keys()), value="3ヶ月")
    window = WINDOW_BY_SPAN[span]

    query = None
    if src.startswith("📷"):
        up = st.file_uploader("チャート画像（PNG / JPG）。折れ線・ローソク足どちらでも可", type=["png", "jpg", "jpeg"])
        if up is None:
            st.info("チャートのスクリーンショットや写真をアップロードしてください。"
                    "余計な文字が少なく、チャート本体が大きく写っているほど精度が上がります。")
            return
        img = Image.open(up)
        query = extract_series_from_image(img, n_points=window)
        ic1, ic2 = st.columns(2)
        with ic1:
            st.image(img, caption="アップロードされた画像", use_container_width=True)
        with ic2:
            if query is None:
                st.error("チャートの線をうまく抽出できませんでした。トリミングして本体だけにした画像で再度お試しください。")
                return
            fig_q = go.Figure(go.Scatter(y=query, mode="lines", line=dict(color="#e07030", width=2)))
            fig_q.update_layout(title="抽出した形状（これで照合します）", height=260,
                                margin=dict(t=40, b=10, l=10, r=10),
                                xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig_q, use_container_width=True)
        st.caption("↑ 右の形が元チャートと合っていることを確認してから、下の結果を見てください。")
    else:
        options = summary["symbol"] + "  " + summary["name"]
        choice = st.selectbox("銘柄（直近の形を照合パターンとして使う）", options.tolist())
        symbol = choice.split()[0]
        df = load_stock(symbol)
        query = df["Close"].dropna().iloc[-window:].to_numpy()
        fig_q = go.Figure(go.Scatter(x=df.index[-window:], y=query, mode="lines",
                                     line=dict(color="#e07030", width=2)))
        fig_q.update_layout(title=f"照合パターン: {TICKERS.get(symbol, (symbol,))[0]} 直近{span}",
                            height=260, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_q, use_container_width=True)

    k = st.slider("参照する類似局面の数", 50, 300, 100, 25,
                  help="多いほど安定しますが、形の似ていない局面も混ざります")

    lib = get_library(window)
    res = find_similar(query, lib, k=k)
    if res is None:
        st.error("類似局面を十分に見つけられませんでした。期間設定を変えてお試しください。")
        return

    # ---- 確率サマリー ----
    st.markdown(f"#### 📊 類似局面 {res['n']}件の「その後」")
    cols = st.columns(3)
    for col, h in zip(cols, (5, 10, 20)):
        s = res["horizons"][h]
        col.metric(f"{h}営業日後に上昇していた確率", f"{s['p_up'] * 100:.0f}%",
                   f"中央値 {s['median'] * 100:+.1f}%")

    # ---- ファンチャート（その後20日のパーセンタイル）----
    days = np.arange(1, len(res["paths_pct"]["p50"]) + 1)
    fig = go.Figure()
    band = [("p90", "p10", "rgba(80,128,208,0.15)", "10〜90%タイル"),
            ("p75", "p25", "rgba(80,128,208,0.30)", "25〜75%タイル")]
    for hi, lo, color, label in band:
        fig.add_trace(go.Scatter(x=days, y=(res["paths_pct"][hi] - 1) * 100,
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=days, y=(res["paths_pct"][lo] - 1) * 100,
                                 line=dict(width=0), fill="tonexty", fillcolor=color,
                                 name=label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=(res["paths_pct"]["p50"] - 1) * 100,
                             line=dict(color="#e07030", width=2.5), name="中央値"))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title="類似局面のその後20営業日（現在=0%とした変化率の分布）",
                      xaxis_title="営業日後", yaxis_title="変化率 (%)",
                      height=420, margin=dict(t=50, b=40, l=10, r=10),
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # ---- ヒストグラム + 類似局面リスト ----
    h1, h2 = st.columns([1, 1])
    with h1:
        fig_h = go.Figure(go.Histogram(x=res["ret20"] * 100, nbinsx=30,
                                       marker_color="#5080d0"))
        fig_h.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_h.update_layout(title="20営業日後リターンの分布", xaxis_title="リターン (%)",
                            yaxis_title="件数", height=360, margin=dict(t=50, b=40, l=10, r=10))
        st.plotly_chart(fig_h, use_container_width=True)
    with h2:
        st.markdown("**類似度が高かった局面 トップ10**")
        top = res["matches"].head(10).copy()
        top.insert(1, "企業名", [TICKERS.get(s, (s,))[0] for s in top["symbol"]])
        top.columns = ["コード", "企業名", "類似日", "類似度%", "20日後%"]
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.caption("⚠️ これは「過去に形が似ていた局面の頻度集計」です。相場環境が違えば同じ形でも結果は変わります。投資判断の唯一の根拠にしないでください。")


def _check_password() -> bool:
    """Web公開用の簡易パスワード保護。secrets に APP_PASSWORD が無ければ素通し（ローカル利用）。"""
    try:
        pw = st.secrets["APP_PASSWORD"]
    except Exception:
        return True
    if st.session_state.get("auth_ok"):
        return True
    entered = st.text_input("パスワード", type="password")
    if entered == pw:
        st.session_state["auth_ok"] = True
        st.rerun()
    elif entered:
        st.error("パスワードが違います")
    return False


def main() -> None:
    if not _check_password():
        return
    st.title("📈 日本株テクニカル分析 + 統計予測")
    st.caption("教育・参考目的のツールです。投資判断はご自身の責任で行ってください。データ: Yahoo Finance（無料・遅延あり）")

    if not SUMMARY_FILE.exists():
        st.error("データがありません。先に `python fetch_data.py` と `python process_data.py` を実行してください。")
        return

    summary = load_summary()

    mode = st.sidebar.radio("モード", ["📈 銘柄分析", "🔮 パターン照合"], horizontal=True)
    if mode == "🔮 パターン照合":
        page_pattern(summary)
        return

    # ---- サイドバー: 銘柄選択 ----
    with st.sidebar:
        st.header("銘柄選択")
        group = st.radio("区分", ["すべて", "日本株", "米国株(S&P500)",
                                  "大型株(TOPIX100)", "中型株(Mid400)", "小型株"], index=0)
        group_map = {"大型株(TOPIX100)": "large", "中型株(Mid400)": "mid",
                     "小型株": "small", "米国株(S&P500)": "us"}
        pool = summary
        if group == "日本株":
            pool = summary[summary["group"].isin(["large", "mid", "small"])]
        elif group in group_map:
            pool = summary[summary["group"] == group_map[group]]

        sectors = ["すべて"] + sorted(pool["sector"].unique())
        sector = st.selectbox("業種", sectors)
        if sector != "すべて":
            pool = pool[pool["sector"] == sector]

        options = pool["symbol"] + "  " + pool["name"]
        choice = st.selectbox("銘柄", options.tolist())
        symbol = choice.split()[0]

        period = st.select_slider("表示期間", ["3ヶ月", "6ヶ月", "1年", "3年", "5年"], value="1年")

    name, _, sector_name = TICKERS.get(symbol, (symbol, "?", "?"))
    df = load_stock(symbol)
    days = {"3ヶ月": 63, "6ヶ月": 126, "1年": 252, "3年": 756, "5年": 10000}[period]
    view = df.iloc[-days:]

    row = summary[summary["symbol"] == symbol].iloc[0]

    # ---- ヘッダー指標 ----
    def fmt_price(v: float) -> str:  # 日本株は¥・整数、米国株は$・小数2桁
        return f"¥{v:,.0f}" if symbol.endswith(".T") else f"${v:,.2f}"

    st.subheader(f"{name}（{symbol}） — {sector_name}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("終値", fmt_price(row["close"]), f"データ日: {row['last_date']}", delta_color="off")
    c2.metric("RSI(14)", f"{row['rsi']:.1f}", row["rsi_label"], delta_color="off")
    c3.metric("MACD", row["macd_label"])
    c4.metric("トレンド(SMA)", row["trend"])
    c5.metric("来週予測(アンサンブル)", fmt_price(row["forecast_1w"]), f"{row['forecast_1w_pct']:+.2f}%")

    # ---- メインチャート（ローソク足 + SMA + BB + 予測）----
    fc = get_forecast(symbol)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=("価格・移動平均・ボリンジャーバンド・来週予測", "RSI(14)", "MACD(12,26,9)"),
    )

    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name="ローソク足",
        increasing_line_color="#e05555", decreasing_line_color="#4a7bd4",
    ), row=1, col=1)

    for col, color in [("SMA5", "#f0a030"), ("SMA20", "#30b070"), ("SMA50", "#9060d0")]:
        fig.add_trace(go.Scatter(x=view.index, y=view[col], name=col,
                                 line=dict(width=1.2, color=color)), row=1, col=1)

    fig.add_trace(go.Scatter(x=view.index, y=view["BB_upper"], name="BB上限",
                             line=dict(width=0.8, color="rgba(120,120,120,0.5)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["BB_lower"], name="BB下限",
                             line=dict(width=0.8, color="rgba(120,120,120,0.5)"),
                             fill="tonexty", fillcolor="rgba(120,120,120,0.08)"), row=1, col=1)

    # 予測ライン + 信頼区間
    last_dt, last_close = view.index[-1], view["Close"].iloc[-1]
    fx = [last_dt] + list(fc["Date"])
    fig.add_trace(go.Scatter(x=fx, y=[last_close] + list(fc["upper"]), showlegend=False,
                             line=dict(width=0, color="rgba(240,80,80,0)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=fx, y=[last_close] + list(fc["lower"]), name="95%信頼区間",
                             line=dict(width=0, color="rgba(240,80,80,0)"),
                             fill="tonexty", fillcolor="rgba(240,120,60,0.18)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fx, y=[last_close] + list(fc["forecast"]), name="来週予測",
                             line=dict(width=2, dash="dash", color="#e07030")), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=view.index, y=view["RSI"], name="RSI",
                             line=dict(width=1.2, color="#5080d0")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#e05555", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#30b070", row=2, col=1)

    # MACD
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_hist"], name="ヒストグラム",
                         marker_color="rgba(140,140,140,0.5)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], name="MACD",
                             line=dict(width=1.2, color="#e07030")), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD_signal"], name="シグナル",
                             line=dict(width=1.2, color="#5080d0")), row=3, col=1)

    fig.update_layout(height=820, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02),
                      margin=dict(t=60, b=20, l=10, r=10))
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ---- スクリーニング一覧 ----
    st.subheader("📋 全銘柄スクリーニング")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "売られすぎ (RSI≤30)", "ゴールデンクロス", "BB±2σタッチ", "出来高急増", "全銘柄一覧"])
    cols = ["symbol", "name", "sector", "close", "rsi", "rsi_label", "macd_label",
            "trend", "bb_signal", "vol_ratio", "forecast_1w_pct"]
    with tab1:
        st.dataframe(summary[summary["rsi"] <= 30][cols], use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(summary[summary["macd_label"] == "ゴールデンクロス"][cols],
                     use_container_width=True, hide_index=True)
    with tab3:
        st.caption("上限タッチ=強い上昇圧力（買われすぎ側）/ 下限タッチ=強い下落圧力（売られすぎ側）")
        st.dataframe(summary[summary["bb_signal"] != ""][cols].sort_values("bb_signal"),
                     use_container_width=True, hide_index=True)
    with tab4:
        st.caption("直近出来高が過去20日平均の2倍以上。何か材料が出ている可能性")
        st.dataframe(summary[summary["vol_ratio"] >= 2.0][cols].sort_values("vol_ratio", ascending=False),
                     use_container_width=True, hide_index=True)
    with tab5:
        st.dataframe(summary[cols].sort_values("forecast_1w_pct", ascending=False),
                     use_container_width=True, hide_index=True)

    st.caption("⚠️ 予測は直近60営業日への線形回帰による簡易的な参考値であり、将来の価格を保証するものではありません。")


if __name__ == "__main__":
    main()
