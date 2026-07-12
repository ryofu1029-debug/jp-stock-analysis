# -*- coding: utf-8 -*-
"""フェーズ2: テクニカル指標の計算（pandas のみ、外部TAライブラリ不使用）。

- SMA 5 / 20 / 50（移動平均線）
- RSI 14（Wilder方式。70以上=買われすぎ / 30以下=売られすぎ）
- MACD (12, 26, 9)（MACDがシグナルを上抜け=ゴールデンクロス）
- ボリンジャーバンド (20日, ±2σ)
"""
import pandas as pd


def add_sma(df: pd.DataFrame, windows: tuple[int, ...] = (5, 20, 50)) -> pd.DataFrame:
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(w).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder の平滑化 = ewm(alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - 100 / (1 + rs)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + num_std * std
    df["BB_lower"] = mid - num_std * std
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """全指標を一括計算して列を追加した DataFrame を返す。"""
    df = df.copy()
    add_sma(df)
    add_rsi(df)
    add_macd(df)
    add_bollinger(df)
    return df


def latest_signals(df: pd.DataFrame) -> dict:
    """最新日のシグナル判定をまとめて返す（サマリー用）。"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = last["RSI"]
    if pd.isna(rsi):
        rsi_label = "-"
    elif rsi >= 70:
        rsi_label = "買われすぎ"
    elif rsi <= 30:
        rsi_label = "売られすぎ"
    else:
        rsi_label = "中立"

    # MACD クロス判定（直近でクロスしたか）
    if prev["MACD"] <= prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
        macd_label = "ゴールデンクロス"
    elif prev["MACD"] >= prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
        macd_label = "デッドクロス"
    elif last["MACD"] > last["MACD_signal"]:
        macd_label = "上昇トレンド"
    else:
        macd_label = "下降トレンド"

    trend = "上向き" if last["SMA5"] > last["SMA20"] > last["SMA50"] else (
        "下向き" if last["SMA5"] < last["SMA20"] < last["SMA50"] else "横ばい"
    )

    # ボリンジャーバンド ±2σ タッチ判定
    if pd.notna(last["BB_upper"]) and last["Close"] >= last["BB_upper"]:
        bb_signal = "上限タッチ"
    elif pd.notna(last["BB_lower"]) and last["Close"] <= last["BB_lower"]:
        bb_signal = "下限タッチ"
    else:
        bb_signal = ""

    # 出来高急増: 直近出来高 ÷ 過去20日平均
    vol_avg = df["Volume"].iloc[-21:-1].mean()
    vol_ratio = float(last["Volume"] / vol_avg) if vol_avg > 0 else 0.0

    return {
        "close": round(float(last["Close"]), 1),
        "rsi": round(float(rsi), 1) if pd.notna(rsi) else None,
        "rsi_label": rsi_label,
        "macd_label": macd_label,
        "trend": trend,
        "bb_signal": bb_signal,
        "vol_ratio": round(vol_ratio, 2),
    }
