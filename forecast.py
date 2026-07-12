# -*- coding: utf-8 -*-
"""フェーズ3+: 統計予測アンサンブル（v2）。

3モデルの平均に曜日効果を加え、実測ボラティリティから信頼区間を出す:
  1. 線形回帰     — 直近60営業日の終値に直線をフィット（v1と同じ）
  2. EMAドリフト  — EMA20の直近の傾きがそのまま続くと仮定
  3. ARIMA(1,1,1) — 対数価格の自己回帰（statsmodels。収束失敗時はスキップ）

曜日効果: 過去250営業日の曜日別平均リターンの「全体平均からの乖離」を予測日に加算。
信頼区間: 直近120営業日の日次リターン標準偏差 × √経過日数 × 1.96。
  ※ v1は回帰の残差から出していたが、それだと不確実性を過小評価するので
    実際の値動きのブレ（ボラティリティ）ベースに変更した。
※ 教育・参考目的の簡易モデル。投資判断の根拠にはしない。
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

LOOKBACK = 60      # 線形回帰に使う日数
HORIZON = 5        # 予測する営業日数（来週）
VOL_LOOKBACK = 120  # 信頼区間用ボラティリティの計測期間
WD_LOOKBACK = 250   # 曜日効果の計測期間


def _lr_path(close: pd.Series, horizon: int) -> np.ndarray:
    recent = close.iloc[-LOOKBACK:]
    x = np.arange(len(recent)).reshape(-1, 1)
    model = LinearRegression().fit(x, recent.to_numpy())
    fx = np.arange(len(recent), len(recent) + horizon).reshape(-1, 1)
    return model.predict(fx)


def _ema_drift_path(close: pd.Series, horizon: int) -> np.ndarray:
    """EMA20の直近10日の平均傾きが続くと仮定した外挿。"""
    ema = close.ewm(span=20, adjust=False).mean()
    drift = (ema.iloc[-1] - ema.iloc[-11]) / 10
    return close.iloc[-1] + drift * np.arange(1, horizon + 1)


def _arima_path(close: pd.Series, horizon: int) -> np.ndarray | None:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_p = np.log(close.iloc[-250:].to_numpy())
            fit = ARIMA(log_p, order=(1, 1, 1)).fit(method_kwargs={"maxiter": 50})
            return np.exp(fit.forecast(horizon))
    except Exception:
        return None  # 収束失敗・未インストール時は他モデルだけで進む


def _weekday_effect(close: pd.Series) -> pd.Series:
    """曜日ごとの平均対数リターンの全体平均からの乖離（月=0〜金=4）。"""
    r = np.log(close).diff().iloc[-WD_LOOKBACK:].dropna()
    return r.groupby(r.index.weekday).mean() - r.mean()


def forecast_next_week(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """アンサンブル予測。戻り値: Date, forecast, lower, upper の DataFrame。"""
    close = df["Close"].dropna()

    paths = [_lr_path(close, horizon), _ema_drift_path(close, horizon)]
    arima = _arima_path(close, horizon)
    if arima is not None:
        paths.append(arima)
    ensemble = np.mean(paths, axis=0)

    last_date = close.index[-1]
    future_dates = pd.bdate_range(last_date, periods=horizon + 1)[1:]
    # ※ 土日は除くが日本の祝日は除いていない（表示日付が1日ずれることがある）

    # 曜日効果（対数リターンの乖離を累積して掛ける）
    wd_dev = _weekday_effect(close)
    adj = np.cumsum([wd_dev.get(d.weekday(), 0.0) for d in future_dates])
    forecast = ensemble * np.exp(adj)

    # 信頼区間: 実測ボラティリティ × √経過日数
    sigma = np.log(close).diff().iloc[-VOL_LOOKBACK:].std()
    steps = np.arange(1, horizon + 1)
    margin = 1.96 * sigma * np.sqrt(steps)

    return pd.DataFrame({
        "Date": future_dates,
        "forecast": np.round(forecast, 1),
        "lower": np.round(forecast * np.exp(-margin), 1),
        "upper": np.round(forecast * np.exp(margin), 1),
    })


def forecast_summary(df: pd.DataFrame) -> dict:
    """サマリー用: 来週末時点の予測値と現在値からの変化率。"""
    fc = forecast_next_week(df)
    last_close = float(df["Close"].dropna().iloc[-1])
    end = fc.iloc[-1]
    return {
        "forecast_1w": float(end["forecast"]),
        "forecast_1w_pct": round((float(end["forecast"]) / last_close - 1) * 100, 2),
        "forecast_lower": float(end["lower"]),
        "forecast_upper": float(end["upper"]),
    }
