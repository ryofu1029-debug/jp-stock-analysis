# 日米株テクニカル分析 + 統計予測

日本株約600社（TOPIX100 + Mid400 + Small上位）と米国株約500社（S&P500全構成銘柄）、
計約1,100社の株価をテクニカル指標と統計予測アンサンブルで分析するツール。
すべて無料API（Yahoo Finance / yfinance + JPX公式リスト + Wikipedia）のみ使用。

## セットアップ

```
pip install yfinance pandas numpy scikit-learn statsmodels xlrd plotly streamlit
```

## 使い方

```
python build_tickers.py   # ① JPX公式リストから銘柄リスト生成（銘柄数を変えるときだけ）
python fetch_data.py      # ② 全銘柄 × 過去5年の日足データを取得 → data/raw/
python process_data.py    # ③ 指標計算 + アンサンブル予測 → data/processed/ + data/summary.csv
streamlit run app.py      # ④ ブラウザUIで分析表示
mobile.cmd                # ④' スマホからも使えるモードで起動（同一Wi-Fi）
```

個別銘柄の再取得: `python fetch_data.py 7203.T`

## 構成

| ファイル | 役割 |
|---|---|
| `build_tickers.py` | JPX公式リスト(TOPIX規模区分) + Wikipedia(S&P500)から銘柄を自動選定 → `data/tickers.csv` |
| `tickers.py` | 銘柄リストのローダー（手動時代の300社は `tickers_manual_300.py` に保存） |
| `fetch_data.py` | yfinanceでOHLCV取得（バッチ+リトライ+進捗表示） |
| `indicators.py` | SMA(5/20/50), RSI(14), MACD(12,26,9), ボリンジャーバンド(20,±2σ), BBタッチ/出来高急増判定 |
| `forecast.py` | 予測アンサンブル: 線形回帰 + EMAドリフト + ARIMA(1,1,1) + 曜日効果、実測ボラベースの95%信頼区間 |
| `process_data.py` | 指標+予測の一括実行、サマリー生成、銘柄リスト外の残骸CSV掃除 |
| `app.py` | Streamlit UI（銘柄分析 / パターン照合 / スクリーニング） |
| `chart_extract.py` | チャート画像（写真/スクショ）から価格の形状を抽出 |
| `pattern_match.py` | 形が似た過去局面をk近傍(相関)で検索し、その後の値動きを確率集計 |
| `ssl_setup.py` | このPC固有: AvastのSSLスキャン対策（証明書バンドル結合） |

## 銘柄数を変える（600 → 1000社）

```
python build_tickers.py 500     # TOPIX Small 1 を500社取り込み → 約1000社
python fetch_data.py && python process_data.py
```

## 予測モデル（v2）

3モデルの平均に曜日効果を加算:
1. **線形回帰** — 直近60営業日への直線フィット
2. **EMAドリフト** — EMA20の直近の傾きが継続すると仮定
3. **ARIMA(1,1,1)** — 対数価格の自己回帰（statsmodels）

信頼区間は直近120営業日の実測ボラティリティ × √日数 × 1.96。

## スクリーニング

- 売られすぎ（RSI≤30）/ ゴールデンクロス / **BB±2σタッチ** / **出来高急増（20日平均の2倍以上）** / 全銘柄一覧

## パターン照合モード

チャート画像をアップロード（または銘柄を選択）すると、全銘柄×5年の窓から
形が似た局面をピアソン相関で探し、その後20営業日の上昇確率・リターン分布
（ファンチャート/ヒストグラム）を表示する。

## 注意

- 教育・参考目的。投資判断の根拠にしないこと。
- Yahoo Financeの無料データは遅延・欠損の可能性あり。
- 予測日付は土日を除くが日本の祝日は考慮していない。
- 予測・確率は過去データの延長であり、将来を保証しない。
