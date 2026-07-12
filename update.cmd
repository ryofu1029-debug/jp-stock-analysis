@echo off
rem データ更新 → Webアプリへ反映（GitHubにプッシュすると自動で再デプロイされる）
cd /d "%~dp0"
chcp 65001 >nul
echo === 1/3 株価データ取得 ===
python fetch_data.py
echo === 2/3 指標計算 + 予測 ===
python process_data.py
echo === 3/3 Webへ反映（GitHubにプッシュ）===
git add -A
git commit -m "データ更新 %date%"
git push
echo.
echo 完了。数分後にWebアプリに反映されます。
pause
