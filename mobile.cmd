
rem スマホ対応起動: PCと同じWi-Fiに接続したスマホのブラウザからアクセスできる
cd /d "%~dp0"
chcp 65001 >nul
for /f %%a in ('powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 -PrefixOrigin Dhcp,Manual | Where-Object {$_.IPAddress -notlike '169.254*'} | Select-Object -First 1).IPAddress"') do set IP=%%a
echo.
echo ============================================================
echo   スマホのブラウザで開く:  http://%IP%:8501
echo   （PCと同じWi-Fiに接続していること）
echo   ※初回はWindowsファイアウォールの許可ダイアログが出たら
echo     「アクセスを許可する」を押してください
echo ============================================================
echo.
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
pause
