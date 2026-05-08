@echo off
chcp 65001 >nul 2>&1
title Market Regime Analyzer

echo ============================================
echo  Killing existing Python servers...
echo ============================================
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo ============================================
echo  Starting Market Regime Analyzer...
echo ============================================
cd /d "%~dp0"
start "" http://127.0.0.1:5000
python finder.py
pause
