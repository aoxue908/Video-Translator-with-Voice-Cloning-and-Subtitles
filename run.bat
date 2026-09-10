@echo off
title Video Translator with Voice Cloning
echo ===================================================
echo   Starting Video Translator with Voice Cloning
echo ===================================================
echo.

call conda activate vt_env 2>nul

python app.py

if errorlevel 1 (
    echo.
    echo Error occurred while running the application.
    pause
)
