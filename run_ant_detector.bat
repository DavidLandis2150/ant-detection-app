@echo off
REM Windows launcher for Ant Detection System

REM Get the directory where this script is located
cd /d "%~dp0"

REM Run the Python installer/launcher
python install_and_run.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)
