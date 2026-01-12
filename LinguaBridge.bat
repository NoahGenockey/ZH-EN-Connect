@echo off
REM LinguaBridge Local - Simple Desktop Launcher
REM Double-click this file to launch the translation application

echo.
echo ============================================================
echo   LinguaBridge Local - Translation Application
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed
    echo Please install Python from: https://python.org
    pause
    exit /b 1
)

REM Launch the desktop GUI
echo Starting LinguaBridge...
echo.
pythonw run.py gui

REM If pythonw doesn't work, try python
if errorlevel 1 (
    python run.py gui
)
