@echo off
REM LinguaBridge Local - Web Frontend Launcher
REM This script starts the API server and opens the web interface

echo ============================================================
echo  LinguaBridge Local - Web Frontend Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Start API server in background
echo Starting API server...
start "LinguaBridge API" python run.py api

REM Wait for server to start
echo Waiting for server to start...
timeout /t 5 /nobreak >nul

REM Open web browser
echo Opening web interface...
start http://localhost:8000

echo.
echo ============================================================
echo  Web UI: http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo  
echo  Press Ctrl+C in the API window to stop the server
echo ============================================================
echo.

pause
