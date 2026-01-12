@echo off
REM LinguaBridge Local - Desktop App Launcher
REM This script launches the Electron desktop application

echo ============================================================
echo  LinguaBridge Local - Desktop Application
echo ============================================================
echo.

REM Check if Node.js is installed
where node >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed
    echo.
    echo Please install Node.js from: https://nodejs.org
    echo Then run: npm install
    echo.
    pause
    exit /b 1
)

REM Check if dependencies are installed
if not exist "desktop\node_modules\" (
    echo Installing dependencies...
    cd desktop
    call npm install
    cd ..
    echo.
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Launch the desktop app
echo Launching LinguaBridge Local...
echo.
cd desktop
npm start
cd ..
