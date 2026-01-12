@echo off
REM LinguaBridge Local - Enhanced Desktop App with Document Translation
REM Includes PDF and EPUB translation support

echo.
echo ============================================================
echo   LinguaBridge Local - Enhanced with Document Translation
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

REM Launch the enhanced GUI
echo Starting LinguaBridge Enhanced...
echo.
pythonw -m src.app_gui_enhanced

REM If pythonw doesn't work, try python
if errorlevel 1 (
    python -m src.app_gui_enhanced
)

echo.
echo Double-click: LinguaBridge-Enhanced.bat
echo.
