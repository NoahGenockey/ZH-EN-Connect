@echo off
REM LinguaBridge Local - Desktop App with Document Translation
REM Includes PDF and EPUB translation support

echo.
echo ============================================================
echo   LinguaBridge Local - Document Translation
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

REM Launch the GUI
echo Starting LinguaBridge...
echo.
pythonw -m src.app_gui_enhanced

REM If pythonw fails, show error and exit
if errorlevel 1 (
    python -m src.app_gui_enhanced
)

echo.
echo Double-click: LinguaBridge.bat
echo.
