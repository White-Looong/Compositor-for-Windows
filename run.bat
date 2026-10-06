@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PY_BOOT=python"

rem ---- Step 1: create virtualenv (only if interpreter really missing) ----
if exist "%PY%" goto :check_deps

echo [1/3] Creating virtual environment...
where %PY_BOOT% >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python not found in PATH. Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure "Add python.exe to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

%PY_BOOT% -m venv "%VENV%"
if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
)

:check_deps
rem ---- Step 2: probe dependencies for real (do NOT trust the folder alone) ----
rem A half-built venv (folder exists, packages missing) would otherwise be
rem treated as "ready", skip the install, and crash instantly with no visible
rem error. So actually import the packages here.
"%PY%" -c "import PySide6, numpy, cv2, PIL" >nul 2>nul
if not errorlevel 1 goto :run

echo [2/3] Installing dependencies (first run takes a few minutes)...
"%PY%" -m pip install --upgrade pip -q
"%PY%" -m pip install -r "%ROOT%requirements.txt"
"%PY%" -c "import PySide6, numpy, cv2, PIL" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Try again, or install manually:
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:run
echo [3/3] Starting Compositor...
cd /d "%ROOT%"
"%PY%" -m src.main

if errorlevel 1 (
    echo.
    echo Something went wrong. See the message above.
    pause
)
endlocal
