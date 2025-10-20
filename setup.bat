@echo off
REM Setup script for Satellite RL Project (Windows)
REM This script automates the virtual environment creation and dependency installation

setlocal enabledelayedexpansion

echo =========================================
echo Satellite RL Project Setup
echo =========================================
echo.

REM Check if Python is installed
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python !PYTHON_VERSION!
echo.

REM Create virtual environment
echo Creating virtual environment (venv_windows)...
if exist "venv_windows\" (
    echo Warning: venv_windows already exists
    set /p RECREATE="Do you want to remove it and create a fresh one? (y/n): "
    if /i "!RECREATE!"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv_windows
        python -m venv venv_windows
        echo Created fresh virtual environment
    ) else (
        echo Using existing virtual environment
    )
) else (
    python -m venv venv_windows
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv_windows\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
echo This may take a few minutes...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo Error: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo Dependencies installed
echo.

REM Install Jupyter kernel
echo Installing Jupyter kernel...
python -m ipykernel install --user --name=satellite_rl
if errorlevel 1 (
    echo Warning: Failed to install Jupyter kernel
    echo You may need to install it manually later
) else (
    echo Jupyter kernel 'satellite_rl' installed
)
echo.

REM Verify installation
echo Verifying installation...
python -c "import numpy, matplotlib, gymnasium, jupyter" 2>nul
if errorlevel 1 (
    echo Warning: Some packages may not be installed correctly
) else (
    echo Core packages verified successfully
)
echo.

REM Display next steps
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo.
echo 1. Activate the virtual environment:
echo    venv_windows\Scripts\activate
echo.
echo 2. Launch Jupyter Lab:
echo    jupyter lab
echo.
echo 3. Open the notebook:
echo    notebook/satellite_sensor_tasking.ipynb
echo.
echo 4. Select kernel: satellite_rl
echo.
echo 5. Run all cells!
echo.
echo =========================================
echo To run tests:
echo    pytest tests/ -v
echo =========================================
echo.

pause
