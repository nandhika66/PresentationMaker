@echo off
REM One-time setup for PresentationMaker (Windows).
REM Creates .venv in THIS folder and installs everything.
cd /d "%~dp0"

if exist .venv (
    echo Removing old .venv ...
    rmdir /s /q .venv
)

echo Creating virtual environment with Python 3.11 ...
py -3.11 -m venv .venv
if errorlevel 1 (
    echo Python 3.11 not found via "py -3.11". Trying default python ...
    python -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment. Install Python 3.11 from python.org.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
python check_setup.py
echo.
echo Setup finished. Next time, activate with:  .venv\Scripts\activate
pause
