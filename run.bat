@echo off

REM Check Python version is 3.9 or higher
for /f "tokens=2 delims= " %%i in ('python --version') do set PY_VER=%%i
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo Python 3.9 or higher is required.
    pause
    exit /b 1
) else if %MAJOR%==3 if %MINOR% LSS 9 (
    echo Python 3.9 or higher is required.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment with Python...
    python -m venv .venv
)

REM Activate the virtual environment
call .venv\Scripts\activate

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

REM Set Python paths for PySpark
set PYSPARK_PYTHON=%CD%\.venv\Scripts\python.exe
set PYSPARK_DRIVER_PYTHON=%CD%\.venv\Scripts\python.exe

REM Run the Flask app
echo Running Flask app...
python src/app.py

REM Deactivate the virtual environment after execution
deactivate
