@echo off
REM Smart Traffic Management System - Windows Startup Script

echo Starting Smart Traffic Management System...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo Installing/updating dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "traffic_sample1.mp4" (
    echo WARNING: traffic_sample1.mp4 not found
    echo Please ensure the video file is in the backend directory
)

if not exist "yolov8n.pt" (
    echo WARNING: yolov8n.pt not found
    echo Please ensure the YOLO model file is in the backend directory
)

REM Initialize database
echo Initializing database...
python manage.py init-db
if errorlevel 1 (
    echo ERROR: Failed to initialize database
    pause
    exit /b 1
)

REM Start the application
echo Starting application server...
echo.
echo Application will be available at: http://localhost:5000
echo Default login: admin / admin123
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py

echo.
echo Application stopped.
pause
