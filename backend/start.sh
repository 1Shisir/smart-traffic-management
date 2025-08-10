#!/bin/bash
# Smart Traffic Management System - Linux/Mac Startup Script

echo "Starting Smart Traffic Management System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing/updating dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Check if required files exist
if [ ! -f "traffic_sample1.mp4" ]; then
    echo "WARNING: traffic_sample1.mp4 not found"
    echo "Please ensure the video file is in the backend directory"
fi

if [ ! -f "yolov8n.pt" ]; then
    echo "WARNING: yolov8n.pt not found"
    echo "Please ensure the YOLO model file is in the backend directory"
fi

# Initialize database
echo "Initializing database..."
python manage.py init-db
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to initialize database"
    exit 1
fi

# Start the application
echo "Starting application server..."
echo ""
echo "Application will be available at: http://localhost:5000"
echo "Default login: admin / admin123"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
