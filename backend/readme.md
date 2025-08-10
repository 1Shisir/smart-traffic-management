# Smart Traffic Management System - Backend

A Flask-based backend application for processing traffic video feeds and managing traffic data using computer vision and machine learning.

## Features

- 🚦 Real-time traffic monitoring using YOLO object detection
- 📊 Traffic data collection and analysis
- 🔐 JWT-based authentication system
- 🌐 WebSocket support for real-time updates
- 📱 RESTful API for frontend integration
- 💾 SQLite database for data persistence
- 🔍 Health monitoring endpoints
- 🛠️ CLI management tools

## Quick Start

### Windows
```bash
start.bat
```

### Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

The application will be available at: http://localhost:5000

**Default login:** `admin` / `admin123`
Shows a bar chart of vehicle counts using Chart.js.
Lists historical data in a table.


Modular Backend: Organized Flask app with separate routes, models, and utilities.
Responsive Design: React frontend styled with Tailwind CSS for a modern, responsive UI.

Prerequisites

Python 3.8+: For the Flask backend.
Node.js 16+: For the React frontend.
SQLite: For data storage.
Video File: A sample video (traffic_sample2.mp4) for processing.

Project Structure
traffic-monitoring-system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── api.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── traffic_data.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── video_processor.py
│   ├── main.py
│   ├── requirements.txt
│   ├── traffic.db
│   └── traffic_sample2.mp4
├── traffic-frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.js
│   │   │   ├── VehicleChart.js
│   │   │   ├── TrafficLight.js
│   │   │   ├── VideoPreview.js
│   │   │   └── HistoryTable.js
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   ├── tailwind.config.js
│   ├── package.json
│   └── README.md
└── README.md

Setup Instructions
Backend (Flask)

Navigate to the backend directory:
cd backend


Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Configure the video file:

Place traffic_sample2.mp4 in traffic-backend/.
Update app/config.py if the video path differs:VIDEO_PATH = '/path/to/traffic_sample2.mp4'




Initialize the SQLite database:

The database (traffic.db) is created automatically in traffic-backend/.
To verify:sqlite3 traffic.db "SELECT * FROM traffic_data;"




Run the Flask app:
python main.py


The backend runs on http://localhost:5000.



Frontend (React)

Navigate to the frontend directory:
cd traffic-frontend


Install dependencies:
npm install


Start the React development server:
npm run dev


The frontend runs on http://localhost:5173.



Usage

Start the Backend:

Run python main.py in traffic-backend/.
The backend processes traffic_sample2.mp4, detects vehicles, stores data in traffic.db, and emits real-time updates via SocketIO.


Access the Frontend:

Open http://localhost:5173 in a browser.
The dashboard displays:
Real-time vehicle counts (total, car, bus, truck, motorcycle).
Traffic light state (Green, Yellow, Red) with duration.
A bar chart of vehicle counts.
A live video preview from /video-preview.
A table of historical data from /api/data.




Test Endpoints:

API: curl http://localhost:5000/api/data to fetch historical data.
Video Preview: http://localhost:5000/video-preview for the latest annotated frame.
Video Stream: http://localhost:5000/video-stream for the video file (optional).(Future improvements)

Backend Details

Flask Routes (app/routes/api.py):
/: Serves a placeholder dashboard.html (not used with React).
/api/data: Returns the last 100 traffic_data entries.
/video-preview: Serves the latest annotated frame.
/video-stream: Serves the video file.


Database (app/models/traffic_data.py):
SQLite table traffic_data with columns: id, junction, total_count, car_count, bus_count, truck_count, motorcycle_count, traffic_light, light_duration, timestamp.


Video Processing (app/utils/video_processor.py):
Uses YOLOv8 (yolov8n.pt) for vehicle detection.
Processes every 10th frame for efficiency.
Calculates traffic light state based on total_count (threshold: 12).
Stores data in batches (10 entries) for performance.


SocketIO Events:
update: Emits vehicle counts and timestamp.
traffic_light: Emits traffic light state and duration.



Frontend Details

Components (src/components/):
Dashboard.jsx: Main layout with real-time data, chart, video, and history.
VehicleChart.jsx: Bar chart of vehicle counts using Chart.js.
TrafficLight.jsx: Colored circle and text for traffic light state.
VideoPreview.jsx: Refreshes video preview every second.
HistoryTable.jsx: Displays historical data in a table.


Styling: Tailwind CSS for responsive, modern design.
Dependencies: React, Socket.IO-client, Chart.js, react-chartjs-2, Axios, Tailwind CSS.

Troubleshooting

Backend:
Video Not Found: Ensure traffic_sample2.mp4 exists and update VIDEO_PATH in app/config.py.
Database Errors: Check traffic.db permissions and schema:sqlite3 traffic.db "PRAGMA table_info(traffic_data);"


SocketIO Issues: Verify Flask is running on http://localhost:5000 and CORS is enabled.


Frontend:
SocketIO Connection: Check browser console for errors (e.g., http://localhost:5000/socket.io).
Video Preview: If /video-preview returns 503, wait for video processing to start.
Empty Table: Ensure /api/data returns data (check SQLite).


Performance: Adjust frame skipping (frame_count % 10) or video resolution in app/utils/video_processor.py if processing is slow.


Future Improvements

Live Streaming: Replace /video-preview with multipart/x-mixed-replace for smoother video:# In app/routes/api.py
def generate_stream():
    while True:
        global frame_for_preview
        if frame_for_preview is not None:
            ret, buffer = cv2.imencode('.jpg', frame_for_preview)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)
@api.route('/live-stream')
def live_stream():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


Trend Chart: Add a line chart for total_count over time in Dashboard.js.
Multi-Junction Support: Filter historical data by junction with a dropdown.
Dynamic Thresholds: Calculate traffic light thresholds based on historical data:# In app/utils/video_processor.py
import numpy as np
def get_dynamic_thresholds(session):
    data = session.query(TrafficData.total_count).order_by(TrafficData.timestamp.desc()).limit(1000).all()
    counts = [d.total_count for d in data]
    return np.percentile(counts, 33), np.percentile(counts, 66)



License
MIT License. Feel free to use and modify as needed.
Contact
For issues or contributions, please open a GitHub issue or contact the project maintainer.