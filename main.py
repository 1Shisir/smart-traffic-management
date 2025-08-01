import cv2
import io
import requests
from flask import Flask, render_template, jsonify, send_file
from flask_socketio import SocketIO
import threading
import time
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import os
import logging
from ultralytics import YOLO
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
Base = declarative_base()

# Database setup
engine = create_engine('sqlite:///traffic.db', echo=True)
Session = sessionmaker(bind=engine)

class TrafficData(Base):
    __tablename__ = 'traffic_data'
    id = Column(Integer, primary_key=True)
    junction = Column(String(50))
    total_count = Column(Integer)
    car_count = Column(Integer)
    bus_count = Column(Integer)
    truck_count = Column(Integer)
    motorcycle_count = Column(Integer)
    traffic_light = Column(String(10))
    light_duration = Column(Integer)
    timestamp = Column(DateTime)

# Create database tables
try:
    Base.metadata.create_all(engine)
    logging.info("Database tables created successfully")
except Exception as e:
    logging.error(f"Error creating database tables: {e}")

# Load YOLOv8 model with GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)
vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}

# Draw bounding boxes and count
def detect_vehicles(frame):
    try:
        frame = cv2.resize(frame, (640, 480))
        results = model(frame, verbose=False)[0]
        class_counts = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            if name in vehicle_labels:
                class_counts[name] += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{name} ({int(conf*100)}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_count = sum(class_counts.values())
        return total_count, class_counts, frame
    except Exception as e:
        logging.error(f"YOLO detection error: {e}")
        return 0, {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}, frame

# Video processing function
frame_for_preview = None
entries_buffer = []  # For bulk database operations

def process_video():
    global frame_for_preview, entries_buffer
    with app.app_context():
        try:
            session = Session()
            video_path = 'traffic_sample2.mp4'

            if not os.path.exists(video_path):
                logging.error(f"Video file not found: {video_path}")
                return

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error("Error opening video file")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Processing video: {video_path} ({total_frames} frames)")

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 != 0:
                    continue

                total_count, class_counts, annotated = detect_vehicles(frame)
                frame_for_preview = annotated.copy()
                timestamp = datetime.now()
                logging.info(f"Frame {frame_count}: Vehicles -> Total: {total_count}")

                # Traffic light control logic
                traffic_threshold = 12
                if total_count > traffic_threshold:
                    light_state = "red"
                    light_duration = 30
                elif total_count > traffic_threshold/2:
                    light_state = "yellow"
                    light_duration = 10
                else:
                    light_state = "green"
                    light_duration = 20

                try:
                    # Add to buffer
                    entries_buffer.append(TrafficData(
                        junction="main_junction",
                        total_count=total_count,
                        car_count=class_counts['car'],
                        bus_count=class_counts['bus'],
                        truck_count=class_counts['truck'],
                        motorcycle_count=class_counts['motorcycle'],
                        traffic_light=light_state,
                        light_duration=light_duration,
                        timestamp=timestamp
                    ))

                    # Commit in batches
                    if len(entries_buffer) >= 10:
                        session.bulk_save_objects(entries_buffer)
                        session.commit()
                        entries_buffer = []

                    # Send to frontend
                    socketio.emit('update', {
                        'junction': "main_junction",
                        'count': total_count,
                        'car': class_counts['car'],
                        'bus': class_counts['bus'],
                        'truck': class_counts['truck'],
                        'motorcycle': class_counts['motorcycle'],
                        'time': timestamp.strftime("%H:%M:%S")
                    })

                    # Send traffic light state
                    socketio.emit('traffic_light', {
                        'state': light_state,
                        'duration': light_duration
                    })

                    # Send to Node-RED
                    # payload = {
                    #     "junction": "main_junction",
                    #     "total": total_count,
                    #     "car": class_counts['car'],
                    #     "bus": class_counts['bus'],
                    #     "truck": class_counts['truck'],
                    #     "motorcycle": class_counts['motorcycle'],
                    #     "traffic_light": light_state,
                    #     "light_duration": light_duration,
                    #     "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    # }
                    # try:
                    #     res = requests.post("http://localhost:1880/traffic", json=payload, timeout=5)
                    #     logging.info(f"Node-RED POST: {res.status_code}")
                    # except requests.exceptions.Timeout:
                    #     logging.warning("Node-RED request timed out")
                    # except requests.exceptions.RequestException as e:
                    #     logging.error(f"Node-RED error: {e}")

                except Exception as e:
                    logging.error(f"Database error: {e}")
                    session.rollback()

                time.sleep(0.1)

            # Commit any remaining entries
            if entries_buffer:
                session.bulk_save_objects(entries_buffer)
                session.commit()
                
            cap.release()
            logging.info("Video processing completed")

        except Exception as e:
            logging.error(f"Error in video processing: {e}")

# App routes (unchanged from your original code)
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def traffic_data():
    try:
        session = Session()
        data = session.query(TrafficData).order_by(TrafficData.timestamp.desc()).limit(100).all()
        return jsonify([{
            'junction': d.junction,
            'total': d.total_count,
            'car': d.car_count,
            'bus': d.bus_count,
            'truck': d.truck_count,
            'motorcycle': d.motorcycle_count,
            'timestamp': d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'traffic_light': d.traffic_light
        } for d in data])
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify([])

@app.route('/video-preview')
def video_preview():
    try:
        global frame_for_preview
        if frame_for_preview is None:
            return "Video not ready", 503

        ret, buffer = cv2.imencode('.jpg', frame_for_preview)
        if not ret:
            return "Image encoding failed", 500

        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    except Exception as e:
        logging.error(f"Video preview error: {e}")
        return "Preview unavailable", 500

@app.route('/video-stream')
def video_stream():
    video_path = 'traffic_sample2.mp4'
    try:
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        logging.error(f"Video stream error: {e}")
        return "Video not available", 500
    

if __name__ == '__main__':
    threading.Thread(target=process_video, daemon=True).start()
    logging.info("Starting Flask server")
    socketio.run(app, debug=True, use_reloader=False)