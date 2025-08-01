from flask import Blueprint, render_template, jsonify, send_file
import io
import cv2
import logging
from app.models.traffic_data import TrafficData

api = Blueprint('api', __name__)
frame_for_preview = None

@api.route('/')
def dashboard():
    return render_template('dashboard.html')

@api.route('/api/data')
def traffic_data():
    try:
        from app import Session
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
            'traffic_light': d.traffic_light,
            'light_duration': d.light_duration
        } for d in data])
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify([])

@api.route('/video-preview')
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

@api.route('/video-stream')
def video_stream():
    from app.config import Config
    try:
        return send_file(Config.VIDEO_PATH, mimetype='video/mp4')
    except Exception as e:
        logging.error(f"Video stream error: {e}")
        return "Video not available", 500