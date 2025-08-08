from flask import Blueprint, render_template, jsonify, send_file,request,make_response,redirect,url_for
import io
import cv2
import logging
from flask_jwt_extended import jwt_required,create_access_token, unset_jwt_cookies,set_access_cookies,get_jwt_identity
from app.models.traffic_data import TrafficData
from app.models.user import User
from app.config import Config


api = Blueprint('api', __name__)
frame_for_preview = None

@api.route('/dashboard')
@jwt_required()
def dashboard():
    # username = get_jwt_identity()
    return render_template('dashboard.html')

@api.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        from app import Session
        session = Session()
        username = request.form.get('username')
        password = request.form.get('password')
        user = session.query(User).filter_by(username=username).first()
        if user and user.check_password(password):
            try:
                access_token = create_access_token(identity=username)
                response = make_response(redirect(url_for('api.dashboard')))
                set_access_cookies(response, access_token)
                logging.info(f"User {username} logged in successfully, redirecting to dashboard")
                return response
            except Exception as e:
                logging.error(f"Error setting JWT cookie for {username}: {e}")
                return render_template('login.html', error='Login failed, please try again')
        logging.warning(f"Failed login attempt for username: {username}")
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html', error=None)

@api.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    response = make_response(redirect(url_for('api.login')))
    unset_jwt_cookies(response)
    logging.info(f"User {get_jwt_identity()} logged out")
    return response

@api.route('/api/data')
@jwt_required()
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
@jwt_required()
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
@jwt_required()
def video_stream():
    try:
        video_path = Config.VIDEO_PATH
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        logging.error(f"Video stream error: {e}")
        return "Video not available", 500