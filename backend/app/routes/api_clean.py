"""
Clean API routes with simplified Socket.IO communication (no authentication for sockets)
"""

import logging
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, make_response, redirect, url_for
from flask_jwt_extended import (
    jwt_required, get_jwt_identity, set_access_cookies, 
    unset_jwt_cookies, decode_token
)
from flask_socketio import emit
from sqlalchemy import desc

# Local imports
from app.config import Config
from app.models.traffic_data import TrafficData
from app.models.user import User
from app.services import AuthService, TrafficDataService
from app.utils.video_processor import start_video_processing, stop_video_processing, is_processing_active
from app import Session

# Initialize blueprint
api = Blueprint('api', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# REST API endpoints (keep authentication for REST)
@api.route('/login', methods=['POST'])
def login():
    """API endpoint for user authentication."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'Request body must be JSON'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'message': 'Username and password are required'}), 400
        
        success, access_token, error_message = AuthService.authenticate_user(username, password)
        
        if success:
            response = make_response(jsonify({
                'message': 'Login successful',
                'access_token': access_token
            }))
            set_access_cookies(response, access_token)
            return response
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            return jsonify({'message': error_message or 'Invalid credentials'}), 401
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'message': 'Internal server error'}), 500

@api.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    response = make_response(redirect(url_for('api.login')))
    unset_jwt_cookies(response)
    return response

@api.route('/data')
@jwt_required()
def get_data():
    """Get recent traffic data for dashboard."""
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = max(1, min(limit, 100))
        
        session = Session()
        try:
            recent_data = (session.query(TrafficData)
                         .order_by(desc(TrafficData.timestamp))
                         .limit(limit)
                         .all())
            
            result = []
            for record in recent_data:
                safe_record = {
                    'id': record.id,
                    'junction': record.junction,
                    'total_count': record.total_count,
                    'car_count': record.car_count,
                    'bus_count': record.bus_count,
                    'truck_count': record.truck_count,
                    'motorcycle_count': record.motorcycle_count,
                    'traffic_light': record.traffic_light,
                    'light_duration': record.light_duration,
                    'timestamp': record.timestamp.isoformat(),
                    'time': record.timestamp.strftime("%H:%M:%S"),
                    'date': record.timestamp.strftime("%Y-%m-%d")
                }
                result.append(safe_record)
            
            return jsonify(result)
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error fetching traffic data: {e}")
        return jsonify({'error': 'Failed to fetch traffic data'}), 500

@api.route('/current-status')
@jwt_required()
def get_current_status():
    """Get current traffic status and system state."""
    try:
        session = Session()
        try:
            latest_record = (session.query(TrafficData)
                           .order_by(desc(TrafficData.timestamp))
                           .first())
            
            if latest_record:
                current_data = {
                    'junction': latest_record.junction,
                    'time': latest_record.timestamp.strftime("%H:%M:%S"),
                    'count': latest_record.total_count,
                    'car': latest_record.car_count,
                    'bus': latest_record.bus_count,
                    'truck': latest_record.truck_count,
                    'motorcycle': latest_record.motorcycle_count,
                    'traffic_light': latest_record.traffic_light,
                    'light_duration': latest_record.light_duration,
                    'timestamp': latest_record.timestamp.isoformat(),
                    'last_updated': latest_record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                current_data = {
                    'junction': 'Main St & 1st Ave',
                    'time': 'No Data',
                    'count': 0,
                    'car': 0,
                    'bus': 0,
                    'truck': 0,
                    'motorcycle': 0,
                    'traffic_light': 'red',
                    'light_duration': 30,
                    'timestamp': None,
                    'last_updated': 'Never'
                }
            
            status = {
                'current_data': current_data,
                'system_status': {
                    'processing_active': is_processing_active(),
                    'total_records': session.query(TrafficData).count(),
                    'last_update': current_data.get('last_updated')
                }
            }
            
            return jsonify(status)
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error fetching current status: {e}")
        return jsonify({'error': 'Failed to fetch current status'}), 500

@api.route('/video-stream')
def video_stream():
    """Serve video stream without authentication for public access."""
    try:
        video_path = Config.VIDEO_PATH
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name='traffic_stream.mp4'
        )
    except Exception as e:
        logger.error(f"Error serving video stream: {e}")
        return jsonify({'error': 'Failed to serve video stream'}), 500


def register_socketio_handlers(socketio):
    """Register simplified SocketIO event handlers without authentication"""
    
    # Simple global processing state
    processing_state = {
        'active': False,
        'user': None,
        'junction': None,
        'started_at': None
    }
    
    @socketio.on('connect')
    def handle_connect():
        """Simple connection handler without authentication."""
        logger.info(f"✅ Client connected with socket {request.sid}")
        
        # Send current processing state to newly connected client
        socketio.emit('processing_state', {
            'active': processing_state['active'],
            'user': processing_state['user'],
            'junction': processing_state['junction'],
            'started_at': processing_state['started_at']
        }, to=request.sid)
        
        return True
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Simple disconnect handler."""
        logger.info(f"Client {request.sid} disconnected")
    
    @socketio.on('test_connection')
    def handle_test_connection(data):
        """Simple test connection handler without authentication."""
        logger.info(f"Test connection from {request.sid}")
        
        # Broadcast test response
        socketio.emit('test_response', {
            'message': f'Connection test successful',
            'timestamp': datetime.now().isoformat(),
            'socket_id': request.sid
        })
    
    @socketio.on('start_processing')
    def handle_start_processing(data):
        """Start video processing without authentication."""
        nonlocal processing_state
        
        try:
            # Use default user for processing
            user = "admin"
            logger.info(f"Starting video processing for {user}")
            
            # Check if already processing
            if processing_state['active']:
                socketio.emit('processing_error', {
                    'message': f'Video processing is already active for {processing_state["user"]}',
                    'current_user': processing_state['user']
                })
                return
            
            # Get session for database operations
            from app import db
            session = db.session
            
            # Start processing
            junction = data.get('junction', 'main_junction') if data else 'main_junction'
            video_path = data.get('video_path') if data else None
            
            success = start_video_processing(
                app=None,
                socketio_param=socketio, 
                session=session, 
                junction=junction, 
                video_path=video_path,
                user_room=None
            )
            
            if success:
                # Update global state
                processing_state.update({
                    'active': True,
                    'user': user,
                    'junction': junction,
                    'started_at': datetime.now().isoformat()
                })
                
                # Broadcast processing started to all clients
                socketio.emit('processing_started', {
                    'message': f'Video processing started for {junction}',
                    'user': user,
                    'junction': junction,
                    'timestamp': processing_state['started_at']
                })
                
                logger.info(f"🚀 Processing started by {user} for {junction}")
            else:
                socketio.emit('processing_error', {
                    'message': 'Failed to start video processing'
                })
                
        except Exception as e:
            logger.error(f"Error starting video processing: {e}")
            socketio.emit('processing_error', {
                'message': f'Failed to start video processing: {str(e)}'
            })
    
    @socketio.on('stop_processing')
    def handle_stop_processing(data):
        """Stop video processing without authentication."""
        nonlocal processing_state
        
        try:
            user = "admin"
            logger.info(f"Stopping video processing for {user}")
            
            if not processing_state['active']:
                socketio.emit('processing_error', {
                    'message': 'Video processing is not currently active'
                })
                return
            
            success = stop_video_processing()
            
            if success:
                # Update global state
                processing_state.update({
                    'active': False,
                    'user': None,
                    'junction': None,
                    'started_at': None
                })
                
                # Broadcast processing stopped to ALL clients immediately
                socketio.emit('processing_stopped', {
                    'message': f'Video processing stopped by {user}',
                    'user': user,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"🛑 Processing stopped by {user}")
            else:
                socketio.emit('processing_error', {
                    'message': 'Failed to stop video processing'
                })
                
        except Exception as e:
            logger.error(f"Error stopping video processing: {e}")
            socketio.emit('processing_error', {
                'message': f'Failed to stop video processing: {str(e)}'
            })
