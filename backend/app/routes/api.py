# Standard library imports
import io
import logging
import os
import threading
from datetime import datetime
from typing import Optional, Dict, Any

# Third-party imports
import cv2
from flask import Blueprint, render_template, jsonify, send_file, request, make_response, redirect, url_for, current_app
from flask_jwt_extended import jwt_required, create_access_token, unset_jwt_cookies, set_access_cookies, get_jwt_identity, verify_jwt_in_request
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

@api.route('/dashboard')
@jwt_required()
def dashboard():
    # username = get_jwt_identity()
    return render_template('dashboard.html')

@api.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    # Enhanced input validation and sanitization
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'Request body must be JSON'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # Input validation
        if not username or not password:
            return jsonify({'message': 'Username and password are required'}), 400
        
        # Sanitize username (prevent injection)
        if len(username) > 50 or not username.isalnum():
            return jsonify({'message': 'Invalid username format'}), 400
        
        if len(password) > 128:
            return jsonify({'message': 'Password too long'}), 400
        
        # Rate limiting would go here in production
        
        success, access_token, error_message = AuthService.authenticate_user(username, password)
        
        if success:
            response = make_response(jsonify({
                'message': 'Login successful',
                'access_token': access_token
            }))
            set_access_cookies(response, access_token)
            return response
        else:
            # Log failed attempt (for security monitoring)
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
    """Get recent traffic data for dashboard with enhanced security."""
    try:
        # Enhanced input validation with strict limits
        limit = request.args.get('limit', 10, type=int)
        limit = max(1, min(limit, 100))  # Enforce strict limits: 1-100 records
        
        # Validate user authorization
        user_identity = get_jwt_identity()
        if not user_identity:
            return jsonify({'error': 'Invalid authentication'}), 401
        
        session = Session()
        try:
            # Use parameterized query to prevent SQL injection
            recent_data = (session.query(TrafficData)
                         .order_by(desc(TrafficData.timestamp))
                         .limit(limit)
                         .all())
            
            if not recent_data:
                # Return empty data structure if no records
                return jsonify([])
            
            # Convert to list of dictionaries with safe data types
            result = []
            for record in recent_data:
                # Ensure all data is properly sanitized
                safe_record = {
                    'id': int(record.id) if record.id else 0,
                    'junction': str(record.junction)[:50] if record.junction else 'Unknown',  # Limit string length
                    'total_count': max(0, int(record.total_count)) if record.total_count else 0,
                    'car_count': max(0, int(record.car_count)) if record.car_count else 0,
                    'bus_count': max(0, int(record.bus_count)) if record.bus_count else 0,
                    'truck_count': max(0, int(record.truck_count)) if record.truck_count else 0,
                    'motorcycle_count': max(0, int(record.motorcycle_count)) if record.motorcycle_count else 0,
                    'traffic_light': str(record.traffic_light)[:10] if record.traffic_light else 'unknown',
                    'light_duration': max(0, int(record.light_duration)) if record.light_duration else 0,
                    'timestamp': record.timestamp.isoformat() if record.timestamp else '',
                    'time': record.timestamp.strftime("%H:%M:%S") if record.timestamp else '',
                    'date': record.timestamp.strftime("%Y-%m-%d") if record.timestamp else ''
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
            # Get the most recent traffic record
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
                # Default data if no records exist
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
            
            # Add system status
            from app.utils.video_processor import is_processing_active
            
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

@api.route('/generate-sample-data', methods=['POST'])
@jwt_required()
def generate_sample_data():
    """Generate sample traffic data for testing."""
    try:
        import random
        from datetime import timedelta
        
        num_records = request.json.get('count', 20) if request.json else 20
        num_records = min(num_records, 100)  # Cap at 100 records
        
        session = Session()
        try:
            # Clear existing data
            session.query(TrafficData).delete()
            session.commit()
            
            # Generate sample data
            base_time = datetime.now() - timedelta(hours=1)
            
            for i in range(num_records):
                # Create realistic traffic patterns
                timestamp = base_time + timedelta(minutes=i * 3)
                hour = timestamp.hour
                
                # Rush hour simulation
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    total_count = random.randint(15, 30)
                elif 22 <= hour or hour <= 6:
                    total_count = random.randint(2, 8)
                else:
                    total_count = random.randint(8, 18)
                
                # Vehicle distribution
                car_count = int(total_count * 0.75)
                bus_count = int(total_count * 0.08)
                truck_count = int(total_count * 0.12)
                motorcycle_count = total_count - car_count - bus_count - truck_count
                motorcycle_count = max(0, motorcycle_count)
                
                # Traffic light logic
                if total_count >= 20:
                    traffic_light = "red"
                    light_duration = 40
                elif total_count >= 12:
                    traffic_light = "yellow"
                    light_duration = 5
                else:
                    traffic_light = "green"
                    light_duration = 30
                
                traffic_entry = TrafficData(
                    junction="Main St & 1st Ave",
                    total_count=total_count,
                    car_count=car_count,
                    bus_count=bus_count,
                    truck_count=truck_count,
                    motorcycle_count=motorcycle_count,
                    traffic_light=traffic_light,
                    light_duration=light_duration,
                    timestamp=timestamp
                )
                
                session.add(traffic_entry)
            
            session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Generated {num_records} sample records',
                'count': num_records
            })
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return jsonify({'error': 'Failed to generate sample data'}), 500

@api.route('/video-feed', methods=['GET'])
@jwt_required()
def video_feed():
    """Get current video frame."""
    try:
        import cv2
        import numpy as np
        from flask import Response
        import io
        
        # For demo purposes, create a simple test frame
        # In a real implementation, this would come from your video processor
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some demo content
        cv2.putText(frame, 'LIVE VIDEO FEED', (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Time: {datetime.now().strftime("%H:%M:%S")}', 
                   (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'Monitoring Traffic...', (180, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        return Response(frame_bytes, mimetype='image/jpeg')
        
    except ImportError:
        # If OpenCV is not available, return a placeholder
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a simple placeholder image
            img = Image.new('RGB', (640, 480), color='black')
            draw = ImageDraw.Draw(img)
            
            # Add text
            draw.text((200, 200), "LIVE VIDEO FEED", fill='green')
            draw.text((220, 230), f"Time: {datetime.now().strftime('%H:%M:%S')}", fill='white')
            draw.text((200, 260), "OpenCV not installed", fill='yellow')
            draw.text((180, 290), "Install cv2 for real video", fill='red')
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            return Response(img_byte_arr.getvalue(), mimetype='image/jpeg')
        except ImportError:
            # Neither OpenCV nor PIL available
            return jsonify({'error': 'Video feed requires OpenCV or PIL installation'}), 503
        
    except Exception as e:
        logger.error(f"Error in video feed: {e}")
        return jsonify({'error': 'Video feed not available'}), 500

@api.route('/traffic-data')
@jwt_required()
def traffic_data():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        
        data_result = TrafficDataService.get_traffic_data(page, per_page)
        
        return jsonify({
            'data': data_result.get('data', []),
            'total': data_result.get('total', 0),
            'page': page,
            'per_page': per_page,
            'pages': data_result.get('pages', 0)
        })
    except Exception as e:
        logger.error(f"Error fetching traffic data: {e}")
        return jsonify({'error': 'Failed to fetch traffic data'}), 500

@api.route('/traffic-summary')
@jwt_required()
def traffic_summary():
    try:
        summary = TrafficDataService.get_traffic_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error fetching traffic summary: {e}")
        return jsonify({'error': 'Failed to fetch traffic summary'}), 500

@api.route('/recent-traffic')
@jwt_required()
def recent_traffic():
    try:
        limit = request.args.get('limit', 10, type=int)
        recent_data = TrafficDataService.get_recent_data(limit)
        return jsonify([item.to_dict() for item in recent_data])
    except Exception as e:
        logger.error(f"Error fetching recent traffic data: {e}")
        return jsonify({'error': 'Failed to fetch recent traffic data'}), 500

@api.route('/video-preview')
@jwt_required()
def video_preview():
    """Serve the current processed frame as an image."""
    try:
        from app.utils.video_processor import frame_for_preview
        
        if frame_for_preview is None:
            # Return a default image or empty response
            return jsonify({'error': 'No video frame available'}), 404
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame_for_preview)
        io_buf = io.BytesIO(buffer)
        
        return send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='preview.jpg'
        )
    except Exception as e:
        logger.error(f"Error serving video preview: {e}")
        return jsonify({'error': 'Failed to generate preview'}), 500

@api.route('/video-stream')
@jwt_required()
def video_stream():
    """Serve video stream."""
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
    """Register SocketIO event handlers"""
    
    @socketio.on('start_processing')
    def handle_start_processing(data):
        """Handle video processing start request via WebSocket."""
        try:
            # Extract token from data for WebSocket authentication
            token = data.get('token') if data else None
            if not token:
                socketio.emit('processing_error', {
                    'message': 'Authentication token required'
                })
                return
                
            # Manually verify the JWT token
            try:
                from flask_jwt_extended import decode_token
                decoded_token = decode_token(token)
                user = decoded_token['sub']  # 'sub' contains the identity
                logger.info(f"User {user} requested to start video processing")
            except Exception as e:
                logger.error(f"Token verification failed: {e}")
                socketio.emit('processing_error', {
                    'message': 'Invalid authentication token'
                })
                return
            
            junction = data.get('junction', 'junction1') if data else 'junction1'
            
            # Check if processing is already active
            if is_processing_active():
                socketio.emit('processing_error', {
                    'message': 'Video processing is already active',
                    'user': user
                })
                return
                
            # Validate junction exists in config
            if hasattr(Config, 'JUNCTIONS') and junction not in Config.JUNCTIONS:
                socketio.emit('processing_error', {
                    'message': f'Unknown junction: {junction}',
                    'user': user
                })
                return
                
            video_path = getattr(Config, 'JUNCTIONS', {}).get(junction, Config.VIDEO_PATH)
            
            if not os.path.exists(video_path):
                socketio.emit('processing_error', {
                    'message': f'Video file not found for junction: {junction}',
                    'user': user
                })
                return
                
            logging.info(f"Starting video processing for junction: {junction}")
            session = Session()
            
            # Start processing
            success = start_video_processing(None, None, session, junction, video_path)
            
            if success:
                socketio.emit('processing_started', {
                    'message': f'Video processing started for {junction}',
                    'junction': junction,
                    'user': user,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                socketio.emit('processing_error', {
                    'message': 'Failed to start video processing',
                    'user': user
                })
            
        except Exception as e:
            logging.error(f"Error starting video processing: {e}")
            socketio.emit('processing_error', {
                'message': f'Failed to start video processing: {str(e)}'
            })

    @socketio.on('stop_processing')
    def handle_stop_processing(data):
        """Handle video processing stop request via WebSocket."""
        try:
            # Extract token from data for WebSocket authentication
            token = data.get('token') if data else None
            if not token:
                socketio.emit('processing_error', {
                    'message': 'Authentication token required'
                })
                return
                
            # Manually verify the JWT token
            try:
                from flask_jwt_extended import decode_token
                decoded_token = decode_token(token)
                user = decoded_token['sub']  # 'sub' contains the identity
                logger.info(f"User {user} requested to stop video processing")
            except Exception as e:
                logger.error(f"Token verification failed: {e}")
                socketio.emit('processing_error', {
                    'message': 'Invalid authentication token'
                })
                return
            
            if not is_processing_active():
                socketio.emit('processing_error', {
                    'message': 'Video processing is not currently active',
                    'user': user
                })
                return
            
            success = stop_video_processing()
            
            if success:
                socketio.emit('processing_stopped', {
                    'message': 'Video processing stopped successfully',
                    'user': user,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                socketio.emit('processing_error', {
                    'message': 'Failed to stop video processing',
                    'user': user
                })
                
        except Exception as e:
            logging.error(f"Error stopping video processing: {e}")
            socketio.emit('processing_error', {
                'message': f'Failed to stop video processing: {str(e)}'
            })

    @socketio.on('get_processing_status')
    def handle_get_processing_status(data=None):
        """Get current processing status."""
        try:
            # Verify user authentication
            verify_jwt_in_request()
            user = get_jwt_identity()
            
            socketio.emit('processing_status', {
                'active': is_processing_active(),
                'user': user,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logging.error(f"Error getting processing status: {e}")
            socketio.emit('auth_required', {
                'message': 'Authentication required to get status'
            })

    @socketio.on('stop_server')
    def handle_stop_server(data):
        """Handle server shutdown request."""
        try:
            # Verify user authentication
            verify_jwt_in_request()
            user = get_jwt_identity()
            
            logging.info(f"Server shutdown requested by user: {user}")
            
            # Emit server shutdown notification to all clients
            socketio.emit('server_shutdown', {
                'message': 'Server is shutting down',
                'user': user,
                'timestamp': datetime.now().isoformat()
            })
            
            # Stop any active processing
            if is_processing_active():
                stop_video_processing()
            
            # Schedule server shutdown after a brief delay
            import threading
            def shutdown_server():
                import time
                time.sleep(2)  # Give time for the response to be sent
                import os
                os._exit(0)  # Force shutdown
            
            shutdown_thread = threading.Thread(target=shutdown_server)
            shutdown_thread.daemon = True
            shutdown_thread.start()
            
        except Exception as e:
            logging.error(f"Error handling server stop: {e}")
            socketio.emit('auth_required', {
                'message': 'Authentication required to stop server'
            })

    @socketio.on('connect')
    def handle_connect(auth=None):
        """Handle WebSocket connection."""
        try:
            verify_jwt_in_request()
            user = get_jwt_identity()
            logger.info(f"User {user} connected to WebSocket")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        logger.info("User disconnected from WebSocket")
