# Standard library imports
import logging
import os
import threading
from datetime import datetime

# Third-party imports
from flask import Blueprint, jsonify, send_file, request, make_response, redirect, url_for
from flask_jwt_extended import jwt_required, create_access_token, unset_jwt_cookies, set_access_cookies, get_jwt_identity, verify_jwt_in_request
from flask_socketio import join_room, leave_room, emit
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

@api.route('/login', methods=['POST'])
def login():
    """API endpoint for user authentication."""
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
def get_data():
    """Get recent traffic data for dashboard."""
    try:
        # Enhanced input validation with strict limits
        limit = request.args.get('limit', 10, type=int)
        limit = max(1, min(limit, 100))  # Enforce strict limits: 1-100 records
        
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
    """Register SocketIO event handlers with user room management"""
    
    # Store active user sessions
    user_sessions = {}  # {user_id: {'socket_id': id, 'room': room_name, 'processing': bool}}
    
    @socketio.on('connect')
    def handle_connect(auth):
        """Handle WebSocket connection with JWT authentication."""
        try:
            # Authenticate on every connection
            if not auth or 'token' not in auth:
                logger.error("WebSocket connection rejected: No token provided")
                return False
            
            token = auth['token']
            
            # Verify JWT token manually
            from flask_jwt_extended import decode_token
            try:
                decoded_token = decode_token(token)
                user_id = decoded_token['sub']
                logger.info(f"User {user_id} authenticated via WebSocket")
            except Exception as e:
                logger.error(f"WebSocket JWT verification failed: {e}")
                return False
            
            # Create user-specific room
            user_room = f"user_{user_id}"
            join_room(user_room)
            
            logger.info(f"🏠 ========== ROOM JOINED ==========")
            logger.info(f"🏠 User ID: {user_id}")
            logger.info(f"🏠 Room name: {user_room}")
            logger.info(f"🏠 Socket ID: {request.sid}")
            logger.info(f"🏠 =================================")
            
            # Store session info
            user_sessions[user_id] = {
                'socket_id': request.sid,
                'room': user_room,
                'processing': user_sessions.get(user_id, {}).get('processing', False)  # Preserve processing state
            }
            
            logger.info(f"User {user_id} joined room {user_room} with socket {request.sid}")
            
            # If user was processing, notify about reconnection
            if user_sessions[user_id]['processing']:
                emit('processing_reconnected', {
                    'message': 'Reconnected to active processing session',
                    'processing_active': True
                }, room=user_room)
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        # Don't remove user session immediately - they might reconnect
        logger.info(f"Socket {request.sid} disconnected")
        # Session cleanup will happen on new connection or timeout

    @socketio.on('test_connection')
    def handle_test_connection(data):
        """Handle test connection from frontend with room support."""
        try:
            # Extract JWT token to identify user room
            from flask_jwt_extended import decode_token
            
            # Try to get token from multiple sources
            token = None
            
            # 1. Try from data payload (sent by frontend)
            if data and 'token' in data:
                token = data['token']
                logger.info(f"🧪 TEST: Token found in data payload")
            
            # 2. Try from auth data (connection auth)
            if not token:
                auth_data = getattr(request, 'event', {}).get('auth', {})
                token = auth_data.get('token') if auth_data else None
                if token:
                    logger.info(f"🧪 TEST: Token found in auth data")
            
            # 3. Try to find user session by socket ID
            if not token:
                for user_id, session_info in user_sessions.items():
                    if session_info.get('socket_id') == request.sid:
                        token = 'found_via_session'  # We know the user from session
                        user_room = f"user_{user_id}"
                        logger.info(f"🧪 TEST: User {user_id} found via session, room {user_room}")
                        emit('test_response', {
                            'message': 'Room-based test successful (session)',
                            'user_id': user_id,
                            'user_room': user_room,
                            'received_data': data,
                            'timestamp': datetime.now().isoformat()
                        }, room=user_room)
                        return
            
            if token and token != 'found_via_session':
                decoded_token = decode_token(token)
                user_id = decoded_token['sub']
                user_room = f"user_{user_id}"
                
                logger.info(f"🧪 TEST: User {user_id} testing room {user_room}")
                emit('test_response', {
                    'message': 'Room-based test successful',
                    'user_id': user_id,
                    'user_room': user_room,
                    'received_data': data,
                    'timestamp': datetime.now().isoformat()
                }, room=user_room)
            else:
                logger.info(f"🧪 TEST: Anonymous test connection - no token found")
                emit('test_response', {
                    'message': 'Test response (no auth)',
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Test connection error: {e}")
            emit('test_response', {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    @socketio.on('join_room')
    def handle_join_room(data):
        """Handle manual room join request."""
        try:
            token = data.get('token') if data else None
            requested_room = data.get('room') if data else None
            
            if token:
                from flask_jwt_extended import decode_token
                decoded_token = decode_token(token)
                user_id = decoded_token['sub']
                user_room = f"user_{user_id}"
                
                # Join the room
                join_room(user_room)
                
                logger.info(f"🏠 MANUAL JOIN: User {user_id} manually joined room {user_room}")
                logger.info(f"🏠 MANUAL JOIN: Socket ID {request.sid}")
                
                # Update session
                user_sessions[user_id] = {
                    'socket_id': request.sid,
                    'room': user_room,
                    'processing': user_sessions.get(user_id, {}).get('processing', False)
                }
                
                emit('room_joined', {
                    'message': f'Successfully joined room {user_room}',
                    'room': user_room,
                    'user_id': user_id
                })
                
            else:
                emit('room_join_error', {
                    'message': 'Token required for room join'
                })
                
        except Exception as e:
            logger.error(f"Manual room join error: {e}")
            emit('room_join_error', {
                'message': f'Failed to join room: {str(e)}'
            })
    
    @socketio.on('start_processing')
    def handle_start_processing(data):
        """Handle video processing start request via WebSocket."""
        try:
            # Extract token from data for WebSocket authentication
            token = data.get('token') if data else None
            if not token:
                emit('processing_error', {'message': 'Authentication token required'})
                return
                
            # Manually verify the JWT token and get user room
            try:
                from flask_jwt_extended import decode_token
                decoded_token = decode_token(token)
                user_id = decoded_token['sub']
                user_room = f"user_{user_id}"
                logger.info(f"User {user_id} requested to start video processing")
            except Exception as e:
                logger.error(f"Token verification failed: {e}")
                emit('processing_error', {'message': 'Invalid authentication token'})
                return
            
            junction = data.get('junction', 'junction1') if data else 'junction1'
            
            # Check if processing is already active
            if is_processing_active():
                emit('processing_error', {
                    'message': 'Video processing is already active',
                    'user': user_id
                }, room=user_room)
                return
                
            # Validate junction exists in config
            if hasattr(Config, 'JUNCTIONS') and junction not in Config.JUNCTIONS:
                emit('processing_error', {
                    'message': f'Unknown junction: {junction}',
                    'user': user_id
                }, room=user_room)
                return
                
            video_path = getattr(Config, 'JUNCTIONS', {}).get(junction, Config.VIDEO_PATH)
            
            if not os.path.exists(video_path):
                emit('processing_error', {
                    'message': f'Video file not found for junction: {junction}',
                    'user': user_id
                }, room=user_room)
                return
                
            logging.info(f"Starting video processing for junction: {junction}")
            session = Session()
            
            # Start processing with socketio instance
            # Update user session to mark as processing
            if user_id in user_sessions:
                user_sessions[user_id]['processing'] = True
            
            # Start processing with socketio instance and user room
            success = start_video_processing(None, socketio, session, junction, video_path, user_room)
            
            if success:
                emit('processing_started', {
                    'message': f'Video processing started for {junction}',
                    'junction': junction,
                    'user': user_id,
                    'timestamp': datetime.now().isoformat()
                }, room=user_room)
            else:
                emit('processing_error', {
                    'message': 'Failed to start video processing',
                    'user': user_id
                }, room=user_room)
            
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
                # Create user-specific room to emit to
                user_room = f"user_{user}"
                logger.info(f"🛑 Emitting processing_stopped to room: {user_room}")
                
                socketio.emit('processing_stopped', {
                    'message': f'Video processing stopped for {user}',
                    'user': user,
                    'timestamp': datetime.now().isoformat()
                }, room=user_room)
                
                logger.info(f"🛑 Successfully emitted processing_stopped to {user_room}")
            else:
                # Create user-specific room for error too
                user_room = f"user_{user}"
                socketio.emit('processing_error', {
                    'message': 'Failed to stop video processing',
                    'user': user
                }, room=user_room)
                
        except Exception as e:
            logging.error(f"Error stopping video processing: {e}")
            # Try to extract user info for room-based error emission
            try:
                token = data.get('token') if data else None
                if token:
                    from flask_jwt_extended import decode_token
                    decoded_token = decode_token(token)
                    user = decoded_token['sub']
                    user_room = f"user_{user}"
                    socketio.emit('processing_error', {
                        'message': f'Failed to stop video processing: {str(e)}'
                    }, room=user_room)
                else:
                    # No token, broadcast to all
                    socketio.emit('processing_error', {
                        'message': f'Failed to stop video processing: {str(e)}'
                    })
            except:
                # Fallback: broadcast to all
                socketio.emit('processing_error', {
                    'message': f'Failed to stop video processing: {str(e)}'
                })

    @socketio.on('get_processing_status')
    def handle_get_processing_status(data=None):
        """Get current processing status."""
        try:
            # Extract token from data for WebSocket authentication
            token = data.get('token') if data else None
            user_id = None
            user_room = None
            
            if token:
                try:
                    from flask_jwt_extended import decode_token
                    decoded_token = decode_token(token)
                    user_id = decoded_token['sub']
                    user_room = f"user_{user_id}"
                    logger.info(f"📊 STATUS: User {user_id} requesting processing status")
                except Exception as e:
                    logger.error(f"Token verification failed in status check: {e}")
            
            # Check if user can be found via session
            if not user_id:
                for uid, session_info in user_sessions.items():
                    if session_info.get('socket_id') == request.sid:
                        user_id = uid
                        user_room = f"user_{user_id}"
                        logger.info(f"📊 STATUS: User {user_id} found via session")
                        break
            
            processing_active = is_processing_active()
            
            response_data = {
                'processing_active': processing_active,
                'user': user_id,
                'timestamp': datetime.now().isoformat()
            }
            
            if user_room:
                # Send to specific user room
                emit('processing_status', response_data, room=user_room)
                logger.info(f"📊 STATUS: Sent to room {user_room}, processing={processing_active}")
            else:
                # Send to current socket
                emit('processing_status', response_data)
                logger.info(f"📊 STATUS: Sent to socket, processing={processing_active}")
            
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            emit('processing_status', {
                'processing_active': False,
                'error': 'Failed to get status',
                'timestamp': datetime.now().isoformat()
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
    def handle_disconnect(reason=None):
        """Handle WebSocket disconnection."""
        try:
            verify_jwt_in_request()
            user = get_jwt_identity()
            if user:
                logger.info(f"User {user} disconnected from WebSocket (reason: {reason})")
            else:
                logger.info(f"Anonymous user disconnected from WebSocket (reason: {reason})")
        except Exception:
            # JWT verification might fail during disconnect, which is normal
            logger.info(f"User disconnected from WebSocket (reason: {reason})")
