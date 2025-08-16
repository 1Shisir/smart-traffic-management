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
        
        # Ensure we get strings, not other data types
        username_raw = data.get('username', '')
        password_raw = data.get('password', '')
        
        # Convert to strings and strip if they are strings
        if isinstance(username_raw, str):
            username = username_raw.strip()
        else:
            return jsonify({'message': 'Username must be a string'}), 400
            
        if isinstance(password_raw, str):
            password = password_raw
        else:
            return jsonify({'message': 'Password must be a string'}), 400
        
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
    
    @socketio.on('connect')
    def handle_connect(auth):
        """Handle WebSocket connection with JWT authentication - for processing sessions only."""
        try:
            # Authenticate on every connection
            if not auth or 'token' not in auth:
                logger.error("🔌 WebSocket connection rejected: No token provided")
                return False
            
            token = auth['token']
            
            # Verify JWT token manually
            from flask_jwt_extended import decode_token
            try:
                decoded_token = decode_token(token)
                user_id = decoded_token['sub']
                logger.info(f"🔌 Processing session: User {user_id} connected via WebSocket")
            except Exception as e:
                logger.error(f"🔌 WebSocket JWT verification failed: {e}")
                return False
            
            # Store user info in socket sessions dictionary
            socket_sessions[request.sid] = user_id
            
            logger.info(f"🔌 Processing socket connected for user {user_id} (Socket: {request.sid})")
            logger.info(f"🔍 Updated socket_sessions: {socket_sessions}")
            
            return True
            
        except Exception as e:
            logger.error(f"🔌 WebSocket connection error: {e}")
            return False

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection - processing session ended."""
        user_id = socket_sessions.get(request.sid, 'unknown')
        logger.info(f"🔌 Processing session ended: User {user_id} disconnected (Socket: {request.sid})")
        logger.info(f"🔍 Socket_sessions before cleanup: {socket_sessions}")
        
        # Remove from socket sessions
        socket_sessions.pop(request.sid, None)
        logger.info(f"🔍 Socket_sessions after cleanup: {socket_sessions}")

    @socketio.on('test_connection')
    def handle_test_connection(data):
        """Handle test connection from frontend - simplified for processing sessions."""
        try:
            # Get user from socket sessions
            user_id = socket_sessions.get(request.sid)
            if user_id:
                user_room = f"user_{user_id}"
                logger.info(f"🧪 Test: User {user_id} testing processing connection")
                emit('test_response', {
                    'message': 'Processing session connection OK',
                    'user': user_id,
                    'room': user_room,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.info(f"🧪 Test: Anonymous connection test")
                emit('test_response', {
                    'message': 'Connection OK (not authenticated)',
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"🧪 Test connection error: {e}")
            emit('test_response', {
                'message': f'Test failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    @socketio.on('join_room')
    def handle_join_room(data):
        """Handle room join for processing session."""
        try:
            # Get user from socket sessions
            user_id = socket_sessions.get(request.sid)
            if not user_id:
                logger.error("🏠 Join room failed: No authenticated user for this socket")
                emit('join_error', {'message': 'Not authenticated'})
                return
            
            # Create user-specific room
            user_room = f"user_{user_id}"
            join_room(user_room)
            
            logger.info(f"🏠 Processing session: User {user_id} joined room {user_room}")
            logger.info(f"🏠 Socket ID: {request.sid}")
            
            # Confirm room join
            emit('room_joined', {
                'message': f'Joined processing room {user_room}',
                'user_room': user_room,
                'user': user_id
            })
            
        except Exception as e:
            logger.error(f"🏠 Join room error: {e}")
            emit('join_error', {'message': str(e)})
    
    @socketio.on('start_processing')
    def handle_start_processing(data):
        """Handle video processing start request via WebSocket."""
        try:
            # Debug: Show current socket sessions
            logger.info(f"🔍 Current socket_sessions: {socket_sessions}")
            logger.info(f"🔍 Current request.sid: {request.sid}")
            
            # Get user from socket sessions
            user_id = socket_sessions.get(request.sid)
            if not user_id:
                logger.error(f"🚀 Start processing failed: No authenticated user for this socket {request.sid}")
                logger.error(f"🚀 Available sessions: {list(socket_sessions.keys())}")
                emit('processing_error', {'message': 'Not authenticated'})
                return
            
            user_room = f"user_{user_id}"
            logger.info(f"🚀 Processing start: User {user_id} requested video processing")
            
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
                
            logger.info(f"🚀 Starting video processing for junction: {junction} (User: {user_id})")
            db_session = Session()
            
            # Ensure user is in the correct room for this processing session
            join_room(user_room)
            logger.info(f"🏠 Processing: User {user_id} ensured in room {user_room} (Socket: {request.sid})")
            
            # Start processing with socketio instance and user room
            success = start_video_processing(None, socketio, db_session, junction, video_path, user_room)
            
            if success:
                logger.info(f"🚀 Processing started successfully for user {user_id}")
                emit('processing_started', {
                    'message': f'Video processing started for {junction}',
                    'junction': junction,
                    'user': user_id,
                    'timestamp': datetime.now().isoformat()
                }, room=user_room)
            else:
                logger.error(f"🚀 Processing failed to start for user {user_id}")
                emit('processing_error', {
                    'message': 'Failed to start video processing',
                    'user': user_id
                }, room=user_room)
            
        except Exception as e:
            logger.error(f"🚀 Error starting video processing: {e}")
            emit('processing_error', {
                'message': f'Failed to start video processing: {str(e)}'
            })

    @socketio.on('stop_processing')
    def handle_stop_processing(data):
        """Handle video processing stop request via WebSocket."""
        try:
            # Get user from socket sessions
            user_id = socket_sessions.get(request.sid)
            if not user_id:
                logger.error("🛑 Stop processing failed: No authenticated user for this socket")
                emit('processing_error', {'message': 'Not authenticated'})
                return
                
            user_room = f"user_{user_id}"
            logger.info(f"🛑 Processing stop: User {user_id} requested to stop video processing")
            
            if not is_processing_active():
                emit('processing_error', {
                    'message': 'Video processing is not currently active',
                    'user': user_id
                }, room=user_room)
                return
            
            # Stop video processing
            logger.info(f"🛑 API: Calling stop_video_processing for user {user_id} in room {user_room}")
            success = stop_video_processing(user_room=user_room)
            logger.info(f"🛑 API: stop_video_processing returned: {success}")
            
            if success:
                logger.info(f"🛑 API: Stop video processing successful for user {user_id} in room {user_room}")
                # Emit processing_stopped event directly from the API handler
                stop_data = {
                    'message': 'Video processing stopped',
                    'timestamp': datetime.now().isoformat(),
                    'user': user_id
                }
                logger.info(f"🛑 API: About to emit processing_stopped to room: {user_room}")
                socketio.emit('processing_stopped', stop_data, room=user_room)
                # Also broadcast globally as a fallback
                socketio.emit('processing_stopped_global', stop_data)
                logger.info(f"🛑 API: Emitted processing_stopped to room: {user_room} and globally")
            else:
                logger.warning(f"🛑 API: Stop video processing failed for user {user_id}")
                socketio.emit('processing_error', {
                    'message': 'Failed to stop video processing',
                    'user': user_id
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
            
            # Check if user can be found via socket ID (simplified - no session tracking)
            if not user_id:
                logger.info(f"📊 STATUS: No user identified for socket {request.sid}")
                # Continue with anonymous status check
            
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
