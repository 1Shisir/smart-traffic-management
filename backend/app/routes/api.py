"""
Clean API routes with simplified Socket.IO communication (no authentication for sockets)
"""

import logging
import os
import cv2
import tempfile
import threading
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, make_response, redirect, url_for
from flask_jwt_extended import (
    jwt_required, get_jwt_identity, set_access_cookies, 
    unset_jwt_cookies, decode_token
)
from flask_socketio import emit
from sqlalchemy import desc
import torch
from ultralytics import YOLO

# Local imports
from app.config import Config
from app.models.traffic_data import TrafficData
from app.models.user import User
from app.services import AuthService, TrafficDataService
from app.utils.video_processor import start_video_processing, stop_video_processing, is_processing_active, detect_vehicles
from ..utils.realtime_polling import write_realtime_data, read_realtime_data
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
def get_data():
    """Get recent traffic data for dashboard (public access)."""
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
def get_current_status():
    """Get current traffic status and system state (public access)."""
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

@api.route('/video-with-detection')
def video_with_detection():
    """Process and serve video with vehicle detection boundaries - simplified approach."""
    try:
        # Get video path from query parameter or use default
        video_path = request.args.get('video_path', Config.VIDEO_PATH)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Create temporary file for processed video
        temp_dir = tempfile.mkdtemp()
        timestamp = int(time.time())
        output_filename = f'detected_video_{timestamp}.avi'
        output_path = os.path.join(temp_dir, output_filename)
        
        logger.info(f"Starting simplified video processing: {video_path} -> {output_path}")
        
        try:
            # Process video with simplified approach
            success = process_video_simple_method(video_path, output_path)
            
            if not success:
                # Cleanup and return error
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                return jsonify({'error': 'Failed to process video - check backend logs for details'}), 500
            
            # Check if output file exists and has reasonable size
            if not os.path.exists(output_path):
                return jsonify({'error': 'Processed video file was not created'}), 500
            
            file_size = os.path.getsize(output_path)
            if file_size < 10000:  # Less than 10KB indicates failure
                return jsonify({'error': 'Processed video file is too small - processing may have failed'}), 500
            
            logger.info(f"Video processing completed successfully. File size: {file_size} bytes")
            
            return send_file(
                output_path,
                mimetype='video/x-msvideo',
                as_attachment=True,
                download_name=f'traffic_detected_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
            )
            
        except Exception as e:
            # Cleanup on error
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass
            logger.error(f"Error during video processing: {e}")
            return jsonify({'error': f'Video processing failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in video_with_detection route: {e}")
        return jsonify({'error': 'Failed to process video with detection'}), 500

def process_video_simple_method(input_path, output_path):
    """Simplified video processing that skips frames for better performance."""
    try:
        logger.info(f"Starting simplified processing: {input_path}")
        
        # Initialize YOLO model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        model = YOLO(Config.YOLO_MODEL).to(device)
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Process every 15th frame for speed (reduces 51k frames to ~3.4k)
        frame_skip = 15
        output_fps = max(fps / frame_skip, 5)  # Minimum 5 FPS
        
        # Use most reliable codec
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        if not out.isOpened():
            logger.error("Cannot create video writer")
            cap.release()
            return False
        
        logger.info(f"Created writer: MJPG codec, {output_fps}fps")
        
        frame_count = 0
        written_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                continue
            
            try:
                # Simple resize to ensure exact dimensions
                frame = cv2.resize(frame, (width, height))
                
                # Detect vehicles
                total_count, class_counts, annotated_frame = detect_vehicles(frame, model, vehicle_labels)
                
                # Ensure annotated frame is correct size
                annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Add simple text overlay
                text = f"Frame {frame_count} - Vehicles: {total_count}"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame
                out.write(annotated_frame)
                written_count += 1
                
                if written_count % 100 == 0:
                    logger.info(f"Written {written_count} frames from {frame_count} total")
                    
            except Exception as e:
                logger.warning(f"Error processing frame {frame_count}: {e}")
                # Write original frame on error
                try:
                    frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                    written_count += 1
                except:
                    pass
        
        # Cleanup
        cap.release()
        out.release()
        
        if model:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Completed: {written_count} frames written")
        
        # Verify output
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            logger.info(f"Output file size: {size} bytes")
            return size > 10000  # Must be at least 10KB
        
        return False
        
    except Exception as e:
        logger.error(f"Error in process_video_simple_method: {e}")
        return False

def process_video_with_boundaries(input_path, output_path):
    """Process video file and add detection boundaries."""
    try:
        # Initialize YOLO model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(Config.YOLO_MODEL).to(device)
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {input_path}")
            return False
        
        # Get video properties and ensure they're valid
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure FPS is valid
        if fps <= 0 or fps > 60:
            fps = 25.0  # Default to 25 FPS
        
        # Ensure dimensions are even numbers (required by many codecs)
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Try different codecs in order of preference (most compatible first)
        output_created = False
        codecs_to_try = [
            ('XVID', 'XVID'),  # Xvid MPEG-4 - most compatible
            ('MJPG', 'MJPG'),  # Motion JPEG - very reliable
            ('mp4v', 'MP4V'),  # MPEG-4 Part 2
            ('X264', 'H264'),  # H.264 - last resort
        ]
        
        out = None
        for codec_name, codec_fourcc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    logger.info(f"Successfully created video writer with codec: {codec_name}")
                    output_created = True
                    break
                else:
                    if out:
                        out.release()
                    logger.warning(f"Failed to create video writer with codec: {codec_name}")
            except Exception as e:
                logger.warning(f"Error with codec {codec_name}: {e}")
                if out:
                    out.release()
        
        if not output_created:
            logger.error("Failed to create video writer with any codec")
            cap.release()
            return False
        
        logger.info(f"Processing {total_frames} frames...")
        
        frame_count = 0
        successful_writes = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # Ensure frame has correct dimensions
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                # Detect vehicles and get annotated frame
                total_count, class_counts, annotated_frame = detect_vehicles(frame, model, vehicle_labels)
                
                # Ensure annotated frame has correct dimensions
                if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Add frame counter and detection info with better positioning
                info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {total_count}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black outline
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text
                
                # Add vehicle counts by type
                y_offset = 55
                for vehicle_type, count in class_counts.items():
                    if count > 0:
                        count_text = f"{vehicle_type.capitalize()}: {count}"
                        cv2.putText(annotated_frame, count_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
                        cv2.putText(annotated_frame, count_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # White text
                        y_offset += 20
                
                # Ensure frame is in correct format (BGR)
                if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                    # Write frame to output video
                    success = out.write(annotated_frame)
                    if success:
                        successful_writes += 1
                    else:
                        logger.warning(f"Failed to write frame {frame_count}")
                else:
                    logger.warning(f"Frame {frame_count} has incorrect format: {annotated_frame.shape}")
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    success_rate = (successful_writes / frame_count) * 100
                    logger.info(f"Processed {frame_count}/{total_frames} frames (Success rate: {success_rate:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                # Try to write original frame if detection fails
                try:
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                except Exception as write_error:
                    logger.error(f"Failed to write original frame {frame_count}: {write_error}")
        
        # Release everything
        cap.release()
        out.release()
        
        # Clean up model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check if output file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
            final_success_rate = (successful_writes / frame_count) * 100 if frame_count > 0 else 0
            logger.info(f"Video processing completed successfully!")
            logger.info(f"Total frames processed: {frame_count}")
            logger.info(f"Successful writes: {successful_writes} ({final_success_rate:.1f}%)")
            logger.info(f"Output file size: {os.path.getsize(output_path)} bytes")
            return True
        else:
            logger.error("Output video file was not created properly")
            return False
        
    except Exception as e:
        logger.error(f"Error in process_video_with_boundaries: {e}")
        return False

@api.route('/video-detection-preview')
def video_detection_preview():
    """Get a single frame from video with detection boundaries for preview."""
    try:
        # Get video path and frame number from query parameters
        video_path = request.args.get('video_path', Config.VIDEO_PATH)
        frame_number = int(request.args.get('frame', 0))
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Initialize YOLO model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(Config.YOLO_MODEL).to(device)
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        # Open video and seek to frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video file'}), 500
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = min(frame_number, total_frames - 1)
        
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return jsonify({'error': 'Failed to read frame'}), 500
        
        # Detect vehicles and get annotated frame
        total_count, class_counts, annotated_frame = detect_vehicles(frame, model, vehicle_labels)
        
        # Add detection info to frame
        info_text = f"Frame: {frame_number}/{total_frames} | Vehicles: {total_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', annotated_frame)
        if not success:
            cap.release()
            return jsonify({'error': 'Failed to encode frame'}), 500
        
        # Cleanup
        cap.release()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return image
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-cache'
        return response
        
    except Exception as e:
        logger.error(f"Error generating detection preview: {e}")
        return jsonify({'error': 'Failed to generate preview'}), 500

@api.route('/video-processing-status')
def video_processing_status():
    """Get the current status of video processing."""
    try:
        # This is a simple status endpoint
        # In a production system, you'd want to store processing status in a database or cache
        
        # For now, we'll check if there are any temp files being processed
        temp_dir = tempfile.gettempdir()
        processing_files = []
        
        try:
            for filename in os.listdir(temp_dir):
                if filename.startswith('detected_video_') and filename.endswith('.mp4'):
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        # Estimate if file is still being written to
                        processing_files.append({
                            'filename': filename,
                            'size': file_size,
                            'created': os.path.getctime(file_path)
                        })
        except Exception as e:
            logger.warning(f"Error checking processing status: {e}")
        
        return jsonify({
            'processing_active': len(processing_files) > 0,
            'processing_files': len(processing_files),
            'message': 'Processing in progress...' if processing_files else 'No active processing'
        })
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return jsonify({'error': 'Failed to get processing status'}), 500

@api.route('/realtime-status')
def get_realtime_status():
    """Get current real-time status for polling (much more reliable than Socket.IO)"""
    try:
        data = read_realtime_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting realtime status: {e}")
        return jsonify({'error': str(e)}), 500


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
            session = Session()
            
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
