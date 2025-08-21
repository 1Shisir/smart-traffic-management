import cv2
import requests
import os
import logging
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import torch
from ultralytics import YOLO
from app.models.traffic_data import TrafficData
from app.config import Config
from .realtime_polling import write_realtime_data, read_realtime_data
from ..services.aws_service import aws_storage

# Global variables for process control
processing_active = False
stop_event = threading.Event()
frame_for_preview = None
current_user_room = None  # Track the current user room for proper event targeting

# SocketIO instance will be set by the main app
socketio = None

def set_socketio(socketio_instance):
    """Set the SocketIO instance for real-time updates."""
    global socketio
    socketio = socketio_instance

def detect_vehicles(frame, model, vehicle_labels):
    """Detect vehicles in frame with proper memory management."""
    try:
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Run YOLO detection with memory optimization
        results = model(frame, verbose=False)[0]
        class_counts = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}

        # Process detections with confidence threshold
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Only process high-confidence detections
            if name in vehicle_labels and conf > 0.5:
                class_counts[name] += 1
                
                # Draw bounding boxes with proper coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)  # Ensure coordinates are positive
                x2, y2 = min(640, x2), min(480, y2)  # Ensure coordinates are within frame
                
                label = f"{name} ({int(conf*100)}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_count = sum(class_counts.values())
        
        # Validate counts are reasonable (prevent integer overflow)
        total_count = min(total_count, 1000)  # Max 1000 vehicles per frame
        for key in class_counts:
            class_counts[key] = min(class_counts[key], 1000)
        
        return total_count, class_counts, frame
        
    except Exception as e:
        logging.error(f"YOLO detection error: {e}")
        # Return safe defaults on error
        return 0, {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}, frame

def get_traffic_light_state(total_count):
    """Determine traffic light state based on vehicle count."""
    if total_count >= Config.TRAFFIC_LIGHT_THRESHOLD:
        return "red", 40
    elif total_count >= 8:
        return "yellow", 5
    else:
        return "green", 30

def start_video_processing(app, socketio_param, session, junction="main_junction", video_path=None, user_room=None):
    """Start video processing using Flask-SocketIO background task."""
    global processing_active, stop_event, socketio, current_user_room
    
    if processing_active:
        logging.warning("Video processing is already active")
        return False
    
    stop_event.clear()
    processing_active = True
    socketio = socketio_param  # Assign the socketio parameter to global variable
    current_user_room = user_room  # Store the current user room globally
    
    # Use regular threading since Flask-SocketIO background task might be causing issues
    processing_thread = threading.Thread(
        target=process_video_with_context,
        args=(socketio_param, session, junction, video_path, user_room),
        daemon=True
    )
    processing_thread.start()
    
    logging.info(f"Started video processing thread for {junction} (user room: {user_room})")
    return True

def process_video_with_context(socketio_instance, session, junction="main_junction", video_path=None, user_room=None):
    """Wrapper function to call process_video with proper context."""
    try:
        # Set the global socketio instance
        global socketio, current_user_room
        socketio = socketio_instance
        current_user_room = user_room  # Ensure the user room is available globally
        
        # Call the actual video processing function
        process_video(session, junction, video_path, user_room)
    except Exception as e:
        logging.error(f"Error in video processing wrapper: {e}")
        global processing_active
        processing_active = False
current_user_room = None  # Track the current user room for proper event targeting

# SocketIO instance will be set by the main app
socketio = None

def set_socketio(socketio_instance):
    """Set the SocketIO instance for real-time updates."""
    global socketio
    socketio = socketio_instance

def detect_vehicles(frame, model, vehicle_labels):
    """Detect vehicles in frame with proper memory management."""
    try:
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Run YOLO detection with memory optimization
        results = model(frame, verbose=False)[0]
        class_counts = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}

        # Process detections with confidence threshold
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Only process high-confidence detections
            if name in vehicle_labels and conf > 0.5:
                class_counts[name] += 1
                
                # Draw bounding boxes with proper coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)  # Ensure coordinates are positive
                x2, y2 = min(640, x2), min(480, y2)  # Ensure coordinates are within frame
                
                label = f"{name} ({int(conf*100)}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_count = sum(class_counts.values())
        
        # Validate counts are reasonable (prevent integer overflow)
        total_count = min(total_count, 1000)  # Max 1000 vehicles per frame
        for key in class_counts:
            class_counts[key] = min(class_counts[key], 1000)
        
        return total_count, class_counts, frame
        
    except Exception as e:
        logging.error(f"YOLO detection error: {e}")
        # Return safe defaults on error
        return 0, {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}, frame

def get_traffic_light_state(total_count):
    """Determine traffic light state based on vehicle count."""
    if total_count >= Config.TRAFFIC_LIGHT_THRESHOLD:
        return "red", 40
    elif total_count >= 8:
        return "yellow", 5
    else:
        return "green", 30

def start_video_processing(app, socketio_param, session, junction="main_junction", video_path=None, user_room=None):
    """Start video processing using Flask-SocketIO background task."""
    global processing_active, stop_event, socketio, current_user_room
    
    if processing_active:
        logging.warning("Video processing is already active")
        return False
    
    stop_event.clear()
    processing_active = True
    socketio = socketio_param  # Assign the socketio parameter to global variable
    current_user_room = user_room  # Store the current user room globally
    
    # Use regular threading since Flask-SocketIO background task might be causing issues
    processing_thread = threading.Thread(
        target=process_video_with_context,
        args=(socketio_param, session, junction, video_path, user_room),
        daemon=True
    )
    processing_thread.start()
    
    logging.info(f"Started video processing thread for {junction} (user room: {user_room})")
    return True

def process_video_with_context(socketio_instance, session, junction="main_junction", video_path=None, user_room=None):
    """Wrapper function to call process_video with proper context."""
    try:
        # Set the global socketio instance
        global socketio
        socketio = socketio_instance
        
        # Call the actual video processing function
        process_video(session, junction, video_path, user_room)
    except Exception as e:
        logging.error(f"Error in video processing wrapper: {e}")
        global processing_active
        processing_active = False

def stop_video_processing(user_room=None):
    """Stop video processing and emit stop event to the appropriate room."""
    global processing_active, stop_event, socketio
    
    logging.info(f"🛑 VideoProcessor: stop_video_processing called with user_room={user_room}")
    logging.info(f"🛑 VideoProcessor: processing_active={processing_active}")
    
    if not processing_active:
        logging.warning("🛑 VideoProcessor: Video processing is not active")
        return False
    
    stop_event.set()
    processing_active = False
    logging.info("🛑 VideoProcessor: Video processing stop requested, flags set")
    
    # Emit processing_stopped event to the appropriate room
    if socketio:
        logging.info(f"🛑 VideoProcessor: socketio instance available, preparing to emit")
        stop_data = {
            'message': 'Video processing stopped',
            'timestamp': datetime.now().isoformat()
        }
        
        # Write stop status to polling file
        try:
            write_realtime_data({
                'processing_active': False,
                'status': 'stopped',
                'message': 'Video processing stopped',
                'timestamp': stop_data['timestamp']
            })
            logging.info(f"📝 Written stop status for HTTP polling")
        except Exception as write_error:
            logging.warning(f"📝 ⚠️ Failed to write stop status: {write_error}")
        
        if user_room:
            logging.info(f"🛑 VideoProcessor: Emitting processing_stopped to room: {user_room}")
            socketio.emit('processing_stopped', stop_data, room=user_room)
        else:
            logging.info(f"🛑 VideoProcessor: Broadcasting processing_stopped to all clients")
            socketio.emit('processing_stopped', stop_data)
    else:
        logging.warning(f"🛑 VideoProcessor: socketio instance is None, cannot emit event")
    
    logging.info(f"🛑 VideoProcessor: stop_video_processing returning True")
    return True

def is_processing_active():
    """Check if video processing is currently active."""
    return processing_active

def get_frame_for_preview():
    """Get the current frame for preview (base64 encoded)."""
    global frame_for_preview
    
    if frame_for_preview is None:
        return None
    
    try:
        # Encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame_for_preview)
        if success:
            import base64
            # Convert to base64 string
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{jpg_as_text}"
        else:
            return None
    except Exception as e:
        logging.error(f"Error encoding frame for preview: {e}")
        return None

def process_video(session, junction="main_junction", video_path=None, user_room=None):
    """Process video and emit real-time updates via WebSocket with proper resource management."""
    global frame_for_preview, processing_active, socketio
    
    # Debug: Check parameters and socketio availability
    logging.info(f"🔧 Starting video processing with:")
    logging.info(f"🔧 - Junction: {junction}")
    logging.info(f"🔧 - Video path: {video_path}")
    logging.info(f"🔧 - User room: {user_room}")
    logging.info(f"🔧 - SocketIO available: {socketio is not None}")
    
    if not user_room:
        logging.warning(f"⚠️ WARNING: No user_room provided! Events will be broadcasted to all clients.")
    
    entries_buffer = []
    cap = None
    model = None
    
    try:
        if video_path is None:
            video_path = Config.VIDEO_PATH
            
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            if socketio:
                socketio.emit('processing_error', {
                    'message': f'Video file not found: {video_path}'
                })
            processing_active = False
            return

        # Initialize YOLO model with proper error handling
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {device}")
            model = YOLO(Config.YOLO_MODEL).to(device)
            vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            if socketio:
                socketio.emit('processing_error', {
                    'message': f'Failed to load AI model: {str(e)}'
                })
            processing_active = False
            return

        # Initialize video capture with validation
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Error opening video file")
            if socketio:
                socketio.emit('processing_error', {
                    'message': 'Failed to open video file'
                })
            processing_active = False
            return

        # Get video properties with safety checks
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = max(1, fps) if fps and fps > 0 else 25  # Default to 25 fps if invalid
        
        logging.info(f"Processing video: {video_path} ({total_frames} frames, {fps} fps)")

        if socketio:
            socketio.emit('processing_started', {
                'message': f'Video processing started for {junction}',
                'junction': junction,
                'total_frames': total_frames,
                'fps': fps
            })

        frame_count = 0
        last_commit_time = datetime.now()
        
        while processing_active and not stop_event.is_set():
            ret, frame = cap.read()
            
            if not ret:
                # End of video - restart from beginning for continuous processing
                logging.info(f"Restarting video loop for {junction}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            
            # Process every 5th frame for better performance
            if frame_count % 5 != 0:
                continue

            try:
                # Vehicle detection with error handling
                total_count, class_counts, annotated = detect_vehicles(frame, model, vehicle_labels)
                
                # Update global frame for preview endpoint (create copy to prevent memory issues)
                if annotated is not None:
                    frame_for_preview = annotated.copy()
                
                timestamp = datetime.now()
                light_state, light_duration = get_traffic_light_state(total_count)
                
                logging.info(f"Frame {frame_count}: Vehicles -> Total: {total_count}, Traffic Light: {light_state}")

                # Save to database with error handling
                traffic_entry = TrafficData(
                    junction=junction[:50],  # Limit string length
                    total_count=min(total_count, 1000),  # Limit max count
                    car_count=min(class_counts['car'], 1000),
                    bus_count=min(class_counts['bus'], 1000),
                    truck_count=min(class_counts['truck'], 1000),
                    motorcycle_count=min(class_counts['motorcycle'], 1000),
                    traffic_light=light_state,
                    light_duration=light_duration,
                    timestamp=timestamp
                )
                
                entries_buffer.append(traffic_entry)

                # Commit to database in batches with time-based commits
                current_time = datetime.now()
                if (len(entries_buffer) >= 5 or 
                    (current_time - last_commit_time).seconds >= 30):  # Commit every 30 seconds
                    try:
                        session.bulk_save_objects(entries_buffer)
                        session.commit()
                        entries_buffer = []
                        last_commit_time = current_time
                    except Exception as db_error:
                        logging.error(f"Database commit error: {db_error}")
                        session.rollback()
                        entries_buffer = []  # Clear buffer to prevent memory buildup

                # Real-time updates via SocketIO to user room
                if socketio:
                    logging.info(f"📡 Emitting SocketIO update for frame {frame_count}")
                    
                    update_data = {
                        'junction': junction,
                        'count': total_count,
                        'car': class_counts['car'],
                        'bus': class_counts['bus'],
                        'truck': class_counts['truck'],
                        'motorcycle': class_counts['motorcycle'],
                        'traffic_light': light_state,
                        'light_duration': light_duration,
                        'timestamp': timestamp.isoformat(),
                        'time': timestamp.strftime("%H:%M:%S"),
                        'frame_count': frame_count,
                        'total_frames': total_frames
                    }
                    
                    traffic_light_data = {
                        'state': light_state,
                        'duration': light_duration
                    }
                    
                # ✅ WRITE DATA FOR HTTP POLLING (much more reliable than Socket.IO)
                try:
                    # Write real-time data to file for frontend polling
                    realtime_data = {
                        'timestamp': update_data['timestamp'],
                        'car_count': update_data['car'],
                        'truck_count': update_data['truck'],
                        'bus_count': update_data['bus'],
                        'motorcycle_count': update_data['motorcycle'],
                        'total_vehicles': update_data['count'],
                        'frame_count': update_data['frame_count'],
                        'total_frames': update_data['total_frames'],
                        'traffic_light_state': update_data['traffic_light'],
                        'traffic_light_duration': update_data['light_duration'],
                        'processing_active': True,
                        'junction': update_data['junction']
                    }
                    
                    write_realtime_data(realtime_data)
                    logging.info(f"📝 Written real-time data for HTTP polling: frame {frame_count}")
                    
                except Exception as write_error:
                    logging.warning(f"📝 ⚠️ Failed to write real-time data: {write_error}")
                
                # ✅ AWS BACKUP (if enabled)
                try:
                    if aws_storage.is_available():
                        # Save processed frame to AWS every 10 frames to avoid overwhelming
                        if frame_count % 10 == 0 and annotated is not None:
                            # Encode frame as JPEG
                            _, buffer = cv2.imencode('.jpg', annotated)
                            frame_data = buffer.tobytes()
                            
                            # Upload to S3
                            frame_id = f"{junction}_{timestamp.strftime('%Y%m%d_%H%M%S')}_frame_{frame_count}"
                            s3_url = aws_storage.upload_processed_frame(frame_data, frame_id)
                            
                            if s3_url:
                                logging.info(f"☁️ Frame {frame_count} backed up to AWS S3")
                            
                        # Save analytics data every 50 frames
                        if frame_count % 50 == 0:
                            analytics_data = {
                                'session_id': f"{junction}_{timestamp.strftime('%Y%m%d_%H%M')}",
                                'frame_range': f"{frame_count-49}-{frame_count}",
                                'analytics': update_data,
                                'detection_summary': {
                                    'total_vehicles': total_count,
                                    'class_distribution': class_counts,
                                    'traffic_light_state': light_state,
                                    'processing_timestamp': timestamp.isoformat()
                                }
                            }
                            
                            analytics_filename = f"analytics_{frame_id}.json"
                            s3_url = aws_storage.upload_analytics_data(analytics_data, analytics_filename)
                            
                            if s3_url:
                                logging.info(f"☁️ Analytics data backed up to AWS S3")
                                
                except Exception as aws_error:
                    logging.warning(f"☁️ ⚠️ AWS backup failed: {aws_error}")
                
                # ✅ OPTIONAL: Keep Socket.IO as backup (but polling is primary)
                if socketio:
                    try:
                        # Simple broadcast approach - much more reliable
                        logging.info(f"📡 Broadcasting 'update' to all clients")
                        socketio.emit('update', update_data)
                        
                        logging.info(f"📡 Broadcasting 'traffic_light' to all clients")
                        socketio.emit('traffic_light', traffic_light_data)
                        
                        logging.debug(f"📡 ✅ SocketIO events successfully broadcasted")
                    except Exception as emit_error:
                        logging.warning(f"📡 ⚠️ Failed to broadcast events (client may have disconnected): {emit_error}")
                else:
                    logging.warning(f"⚠️ No SocketIO instance available for frame {frame_count}")

            except Exception as frame_error:
                logging.error(f"Frame processing error: {frame_error}")
                continue  # Skip this frame and continue processing

            # Control processing speed with adaptive timing
            time.sleep(max(0.1, 1.0 / fps * 5))  # Adaptive timing based on video FPS

    except Exception as e:
        logging.error(f"Critical error in video processing: {e}")
        if socketio:
            socketio.emit('processing_error', {
                'message': f'Video processing error: {str(e)}'
            })
    
    finally:
        # Comprehensive cleanup
        global current_user_room
        processing_active = False
        
        # Clean up video capture
        if cap and cap.isOpened():
            cap.release()
            logging.info("Video capture released")
        
        # Clean up model memory
        if model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU memory cleared")
        
        # Save any remaining entries
        if entries_buffer and session:
            try:
                session.bulk_save_objects(entries_buffer)
                session.commit()
                logging.info(f"Saved final batch of {len(entries_buffer)} entries")
            except Exception as e:
                logging.error(f"Failed to save final batch: {e}")
                session.rollback()
        
        # Clean up global frame
        frame_for_preview = None
        
        # Emit processing_stopped event to the correct room
        if socketio:
            completion_data = {
                'message': f'Video processing completed for {junction}',
                'junction': junction,
                'timestamp': datetime.now().isoformat()
            }
            
            if current_user_room:
                logging.info(f"🛑 Video processor: Emitting processing_stopped to room {current_user_room}")
                socketio.emit('processing_stopped', completion_data, room=current_user_room)
            else:
                logging.info(f"🛑 Video processor: Broadcasting processing_stopped to all clients")
                socketio.emit('processing_stopped', completion_data)
            
            logging.info(f"🛑 Video processor: Successfully emitted processing_stopped")
        
        # Reset current user room
        current_user_room = None
        
        logging.info(f"Video processing cleanup completed for {junction}")

