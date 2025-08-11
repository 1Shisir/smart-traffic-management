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

# Global variables for process control
processing_active = False
processing_thread = None
stop_event = threading.Event()
frame_for_preview = None

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

def start_video_processing(app, socketio_param, session, junction="main_junction", video_path=None):
    """Start video processing in a separate thread."""
    global processing_active, processing_thread, stop_event, socketio
    
    if processing_active:
        logging.warning("Video processing is already active")
        return False
    
    stop_event.clear()
    processing_active = True
    socketio = socketio_param  # Assign the socketio parameter to global variable
    
    # Start processing in a separate thread
    processing_thread = threading.Thread(
        target=process_video,
        args=(session, junction, video_path),
        daemon=True
    )
    processing_thread.start()
    
    logging.info(f"Started video processing thread for {junction}")
    return True

def stop_video_processing():
    """Stop video processing."""
    global processing_active, stop_event
    
    if not processing_active:
        logging.warning("Video processing is not active")
        return False
    
    stop_event.set()
    processing_active = False
    logging.info("Video processing stop requested")
    return True

def is_processing_active():
    """Check if video processing is currently active."""
    return processing_active

def process_video(session, junction="main_junction", video_path=None):
    """Process video and emit real-time updates via WebSocket with proper resource management."""
    global frame_for_preview, processing_active, socketio
    
    # Debug: Check if socketio is available
    logging.info(f"🔧 Starting video processing with SocketIO: {socketio is not None}")
    
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

                # Real-time updates via SocketIO to all connected clients
                if socketio:
                    logging.info(f"📡 Emitting SocketIO update for frame {frame_count}")
                    socketio.emit('update', {
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
                    })

                    socketio.emit('traffic_light', {
                        'state': light_state,
                        'duration': light_duration
                    })
                    logging.info(f"📡 SocketIO events emitted successfully")
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
        
        if socketio:
            socketio.emit('processing_stopped', {
                'message': f'Video processing stopped for {junction}',
                'junction': junction
            })
        
        logging.info(f"Video processing cleanup completed for {junction}")

