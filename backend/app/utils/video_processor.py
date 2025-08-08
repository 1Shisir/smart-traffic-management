import cv2
import requests
import os
import logging
import time
from datetime import datetime
import torch
from ultralytics import YOLO
from app.models.traffic_data import TrafficData
from app.config import Config

def detect_vehicles(frame, model, vehicle_labels):
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
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{name} ({int(conf*100)}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_count = sum(class_counts.values())
        return total_count, class_counts, frame
    except Exception as e:
        logging.error(f"YOLO detection error: {e}")
        return 0, {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}, frame

def get_traffic_light_state(total_count, threshold=12):
    if total_count > threshold:
        return "red", 30
    elif total_count > threshold / 2:
        return "yellow", 10
    else:
        return "green", 20

def process_video(app, socketio, session):
    global frame_for_preview
    entries_buffer = []
    try:
        if not os.path.exists(Config.VIDEO_PATH):
            logging.error(f"Video file not found: {Config.VIDEO_PATH}")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(Config.YOLO_MODEL).to(device)
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}

        while True:  # Main loop to continuously process the video
            cap = cv2.VideoCapture(Config.VIDEO_PATH)
            if not cap.isOpened():
                logging.error("Error opening video file")
                time.sleep(5)  # Wait before retrying
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logging.info(f"Processing video: {Config.VIDEO_PATH} ({total_frames} frames, {fps} fps)")

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video, will restart from beginning

                frame_count += 1
                if frame_count % 10 != 0:  # Process every 10th frame
                    continue

                # Vehicle detection
                total_count, class_counts, annotated = detect_vehicles(frame, model, vehicle_labels)
                frame_for_preview = annotated.copy()
                timestamp = datetime.now()
                light_state, light_duration = get_traffic_light_state(total_count)
                
                logging.info(f"Frame {frame_count}: Vehicles -> Total: {total_count}, Traffic Light: {light_state}")

                try:
                    # Buffer data for database
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

                    # Commit to database in batches
                    if len(entries_buffer) >= 10:
                        session.bulk_save_objects(entries_buffer)
                        session.commit()
                        entries_buffer = []

                    # Real-time updates via SocketIO
                    socketio.emit('update', {
                        'junction': "main_junction",
                        'count': total_count,
                        'car': class_counts['car'],
                        'bus': class_counts['bus'],
                        'truck': class_counts['truck'],
                        'motorcycle': class_counts['motorcycle'],
                        'time': timestamp.strftime("%H:%M:%S"),
                        'frame_count': frame_count,
                        'total_frames': total_frames
                    }, namespace='/dashboard')

                    socketio.emit('traffic_light', {
                        'state': light_state,
                        'duration': light_duration
                    }, namespace='/dashboard')

                    # Optional: Send frame preview for visualization
                    if frame_for_preview is not None:
                        ret, buffer = cv2.imencode('.jpg', frame_for_preview)
                        if ret:
                            socketio.emit('frame_update', {
                                'image': buffer.tobytes()
                            }, namespace='/dashboard')

                except Exception as e:
                    logging.error(f"Database/SocketIO error: {e}")
                    session.rollback()

                # Adjust sleep time based on video FPS for real-time simulation
                time.sleep(1/fps if fps > 0 else 0.1)

            # End of video - clean up and prepare for next loop
            cap.release()
            logging.info("Restarting video processing...")
            time.sleep(1)  # Brief pause before restarting

    except Exception as e:
        logging.error(f"Error in video processing: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        time.sleep(5)  # Wait before attempting to restart