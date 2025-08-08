from datetime import timedelta
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gives: /.../backend/app

    # Adjust path to go one level up from app to backend, then add the filename
    VIDEO_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'traffic_sample1.mp4'))
    YOLO_MODEL = os.path.abspath(os.path.join(BASE_DIR, '..', 'yolov8n.pt'))  
    SQLALCHEMY_DATABASE_URI = 'sqlite:///traffic.db'
    SQLALCHEMY_ECHO = True
    JWT_SECRET_KEY = 'smart_traffic_secret_key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)

    TEST_USERS = [
        {'username': 'admin', 'password': 'admin123'},
        {'username': 'user', 'password': 'user123'}
    ]

    # Node-RED (Uncomment and set if needed)
    # NODE_RED_URL = 'http://localhost:1880/traffic'
    # NODE_RED_TIMEOUT = 5

    # JWT cookie settings
    JWT_TOKEN_LOCATION = ['cookies']
    JWT_ACCESS_COOKIE_PATH = '/'
    JWT_COOKIE_SECURE = False
    JWT_COOKIE_SAMESITE = 'Lax'
    JWT_SESSION_COOKIE = True
    JWT_COOKIE_CSRF_PROTECT = False
