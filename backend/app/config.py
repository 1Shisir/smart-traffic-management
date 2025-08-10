import os
from datetime import timedelta
from typing import List, Dict, Any


class Config:
    """Application configuration class with environment variable support."""
    
    # Base directory paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/app
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))  # root project directory
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'SQLALCHEMY_DATABASE_URI',
        f'sqlite:///{os.path.join(BASE_DIR, "..", "traffic.db")}'
    )
    SQLALCHEMY_ECHO = os.getenv('SQLALCHEMY_ECHO', 'True').lower() == 'true'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', os.urandom(32).hex())
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        days=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES_DAYS', 1))  # Reduced from 30 to 1 day
    )
    JWT_TOKEN_LOCATION = ['cookies']
    JWT_ACCESS_COOKIE_PATH = '/'
    JWT_COOKIE_SECURE = os.getenv('JWT_COOKIE_SECURE', 'False').lower() == 'true'
    JWT_COOKIE_SAMESITE = os.getenv('JWT_COOKIE_SAMESITE', 'Lax')
    JWT_SESSION_COOKIE = True
    JWT_COOKIE_CSRF_PROTECT = False
    
    # File Paths
    VIDEO_PATH = os.getenv(
        'VIDEO_PATH',
        os.path.abspath(os.path.join(BASE_DIR, '..', 'traffic_sample1.mp4'))
    )
    YOLO_MODEL = os.getenv(
        'YOLO_MODEL',
        os.path.abspath(os.path.join(BASE_DIR, '..', 'yolov8n.pt'))
    )
    
    # Video Processing Configuration
    DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', 0.5))
    TRAFFIC_LIGHT_THRESHOLD = int(os.getenv('TRAFFIC_LIGHT_THRESHOLD', 12))
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', 5))  # Process every Nth frame
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Node-RED Integration (Optional)
    NODE_RED_URL = os.getenv('NODE_RED_URL')
    NODE_RED_TIMEOUT = int(os.getenv('NODE_RED_TIMEOUT', 5))
    
    # Test Users Configuration
    TEST_USERS: List[Dict[str, str]] = [
        {
            'username': os.getenv('ADMIN_USERNAME', 'admin'),
            'password': os.getenv('ADMIN_PASSWORD', 'admin123')
        },
        {
            'username': os.getenv('USER_USERNAME', 'user'),
            'password': os.getenv('USER_PASSWORD', 'user123')
        }
    ]
    
    # Junction Configuration (if using multiple junctions)
    JUNCTIONS: Dict[str, str] = {
        'junction1': VIDEO_PATH,
        'main_junction': VIDEO_PATH,  # Added main_junction alias
        # Add more junctions as needed
        # 'junction2': os.getenv('JUNCTION2_VIDEO_PATH', ''),
    }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required files
        if not os.path.exists(cls.VIDEO_PATH):
            issues.append(f"Video file not found: {cls.VIDEO_PATH}")
        
        if not os.path.exists(cls.YOLO_MODEL):
            issues.append(f"YOLO model file not found: {cls.YOLO_MODEL}")
        
        return issues


class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    JWT_COOKIE_SECURE = False


# Configuration mapping - simplified for local development
config_map = {
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}
