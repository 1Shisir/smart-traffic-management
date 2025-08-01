class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///traffic.db'
    SQLALCHEMY_ECHO = True
    VIDEO_PATH = 'traffic_sample2.mp4'
    YOLO_MODEL = 'yolov8n.pt'
    # NODE_RED_URL = 'http://localhost:1880/traffic'
    # NODE_RED_TIMEOUT = 5