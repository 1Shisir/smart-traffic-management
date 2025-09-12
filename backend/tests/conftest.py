"""
Test configuration for pytest
"""
import pytest
import os
import sys
from unittest.mock import Mock

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def mock_session():
    """Mock database session"""
    return Mock()

@pytest.fixture
def sample_user():
    """Sample user data for testing"""
    user = Mock()
    user.username = "testuser"
    user.check_password = Mock(return_value=True)
    return user

@pytest.fixture
def sample_traffic_data():
    """Sample traffic data for testing"""
    data = Mock()
    data.id = 1
    data.junction = "Main St & 1st Ave"
    data.total_count = 25
    data.car_count = 20
    data.bus_count = 2
    data.truck_count = 2
    data.motorcycle_count = 1
    data.traffic_light = "RED"
    return data

@pytest.fixture
def mock_video_frame():
    """Mock video frame for testing"""
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing"""
    model = Mock()
    model.names = {0: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    return model

@pytest.fixture
def mock_socketio():
    """Mock SocketIO instance for testing"""
    socketio = Mock()
    socketio.emit = Mock()
    return socketio

@pytest.fixture
def sample_detection_results():
    """Sample YOLO detection results for testing"""
    results = []
    
    # Create mock detection boxes
    for i in range(3):
        mock_box = Mock()
        mock_box.cls = [0]  # car class
        mock_box.conf = [0.8 + i * 0.1]  # confidence scores
        mock_box.xyxy = [[100 + i*50, 100 + i*30, 200 + i*50, 200 + i*30]]
        results.append(mock_box)
    
    return results

@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials for testing"""
    return {
        'aws_access_key_id': 'test_access_key',
        'aws_secret_access_key': 'test_secret_key',
        'region_name': 'us-east-1',
        'bucket_name': 'test-traffic-bucket'
    }
