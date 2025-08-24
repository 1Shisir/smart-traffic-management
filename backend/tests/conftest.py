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
