"""
Simple tests for TrafficDataService (Processing Service)
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from app.services.traffic_service import TrafficDataService


class TestTrafficDataService:
    """Test cases for TrafficDataService"""
    
    @patch('app.services.traffic_service.Session')
    def test_get_traffic_data_success(self, mock_session_class):
        """Test successful retrieval of traffic data"""
        # Setup mocks
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock traffic data
        mock_data = Mock()
        mock_data.id = 1
        mock_data.junction = "Main St & 1st Ave"
        mock_data.total_count = 25
        mock_data.car_count = 20
        mock_data.bus_count = 2
        mock_data.truck_count = 2
        mock_data.motorcycle_count = 1
        mock_data.timestamp = datetime(2025, 1, 1, 12, 0, 0)
        mock_data.traffic_light = "RED"
        
        # Setup query chain
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value.limit.return_value.offset.return_value.all.return_value = [mock_data]
        
        # Test
        result = TrafficDataService.get_traffic_data(page=1, per_page=10)
        
        # Assertions
        assert isinstance(result, dict)
        mock_session.close.assert_called_once()
    
    @patch('app.services.traffic_service.Session')
    def test_get_traffic_data_with_junction_filter(self, mock_session_class):
        """Test traffic data retrieval with junction filter"""
        # Setup mocks
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.order_by.return_value.limit.return_value.offset.return_value.all.return_value = []
        
        # Test
        result = TrafficDataService.get_traffic_data(junction="Test Junction")
        
        # Verify filter was called
        mock_query.filter.assert_called_once()
        mock_session.close.assert_called_once()
    
    @patch('app.services.traffic_service.Session')
    def test_get_traffic_data_empty_result(self, mock_session_class):
        """Test traffic data retrieval with no data"""
        # Setup mocks
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.order_by.return_value.limit.return_value.offset.return_value.all.return_value = []
        
        # Test
        result = TrafficDataService.get_traffic_data()
        
        # Assertions
        assert isinstance(result, dict)
        mock_session.close.assert_called_once()
