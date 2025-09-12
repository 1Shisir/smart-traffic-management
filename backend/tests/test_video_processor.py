"""
Tests for video processing and vehicle detection functionality
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime

# Import the modules we want to test
from app.utils.video_processor import (
    detect_vehicles, 
    get_traffic_light_state,
    start_video_processing,
    stop_video_processing,
    is_processing_active,
    process_video
)
from app.routes.api import process_video_with_boundaries


class TestVehicleDetection:
    """Test vehicle detection functionality"""
    
    @patch('app.utils.video_processor.YOLO')
    def test_detect_vehicles_success(self, mock_yolo):
        """Test successful vehicle detection with multiple vehicle types"""
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock YOLO model and results
        mock_model = Mock()
        mock_result = Mock()
        mock_box1 = Mock()
        mock_box2 = Mock()
        mock_box3 = Mock()
        
        # Configure first detection (car)
        mock_box1.cls = [0]  # car class
        mock_box1.conf = [0.85]  # high confidence
        mock_box1.xyxy = [[100, 100, 200, 200]]  # bounding box
        
        # Configure second detection (bus)
        mock_box2.cls = [5]  # bus class
        mock_box2.conf = [0.75]  # high confidence
        mock_box2.xyxy = [[300, 150, 450, 300]]  # bounding box
        
        # Configure third detection (motorcycle)
        mock_box3.cls = [3]  # motorcycle class
        mock_box3.conf = [0.65]  # medium confidence
        mock_box3.xyxy = [[50, 50, 120, 150]]  # bounding box
        
        mock_result.boxes = [mock_box1, mock_box2, mock_box3]
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        # Execute the function
        total_count, class_counts, annotated_frame = detect_vehicles(frame, mock_model, vehicle_labels)
        
        # Assertions
        assert total_count == 3
        assert class_counts['car'] == 1
        assert class_counts['bus'] == 1
        assert class_counts['motorcycle'] == 1
        assert class_counts['truck'] == 0
        assert annotated_frame.shape == (480, 640, 3)
    
    def test_detect_vehicles_low_confidence(self):
        """Test that low confidence detections are filtered out"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock model with low confidence detection
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        
        mock_box.cls = [0]  # car class
        mock_box.conf = [0.3]  # low confidence (below 0.5 threshold)
        mock_box.xyxy = [[100, 100, 200, 200]]
        
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'car'}
        
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        total_count, class_counts, annotated_frame = detect_vehicles(frame, mock_model, vehicle_labels)
        
        # Should detect nothing due to low confidence
        assert total_count == 0
        assert class_counts['car'] == 0
    
    def test_detect_vehicles_error_handling(self):
        """Test error handling in vehicle detection"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock model that raises an exception
        mock_model = Mock()
        mock_model.side_effect = Exception("YOLO model error")
        
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        total_count, class_counts, annotated_frame = detect_vehicles(frame, mock_model, vehicle_labels)
        
        # Should return safe defaults on error
        assert total_count == 0
        assert class_counts == {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}
        assert annotated_frame.shape == (480, 640, 3)


class TestTrafficLightStates:
    """Test traffic light state determination"""
    
    def test_traffic_light_green_state(self):
        """Test green light state for low traffic"""
        state, duration = get_traffic_light_state(5)
        assert state == "green"
        assert duration == 30
    
    def test_traffic_light_yellow_state(self):
        """Test yellow light state for medium traffic"""
        state, duration = get_traffic_light_state(10)
        assert state == "yellow"
        assert duration == 5
    
    def test_traffic_light_red_state(self):
        """Test red light state for heavy traffic"""
        state, duration = get_traffic_light_state(15)
        assert state == "red"
        assert duration == 40
    
    def test_traffic_light_boundary_conditions(self):
        """Test boundary conditions for traffic light states"""
        # Test exact thresholds
        state, duration = get_traffic_light_state(7)  # Just below yellow threshold
        assert state == "green"
        
        state, duration = get_traffic_light_state(8)  # Exact yellow threshold
        assert state == "yellow"
        
        state, duration = get_traffic_light_state(12)  # Exact red threshold
        assert state == "red"



class TestPerformanceAndBoundaryConditions:
    """Test performance and edge cases"""
    
    def test_detect_vehicles_max_count_limit(self):
        """Test that vehicle counts are properly limited"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock model with many detections
        mock_model = Mock()
        mock_result = Mock()
        
        # Create 1500 mock detections (should be limited to 1000)
        mock_boxes = []
        for i in range(1500):
            mock_box = Mock()
            mock_box.cls = [0]  # car class
            mock_box.conf = [0.9]  # high confidence
            mock_box.xyxy = [[i % 640, i % 480, (i + 50) % 640, (i + 50) % 480]]
            mock_boxes.append(mock_box)
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'car'}
        
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        total_count, class_counts, annotated_frame = detect_vehicles(frame, mock_model, vehicle_labels)
        
        # Should be limited to max 1000
        assert total_count <= 1000
        assert class_counts['car'] <= 1000
    
    def test_detect_vehicles_empty_frame(self):
        """Test detection on empty/black frame"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = []  # No detections
        mock_model.return_value = [mock_result]
        mock_model.names = {}
        
        vehicle_labels = {'car', 'bus', 'truck', 'motorcycle'}
        
        total_count, class_counts, annotated_frame = detect_vehicles(frame, mock_model, vehicle_labels)
        
        assert total_count == 0
        assert all(count == 0 for count in class_counts.values())
    
    def test_traffic_light_extreme_values(self):
        """Test traffic light state with extreme vehicle counts"""
        # Test with 0 vehicles
        state, duration = get_traffic_light_state(0)
        assert state == "green"
        
        # Test with very high vehicle count
        state, duration = get_traffic_light_state(999)
        assert state == "red"
        assert duration == 40
        
        # Test with negative count (edge case)
        state, duration = get_traffic_light_state(-5)
        assert state == "green"  # Should default to green for invalid input


if __name__ == "__main__":
    pytest.main([__file__])
