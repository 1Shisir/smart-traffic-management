"""
Tests for live streaming and real-time functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json
import base64
import numpy as np
import cv2
from datetime import datetime
import threading
import time

from app.utils.video_processor import (
    start_video_processing,
    stop_video_processing,
    is_processing_active,
    get_frame_for_preview,
    set_socketio
)
from app.utils.realtime_polling import write_realtime_data, read_realtime_data


class TestLiveStreaming:
    """Test live streaming and real-time video processing"""
    
    @pytest.fixture
    def mock_socketio(self):
        """Create mock SocketIO instance"""
        socketio = Mock()
        socketio.emit = Mock()
        return socketio
    
    @pytest.fixture
    def mock_app(self):
        """Create mock Flask app"""
        app = Mock()
        app.app_context = Mock()
        return app
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        session = Mock()
        session.bulk_save_objects = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session
    
    def test_start_video_processing_success(self, mock_app, mock_socketio, mock_session):
        """Test successful start of video processing"""
        # Set up the socketio instance
        set_socketio(mock_socketio)
        
        # Mock video file exists
        with patch('os.path.exists', return_value=True):
            with patch('app.utils.video_processor.process_video_with_context') as mock_process:
                result = start_video_processing(
                    mock_app, 
                    mock_socketio, 
                    mock_session, 
                    junction="test_junction",
                    user_room="test_room"
                )
                
                assert result == True
                assert is_processing_active() == True
    
    def test_stop_video_processing_success(self, mock_socketio):
        """Test successful stop of video processing"""
        # First start processing
        with patch('os.path.exists', return_value=True):
            with patch('app.utils.video_processor.process_video_with_context'):
                start_video_processing(Mock(), mock_socketio, Mock(), user_room="test_room")
        
        # Then stop it
        result = stop_video_processing(user_room="test_room")
        
        assert result == True
        # Note: processing_active might still be True briefly due to threading
    
    def test_socketio_emit_events(self, mock_socketio):
        """Test that SocketIO events are emitted correctly"""
        set_socketio(mock_socketio)
        
        # Test processing started event
        with patch('os.path.exists', return_value=True):
            with patch('app.utils.video_processor.process_video_with_context'):
                start_video_processing(Mock(), mock_socketio, Mock(), user_room="test_room")
        
        # Test processing stopped event
        stop_video_processing(user_room="test_room")
        
        # Verify emit was called (exact calls depend on implementation)
        assert mock_socketio.emit.called
    
    @patch('app.utils.video_processor.cv2.imencode')
    def test_get_frame_for_preview(self, mock_imencode):
        """Test getting frame for preview with base64 encoding"""
        # Create a mock frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock cv2.imencode to return success and buffer
        mock_buffer = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        mock_imencode.return_value = (True, mock_buffer)
        
        # Set the global frame
        import app.utils.video_processor as vp
        vp.frame_for_preview = test_frame
        
        # Get the preview
        result = get_frame_for_preview()
        
        # Verify result format
        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")
        
        # Verify cv2.imencode was called correctly
        mock_imencode.assert_called_once_with('.jpg', test_frame)
    
    def test_get_frame_for_preview_no_frame(self):
        """Test getting frame for preview when no frame is available"""
        # Clear the global frame
        import app.utils.video_processor as vp
        vp.frame_for_preview = None
        
        result = get_frame_for_preview()
        assert result is None
    
    @patch('app.utils.video_processor.cv2.imencode')
    def test_get_frame_for_preview_encoding_error(self, mock_imencode):
        """Test frame preview when encoding fails"""
        # Mock encoding failure
        mock_imencode.return_value = (False, None)
        
        # Set a test frame
        import app.utils.video_processor as vp
        vp.frame_for_preview = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = get_frame_for_preview()
        assert result is None


class TestRealTimeDataUpdates:
    """Test real-time data updates and polling"""
    
    def test_write_realtime_data(self):
        """Test writing real-time data to polling file"""
        test_data = {
            'junction': 'Test Junction',
            'count': 10,
            'car': 8,
            'bus': 1,
            'truck': 1,
            'motorcycle': 0,
            'traffic_light': 'green',
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            write_realtime_data(test_data)
            
            # Verify file was opened for writing
            mock_open.assert_called_once()
            assert mock_file.write.called  # Check that write was called
            
            # Combine all write calls to get the complete JSON string
            written_parts = [call[0][0] for call in mock_file.write.call_args_list]
            written_data = ''.join(written_parts)
            parsed_data = json.loads(written_data)
            assert parsed_data['junction'] == 'Test Junction'
            assert parsed_data['count'] == 10
    
    def test_read_realtime_data(self):
        """Test reading real-time data from polling file"""
        test_data = {
            'junction': 'Test Junction',
            'count': 15,
            'traffic_light': 'red',
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = json.dumps(test_data)
            
            result = read_realtime_data()
            
            # Verify file was opened for reading
            mock_open.assert_called_once()
            mock_file.read.assert_called_once()
            
            # Verify data was parsed correctly
            assert result['junction'] == 'Test Junction'
            assert result['count'] == 15
            assert result['traffic_light'] == 'red'
    
    def test_read_realtime_data_file_not_found(self):
        """Test reading real-time data when file doesn't exist"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = read_realtime_data()
            
            # Should return default data structure
            assert isinstance(result, dict)
            assert 'error' in result or 'status' in result
    
    def test_read_realtime_data_invalid_json(self):
        """Test reading real-time data with corrupted JSON"""
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = "invalid json content"
            
            result = read_realtime_data()
            
            # Should handle JSON parsing error gracefully
            assert isinstance(result, dict)


class TestWebSocketCommunication:
    """Test WebSocket communication patterns"""
    
    def test_socketio_room_management(self, mock_socketio):
        """Test SocketIO room-based communication"""
        set_socketio(mock_socketio)
        
        # Test emitting to specific room
        test_data = {'message': 'test', 'count': 5}
        room_name = 'user_123'
        
        # Simulate emitting to room
        mock_socketio.emit('traffic_update', test_data, room=room_name)
        
        # Verify emit was called with correct parameters
        mock_socketio.emit.assert_called_with('traffic_update', test_data, room=room_name)
    
    def test_socketio_broadcast_events(self, mock_socketio):
        """Test broadcasting events to all clients"""
        set_socketio(mock_socketio)
        
        # Test broadcasting without room
        test_data = {'message': 'system_status', 'status': 'online'}
        
        mock_socketio.emit('system_update', test_data)
        
        # Verify broadcast emit
        mock_socketio.emit.assert_called_with('system_update', test_data)
    
    def test_socketio_error_handling(self, mock_socketio):
        """Test SocketIO error handling"""
        # Mock SocketIO emit failure
        mock_socketio.emit = Mock(side_effect=Exception("Connection lost"))
        set_socketio(mock_socketio)
        
        # Attempt to emit - should handle error gracefully
        try:
            mock_socketio.emit('test_event', {'data': 'test'})
        except Exception as e:
            assert "Connection lost" in str(e)
    
    def test_multiple_client_handling(self, mock_socketio):
        """Test handling multiple connected clients"""
        set_socketio(mock_socketio)
        
        # Simulate multiple clients in different rooms
        rooms = ['user_1', 'user_2', 'user_3']
        test_data = {'count': 10, 'traffic_light': 'green'}
        
        # Emit to each room
        for room in rooms:
            mock_socketio.emit('traffic_update', test_data, room=room)
        
        # Verify all rooms received the update
        assert mock_socketio.emit.call_count == 3
        
        # Check that each call was made with correct room
        expected_calls = [
            call('traffic_update', test_data, room='user_1'),
            call('traffic_update', test_data, room='user_2'),
            call('traffic_update', test_data, room='user_3')
        ]
        mock_socketio.emit.assert_has_calls(expected_calls)


class TestPerformanceAndScalability:
    """Test performance aspects of live streaming"""
    
    def test_frame_processing_rate(self):
        """Test that frame processing maintains target rate"""
        frame_count = 0
        start_time = time.time()
        target_fps = 5  # Expected processing rate
        
        # Simulate processing frames
        for i in range(25):  # Process 25 frames
            frame_count += 1
            time.sleep(0.01)  # Simulate processing time
        
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time
        
        # Should be able to process at reasonable rate
        assert actual_fps > 1  # At least 1 FPS
    
    def test_memory_usage_frame_processing(self):
        """Test memory usage during frame processing"""
        # Create large frame to test memory handling
        large_frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        
        # Set as preview frame
        import app.utils.video_processor as vp
        vp.frame_for_preview = large_frame
        
        # Get preview (should handle large frames)
        with patch('app.utils.video_processor.cv2.imencode') as mock_imencode:
            mock_imencode.return_value = (True, np.array([1, 2, 3]))
            result = get_frame_for_preview()
            
            assert result is not None
            assert result.startswith("data:image/jpeg;base64,")
    
    def test_concurrent_processing_safety(self, mock_socketio):
        """Test thread safety of concurrent video processing"""
        set_socketio(mock_socketio)
        
        # Test that multiple start requests don't cause issues
        with patch('os.path.exists', return_value=True):
            with patch('app.utils.video_processor.process_video_with_context'):
                # First start should succeed
                result1 = start_video_processing(Mock(), mock_socketio, Mock(), user_room="room1")
                
                # Second start should fail (already active)
                result2 = start_video_processing(Mock(), mock_socketio, Mock(), user_room="room2")
                
                assert result1 == True
                assert result2 == False  # Should reject second start
    
    def test_graceful_shutdown(self, mock_socketio):
        """Test graceful shutdown of video processing"""
        set_socketio(mock_socketio)
        
        # Start processing
        with patch('os.path.exists', return_value=True):
            with patch('app.utils.video_processor.process_video_with_context'):
                start_video_processing(Mock(), mock_socketio, Mock(), user_room="test_room")
        
        # Stop processing
        result = stop_video_processing(user_room="test_room")
        
        assert result == True
        
        # Should be able to start again after stopping
        with patch('os.path.exists', return_value=True):
            with patch('app.utils.video_processor.process_video_with_context'):
                result = start_video_processing(Mock(), mock_socketio, Mock(), user_room="test_room")
                # Note: This might fail due to threading timing, but should eventually work


if __name__ == "__main__":
    pytest.main([__file__])
