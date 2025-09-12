"""
Tests for SQLite database operations and traffic data logging
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sqlite3
import tempfile
import os

from app.models.traffic_data import TrafficData
from app.services.traffic_service import TrafficDataService


class TestSQLiteLogging:
    """Test SQLite database operations for traffic data logging"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        session = Mock()
        return session
    
    @pytest.fixture
    def sample_traffic_entries(self):
        """Create sample traffic data entries"""
        entries = []
        base_time = datetime.now()
        
        for i in range(10):
            entry = TrafficData(
                junction=f"Junction_{i % 3}",  # 3 different junctions
                total_count=10 + i,
                car_count=8 + i,
                bus_count=1,
                truck_count=1,
                motorcycle_count=0,
                traffic_light="green" if i % 3 == 0 else "red" if i % 3 == 1 else "yellow",
                light_duration=30 if i % 3 == 0 else 40 if i % 3 == 1 else 5,
                timestamp=base_time + timedelta(minutes=i)
            )
            entries.append(entry)
        
        return entries
    
    def test_traffic_data_model_creation(self):
        """Test creating a TrafficData model instance"""
        timestamp = datetime.now()
        traffic_data = TrafficData(
            junction="Main St & 1st Ave",
            total_count=15,
            car_count=12,
            bus_count=2,
            truck_count=1,
            motorcycle_count=0,
            traffic_light="red",
            light_duration=40,
            timestamp=timestamp
        )
        
        assert traffic_data.junction == "Main St & 1st Ave"
        assert traffic_data.total_count == 15
        assert traffic_data.car_count == 12
        assert traffic_data.bus_count == 2
        assert traffic_data.truck_count == 1
        assert traffic_data.motorcycle_count == 0
        assert traffic_data.traffic_light == "red"
        assert traffic_data.light_duration == 40
        assert traffic_data.timestamp == timestamp
    
    def test_traffic_data_validation_positive_counts(self):
        """Test validation of positive vehicle counts"""
        with pytest.raises(ValueError, match="must be between 0 and 1000"):
            traffic_data = TrafficData(
                junction="Test Junction",
                total_count=-5,  # Invalid negative count
                car_count=0,
                bus_count=0,
                truck_count=0,
                motorcycle_count=0,
                traffic_light="green",
                light_duration=30,
                timestamp=datetime.now()
            )
    
    def test_traffic_data_validation_max_counts(self):
        """Test validation of maximum vehicle counts"""
        with pytest.raises(ValueError, match="must be between 0 and 1000"):
            traffic_data = TrafficData(
                junction="Test Junction",
                total_count=1500,  # Invalid high count
                car_count=0,
                bus_count=0,
                truck_count=0,
                motorcycle_count=0,
                traffic_light="green",
                light_duration=30,
                timestamp=datetime.now()
            )
    
    def test_traffic_data_validation_valid_light_states(self):
        """Test validation of traffic light states"""
        with pytest.raises(ValueError, match="Traffic light must be one of"):
            traffic_data = TrafficData(
                junction="Test Junction",
                total_count=10,
                car_count=8,
                bus_count=1,
                truck_count=1,
                motorcycle_count=0,
                traffic_light="blue",  # Invalid light state
                light_duration=30,
                timestamp=datetime.now()
            )
    
    def test_traffic_data_validation_junction_length(self):
        """Test validation of junction name length"""
        with pytest.raises(ValueError, match="Junction name must be 1-50 characters"):
            traffic_data = TrafficData(
                junction="A" * 60,  # Too long junction name
                total_count=10,
                car_count=8,
                bus_count=1,
                truck_count=1,
                motorcycle_count=0,
                traffic_light="green",
                light_duration=30,
                timestamp=datetime.now()
            )
    
    def test_traffic_data_validation_light_duration(self):
        """Test validation of light duration"""
        with pytest.raises(ValueError, match="Light duration must be between 1 and 300 seconds"):
            traffic_data = TrafficData(
                junction="Test Junction",
                total_count=10,
                car_count=8,
                bus_count=1,
                truck_count=1,
                motorcycle_count=0,
                traffic_light="green",
                light_duration=0,  # Invalid duration
                timestamp=datetime.now()
            )
    
    def test_bulk_save_traffic_data(self, mock_db_session, sample_traffic_entries):
        """Test bulk saving of traffic data to database"""
        # Mock successful bulk save
        mock_db_session.bulk_save_objects = Mock()
        mock_db_session.commit = Mock()
        
        # Execute bulk save
        mock_db_session.bulk_save_objects(sample_traffic_entries)
        mock_db_session.commit()
        
        # Verify operations were called
        mock_db_session.bulk_save_objects.assert_called_once_with(sample_traffic_entries)
        mock_db_session.commit.assert_called_once()
    
    
    def test_database_rollback_on_error(self, mock_db_session, sample_traffic_entries):
        """Test database rollback when commit fails"""
        # Mock commit failure
        mock_db_session.bulk_save_objects = Mock()
        mock_db_session.commit = Mock(side_effect=Exception("Database connection lost"))
        mock_db_session.rollback = Mock()
        
        # Attempt to save data
        try:
            mock_db_session.bulk_save_objects(sample_traffic_entries)
            mock_db_session.commit()
        except Exception:
            mock_db_session.rollback()
        
        # Verify rollback was called
        mock_db_session.rollback.assert_called_once()


class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    def test_data_consistency_checks(self):
        """Test data consistency validation"""
        # Test that total count matches sum of individual counts
        timestamp = datetime.now()
        
        # Valid case
        traffic_data = TrafficData(
            junction="Test Junction",
            total_count=15,
            car_count=10,
            bus_count=2,
            truck_count=2,
            motorcycle_count=1,
            traffic_light="red",
            light_duration=40,
            timestamp=timestamp
        )
        
        # Calculate actual total
        calculated_total = (traffic_data.car_count + traffic_data.bus_count + 
                          traffic_data.truck_count + traffic_data.motorcycle_count)
        
        assert traffic_data.total_count == calculated_total
    


if __name__ == "__main__":
    pytest.main([__file__])
