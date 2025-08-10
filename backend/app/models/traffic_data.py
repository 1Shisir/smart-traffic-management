from sqlalchemy import Column, Integer, String, DateTime, CheckConstraint
from sqlalchemy.orm import declarative_base, validates
from datetime import datetime

Base = declarative_base()

class TrafficData(Base):
    __tablename__ = 'traffic_data'
    
    id = Column(Integer, primary_key=True)
    junction = Column(String(50), nullable=False)
    total_count = Column(Integer, nullable=False, default=0)
    car_count = Column(Integer, nullable=False, default=0)
    bus_count = Column(Integer, nullable=False, default=0)
    truck_count = Column(Integer, nullable=False, default=0)
    motorcycle_count = Column(Integer, nullable=False, default=0)
    traffic_light = Column(String(10), nullable=False, default='green')
    light_duration = Column(Integer, nullable=False, default=30)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    
    # Add database constraints for data integrity
    __table_args__ = (
        CheckConstraint('total_count >= 0', name='positive_total_count'),
        CheckConstraint('car_count >= 0', name='positive_car_count'),
        CheckConstraint('bus_count >= 0', name='positive_bus_count'),
        CheckConstraint('truck_count >= 0', name='positive_truck_count'),
        CheckConstraint('motorcycle_count >= 0', name='positive_motorcycle_count'),
        CheckConstraint('light_duration > 0', name='positive_light_duration'),
        CheckConstraint("traffic_light IN ('red', 'yellow', 'green')", name='valid_traffic_light'),
    )
    
    @validates('junction')
    def validate_junction(self, key, value):
        """Validate junction name."""
        if not value or len(value) > 50:
            raise ValueError("Junction name must be 1-50 characters")
        return value.strip()
    
    @validates('total_count', 'car_count', 'bus_count', 'truck_count', 'motorcycle_count')
    def validate_counts(self, key, value):
        """Validate vehicle counts are reasonable."""
        if value < 0 or value > 1000:
            raise ValueError(f"{key} must be between 0 and 1000")
        return value
    
    @validates('traffic_light')
    def validate_traffic_light(self, key, value):
        """Validate traffic light state."""
        valid_states = ['red', 'yellow', 'green']
        if value not in valid_states:
            raise ValueError(f"Traffic light must be one of: {valid_states}")
        return value
    
    @validates('light_duration')
    def validate_light_duration(self, key, value):
        """Validate light duration is reasonable."""
        if value <= 0 or value > 300:  # Max 5 minutes
            raise ValueError("Light duration must be between 1 and 300 seconds")
        return value