from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TrafficData(Base):
    __tablename__ = 'traffic_data'
    id = Column(Integer, primary_key=True)
    junction = Column(String(50))
    total_count = Column(Integer)
    car_count = Column(Integer)
    bus_count = Column(Integer)
    truck_count = Column(Integer)
    motorcycle_count = Column(Integer)
    traffic_light = Column(String(10))
    light_duration = Column(Integer)
    timestamp = Column(DateTime)