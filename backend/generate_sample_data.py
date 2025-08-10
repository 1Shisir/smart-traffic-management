#!/usr/bin/env python3
"""
Sample Data Generator for Smart Traffic System
Generates realistic sample traffic data for testing and demonstration.
"""

import sys
import os
import random
from datetime import datetime, timedelta

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, Session
from app.models.traffic_data import TrafficData
from app.config import Config

def generate_sample_data(num_records=50):
    """Generate sample traffic data entries."""
    
    app = create_app()
    
    with app.app_context():
        session = Session()
        
        try:
            # Clear existing data
            session.query(TrafficData).delete()
            session.commit()
            print(f"🗑️ Cleared existing traffic data")
            
            # Generate sample data
            base_time = datetime.now() - timedelta(hours=2)
            
            for i in range(num_records):
                # Create realistic traffic patterns
                hour = (base_time + timedelta(minutes=i * 2)).hour
                
                # Rush hour simulation (7-9 AM, 5-7 PM)
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    # Higher traffic during rush hours
                    total_count = random.randint(15, 35)
                    car_ratio = 0.7
                    bus_ratio = 0.1
                    truck_ratio = 0.15
                    motorcycle_ratio = 0.05
                elif 22 <= hour or hour <= 6:
                    # Lower traffic at night
                    total_count = random.randint(2, 8)
                    car_ratio = 0.8
                    bus_ratio = 0.05
                    truck_ratio = 0.1
                    motorcycle_ratio = 0.05
                else:
                    # Normal traffic
                    total_count = random.randint(8, 20)
                    car_ratio = 0.75
                    bus_ratio = 0.08
                    truck_ratio = 0.12
                    motorcycle_ratio = 0.05
                
                # Calculate individual vehicle counts
                car_count = int(total_count * car_ratio)
                bus_count = int(total_count * bus_ratio)
                truck_count = int(total_count * truck_ratio)
                motorcycle_count = total_count - car_count - bus_count - truck_count
                
                # Ensure non-negative counts
                motorcycle_count = max(0, motorcycle_count)
                
                # Traffic light logic based on total count
                if total_count >= 25:
                    traffic_light = "red"
                    light_duration = 45
                elif total_count >= 15:
                    traffic_light = "red"
                    light_duration = 35
                elif total_count >= 8:
                    traffic_light = "yellow"
                    light_duration = 5
                else:
                    traffic_light = "green"
                    light_duration = 30
                
                # Create traffic data entry
                timestamp = base_time + timedelta(minutes=i * 2)
                
                traffic_entry = TrafficData(
                    junction="Main St & 1st Ave",
                    total_count=total_count,
                    car_count=car_count,
                    bus_count=bus_count,
                    truck_count=truck_count,
                    motorcycle_count=motorcycle_count,
                    traffic_light=traffic_light,
                    light_duration=light_duration,
                    timestamp=timestamp
                )
                
                session.add(traffic_entry)
                
                if (i + 1) % 10 == 0:
                    print(f"📊 Generated {i + 1}/{num_records} sample records...")
            
            session.commit()
            print(f"✅ Successfully generated {num_records} sample traffic data records")
            
            # Display some statistics
            total_records = session.query(TrafficData).count()
            latest_record = session.query(TrafficData).order_by(TrafficData.timestamp.desc()).first()
            
            print(f"📈 Database Statistics:")
            print(f"   Total Records: {total_records}")
            print(f"   Latest Record: {latest_record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Latest Count: {latest_record.total_count} vehicles")
            print(f"   Latest Light: {latest_record.traffic_light}")
            
        except Exception as e:
            print(f"❌ Error generating sample data: {e}")
            session.rollback()
        finally:
            session.close()

if __name__ == "__main__":
    print("🚦 Smart Traffic System - Sample Data Generator")
    print("=" * 50)
    
    # Check if we should generate data
    num_records = 50
    if len(sys.argv) > 1:
        try:
            num_records = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid number of records. Using default: 50")
    
    print(f"🎯 Generating {num_records} sample traffic records...")
    generate_sample_data(num_records)
    print("🎉 Sample data generation complete!")
