"""
Simple polling-based real-time updates API
This approach is much more reliable than Socket.IO for real-time data
"""

from flask import jsonify
from datetime import datetime
import json
import os
from app.config import Config

# Global state file for real-time data
REALTIME_DATA_FILE = os.path.join(Config.BASE_DIR, 'realtime_data.json')

def write_realtime_data(data):
    """Write real-time data to file for polling"""
    try:
        # Always add current timestamp
        data['timestamp'] = datetime.now().isoformat()
        data['last_update'] = datetime.now().isoformat()
        
        with open(REALTIME_DATA_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error writing realtime data: {e}")

def clear_stale_data():
    """Clear stale data on app startup"""
    try:
        if os.path.exists(REALTIME_DATA_FILE):
            # Write fresh default data
            write_realtime_data(get_default_data())
            print("Cleared stale realtime data on startup")
    except Exception as e:
        print(f"Error clearing stale data: {e}")

def read_realtime_data():
    """Read real-time data from file"""
    try:
        if os.path.exists(REALTIME_DATA_FILE):
            with open(REALTIME_DATA_FILE, 'r') as f:
                data = json.load(f)
                
                # Check if data is stale (older than 10 seconds)
                if 'timestamp' in data:
                    try:
                        last_update = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                        now = datetime.now()
                        if (now - last_update).total_seconds() > 10:
                            print(f"Realtime data is stale ({(now - last_update).total_seconds():.1f}s old), returning default")
                            return get_default_data()
                    except Exception as e:
                        print(f"Error parsing timestamp: {e}")
                        return get_default_data()
                
                return data
    except Exception as e:
        print(f"Error reading realtime data: {e}")
    
    # Return default data if file doesn't exist or is corrupted
    return get_default_data()

def get_default_data():
    """Get default realtime data structure"""
def get_default_data():
    """Get default realtime data structure"""
    return {
        'junction': 'Main St & 1st Ave',
        'total_vehicles': 0,
        'car_count': 0,
        'bus_count': 0,
        'truck_count': 0,
        'motorcycle_count': 0,
        'traffic_light_state': 'red',
        'traffic_light_duration': 30,
        'timestamp': datetime.now().isoformat(),
        'processing_active': False,
        'status': 'inactive',
        'last_update': datetime.now().isoformat()
    }

# Add this endpoint to your API routes
def add_polling_endpoints(api):
    """Add polling endpoints to the API blueprint"""
    
    @api.route('/realtime-status')
    def get_realtime_status():
        """Get current real-time status for polling"""
        try:
            data = read_realtime_data()
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
