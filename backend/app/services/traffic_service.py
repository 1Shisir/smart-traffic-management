"""Traffic data service for handling traffic-related operations."""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import desc, func
from app.models.traffic_data import TrafficData
from app import Session


class TrafficDataService:
    """Service class for traffic data operations."""
    
    @staticmethod
    def get_traffic_data(page: int = 1, per_page: int = 100, junction: Optional[str] = None) -> Dict[str, Any]:
        """
        Get paginated traffic data.
        
        Args:
            page: Page number (1-based)
            per_page: Number of records per page
            junction: Optional junction filter
            
        Returns:
            Dictionary containing data, pagination info, and metadata
        """
        session = Session()
        try:
            query = session.query(TrafficData)
            
            if junction:
                query = query.filter(TrafficData.junction == junction)
            
            # Get total count for pagination
            total = query.count()
            
            # Apply pagination and ordering
            data = (query.order_by(desc(TrafficData.timestamp))
                   .limit(per_page)
                   .offset((page - 1) * per_page)
                   .all())
            
            result = [{
                'id': d.id,
                'junction': d.junction,
                'total': d.total_count,
                'car': d.car_count,
                'bus': d.bus_count,
                'truck': d.truck_count,
                'motorcycle': d.motorcycle_count,
                'timestamp': d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'traffic_light': d.traffic_light,
                'light_duration': d.light_duration
            } for d in data]
            
            return {
                'data': result,
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
            
        except Exception as e:
            logging.error(f"Error fetching traffic data: {e}")
            raise
        finally:
            session.close()
    
    @staticmethod
    def get_traffic_summary(junction: Optional[str] = None) -> Dict[str, Any]:
        """Get traffic summary statistics."""
        session = Session()
        try:
            query = session.query(
                func.sum(TrafficData.total_count).label('total_vehicles'),
                func.sum(TrafficData.car_count).label('total_cars'),
                func.sum(TrafficData.bus_count).label('total_buses'),
                func.sum(TrafficData.truck_count).label('total_trucks'),
                func.sum(TrafficData.motorcycle_count).label('total_motorcycles'),
                func.count(TrafficData.id).label('total_records')
            )
            
            if junction:
                query = query.filter(TrafficData.junction == junction)
                
            result = query.first()
            
            return {
                'total_vehicles': result.total_vehicles or 0,
                'total_cars': result.total_cars or 0,
                'total_buses': result.total_buses or 0,
                'total_trucks': result.total_trucks or 0,
                'total_motorcycles': result.total_motorcycles or 0,
                'total_records': result.total_records or 0
            }
            
        except Exception as e:
            logging.error(f"Error fetching traffic summary: {e}")
            raise
        finally:
            session.close()
    
    @staticmethod
    def get_recent_traffic_data(limit: int = 10, junction: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get most recent traffic data entries."""
        session = Session()
        try:
            query = session.query(TrafficData)
            
            if junction:
                query = query.filter(TrafficData.junction == junction)
                
            data = (query.order_by(desc(TrafficData.timestamp))
                   .limit(limit)
                   .all())
            
            return [{
                'id': d.id,
                'junction': d.junction,
                'total': d.total_count,
                'timestamp': d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'traffic_light': d.traffic_light
            } for d in data]
            
        except Exception as e:
            logging.error(f"Error fetching recent traffic data: {e}")
            raise
        finally:
            session.close()
