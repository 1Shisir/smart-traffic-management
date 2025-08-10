"""
Health check and monitoring utilities for the Smart Traffic System.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, List
from flask import Blueprint, jsonify
from app.config import Config
from app import Session
from app.models.traffic_data import TrafficData

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

health = Blueprint('health', __name__)
logger = logging.getLogger(__name__)


@health.route('/health')
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })


@health.route('/health/detailed')
def detailed_health_check():
    """Detailed health check with system information."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Database check
        health_status['checks']['database'] = _check_database()
        
        # File system check
        health_status['checks']['filesystem'] = _check_filesystem()
        
        # System resources check
        health_status['checks']['system'] = _check_system_resources()
        
        # Determine overall status
        failed_checks = [k for k, v in health_status['checks'].items() if not v.get('healthy', False)]
        if failed_checks:
            health_status['status'] = 'unhealthy'
            health_status['failed_checks'] = failed_checks
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500


def _check_database() -> Dict[str, Any]:
    """Check database connectivity and basic operations."""
    try:
        session = Session()
        
        # Test basic query
        start_time = time.time()
        count = session.query(TrafficData).count()
        query_time = time.time() - start_time
        
        session.close()
        
        return {
            'healthy': True,
            'total_records': count,
            'query_time_ms': round(query_time * 1000, 2)
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e)
        }


def _check_filesystem() -> Dict[str, Any]:
    """Check required files and disk space."""
    try:
        checks = {}
        
        # Check required files
        required_files = {
            'video_file': Config.VIDEO_PATH,
            'yolo_model': Config.YOLO_MODEL
        }
        
        for name, path in required_files.items():
            checks[name] = {
                'exists': os.path.exists(path),
                'path': path
            }
            if checks[name]['exists']:
                stat = os.stat(path)
                checks[name]['size_mb'] = round(stat.st_size / (1024 * 1024), 2)
                checks[name]['modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Check disk space (only if psutil is available)
        if PSUTIL_AVAILABLE:
            disk_usage = psutil.disk_usage('/')
            checks['disk_space'] = {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'used_gb': round(disk_usage.used / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'percent_used': round((disk_usage.used / disk_usage.total) * 100, 2)
            }
            sufficient_space = checks['disk_space']['percent_used'] < 90
        else:
            checks['disk_space'] = {'status': 'psutil not available'}
            sufficient_space = True
        
        # Determine if filesystem is healthy
        all_files_exist = all(check['exists'] for check in checks.values() if 'exists' in check)
        
        return {
            'healthy': all_files_exist and sufficient_space,
            'checks': checks
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e)
        }


def _check_system_resources() -> Dict[str, Any]:
    """Check system resource usage."""
    if not PSUTIL_AVAILABLE:
        return {
            'healthy': True,
            'status': 'psutil not available - system monitoring disabled'
        }
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Process information
        process = psutil.Process()
        process_info = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': round(process.memory_info().rss / (1024 * 1024), 2),
            'num_threads': process.num_threads(),
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
        }
        
        system_healthy = (
            cpu_percent < 80 and 
            memory.percent < 85 and 
            process_info['memory_mb'] < 1024  # Process using less than 1GB
        )
        
        return {
            'healthy': system_healthy,
            'cpu_percent': cpu_percent,
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': memory.percent
            },
            'process': process_info
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e)
        }
