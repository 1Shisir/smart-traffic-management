"""
AWS Storage API endpoints for Smart Traffic Management System
"""

import os
import logging
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile

from ..services.aws_service import aws_storage
from ..config import Config

logger = logging.getLogger(__name__)

# Create AWS storage blueprint
aws_bp = Blueprint('aws', __name__, url_prefix='/api/aws')

@aws_bp.route('/status', methods=['GET'])
def get_aws_status():
    """Get AWS service status and statistics"""
    try:
        stats = aws_storage.get_storage_stats()
        return jsonify({
            'success': True,
            'aws_available': stats.get('available', False),
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting AWS status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@aws_bp.route('/upload/video', methods=['POST'])
def upload_video():
    """Upload video file to AWS S3"""
    try:
        if not aws_storage.is_available():
            return jsonify({
                'success': False,
                'error': 'AWS S3 service not available'
            }), 503
        
        # Check if file is provided
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
            }), 400
        
        # Save temporarily and upload
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            file.save(temp_file.name)
            
            # Upload to S3
            s3_url = aws_storage.upload_video_file(
                temp_file.name, 
                secure_filename(file.filename)
            )
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if s3_url:
                return jsonify({
                    'success': True,
                    'message': 'Video uploaded successfully',
                    's3_url': s3_url,
                    'filename': file.filename
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to upload video to S3'
                }), 500
    
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@aws_bp.route('/upload/analytics', methods=['POST'])
def upload_analytics():
    """Upload analytics data to AWS S3"""
    try:
        if not aws_storage.is_available():
            return jsonify({
                'success': False,
                'error': 'AWS S3 service not available'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Add metadata
        analytics_data = {
            'uploaded_at': datetime.now().isoformat(),
            'data_type': 'traffic_analytics',
            'version': '1.0',
            'data': data
        }
        
        # Upload to S3
        s3_url = aws_storage.upload_analytics_data(analytics_data)
        
        if s3_url:
            return jsonify({
                'success': True,
                'message': 'Analytics data uploaded successfully',
                's3_url': s3_url
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to upload analytics data to S3'
            }), 500
    
    except Exception as e:
        logger.error(f"Error uploading analytics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@aws_bp.route('/files', methods=['GET'])
def list_files():
    """List files in AWS S3 bucket"""
    try:
        if not aws_storage.is_available():
            return jsonify({
                'success': False,
                'error': 'AWS S3 service not available'
            }), 503
        
        # Get query parameters
        prefix = request.args.get('prefix', '')
        limit = int(request.args.get('limit', 50))
        
        files = aws_storage.list_files(prefix=prefix, limit=limit)
        
        return jsonify({
            'success': True,
            'files': files,
            'count': len(files)
        })
    
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@aws_bp.route('/files/<path:s3_key>', methods=['DELETE'])
def delete_file(s3_key):
    """Delete file from AWS S3"""
    try:
        if not aws_storage.is_available():
            return jsonify({
                'success': False,
                'error': 'AWS S3 service not available'
            }), 503
        
        success = aws_storage.delete_file(s3_key)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'File {s3_key} deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete file'
            }), 500
    
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@aws_bp.route('/files/<path:s3_key>/url', methods=['GET'])
def get_presigned_url(s3_key):
    """Get presigned URL for secure file access"""
    try:
        if not aws_storage.is_available():
            return jsonify({
                'success': False,
                'error': 'AWS S3 service not available'
            }), 503
        
        # Get expiration time (default: 1 hour)
        expiration = int(request.args.get('expiration', 3600))
        
        presigned_url = aws_storage.generate_presigned_url(s3_key, expiration)
        
        if presigned_url:
            return jsonify({
                'success': True,
                'presigned_url': presigned_url,
                'expires_in': expiration
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate presigned URL'
            }), 500
    
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@aws_bp.route('/backup/database', methods=['POST'])
def backup_database():
    """Backup database to AWS S3"""
    try:
        if not aws_storage.is_available():
            return jsonify({
                'success': False,
                'error': 'AWS S3 service not available'
            }), 503
        
        # Get database file path
        db_path = Config.DATABASE_URI.replace('sqlite:///', '')
        
        if not os.path.exists(db_path):
            return jsonify({
                'success': False,
                'error': 'Database file not found'
            }), 404
        
        # Generate backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"database_backup_{timestamp}.db"
        
        # Upload database file
        s3_url = aws_storage.upload_video_file(db_path, backup_name)  # Reusing upload method
        
        if s3_url:
            return jsonify({
                'success': True,
                'message': 'Database backup completed successfully',
                's3_url': s3_url,
                'backup_name': backup_name
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to backup database to S3'
            }), 500
    
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
