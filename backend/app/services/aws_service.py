"""
AWS S3 Storage Service for Smart Traffic Management System

This service handles:
- Video file uploads to S3
- Processed data storage
- Analytics data backup
- File management and retrieval
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
from werkzeug.utils import secure_filename

from app.config import Config

logger = logging.getLogger(__name__)

class AWSStorageService:
    """AWS S3 Storage Service for traffic management system"""
    
    def __init__(self):
        """Initialize AWS S3 client"""
        self.s3_client = None
        self.bucket_name = None
        self.region_name = None
        self._initialize_s3()
    
    def _initialize_s3(self):
        """Initialize S3 client with credentials"""
        try:
            # Get AWS configuration from environment variables
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            self.region_name = os.getenv('AWS_REGION', 'ap-southeast-2')
            self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME', 'smart-traffic-system-bucket')
            
            if not aws_access_key or not aws_secret_key:
                logger.warning("AWS credentials not found in environment variables")
                return
            
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=self.region_name
            )
            
            # Test connection and create bucket if it doesn't exist
            self._ensure_bucket_exists()
            
            logger.info(f"AWS S3 service initialized successfully for bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS S3 service: {e}")
            self.s3_client = None
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create if it doesn't"""
        if not self.s3_client:
            return False
        
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' exists and is accessible")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.region_name == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region_name}
                        )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create S3 bucket: {create_error}")
                    return False
            else:
                logger.error(f"Error accessing S3 bucket: {e}")
                return False
        
        return True
    
    def is_available(self) -> bool:
        """Check if AWS S3 service is available"""
        return self.s3_client is not None
    
    def upload_video_file(self, file_path: str, video_name: str = None) -> Optional[str]:
        """
        Upload video file to S3
        
        Args:
            file_path: Local path to video file
            video_name: Optional custom name for the video
            
        Returns:
            S3 URL of uploaded file or None if failed
        """
        if not self.is_available():
            logger.warning("AWS S3 service not available")
            return None
        
        if not os.path.exists(file_path):
            logger.error(f"Video file not found: {file_path}")
            return None
        
        try:
            # Generate S3 key
            if not video_name:
                video_name = os.path.basename(file_path)
            
            video_name = secure_filename(video_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"videos/{timestamp}_{video_name}"
            
            # Upload file
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'video/mp4',
                    'Metadata': {
                        'upload_time': timestamp,
                        'original_name': video_name
                    }
                }
            )
            
            # Generate URL
            s3_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"
            
            logger.info(f"Video uploaded successfully to S3: {s3_key}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Failed to upload video to S3: {e}")
            return None
    
    def upload_analytics_data(self, data: Dict[Any, Any], filename: str = None) -> Optional[str]:
        """
        Upload analytics data to S3 as JSON
        
        Args:
            data: Dictionary containing analytics data
            filename: Optional custom filename
            
        Returns:
            S3 URL of uploaded file or None if failed
        """
        if not self.is_available():
            logger.warning("AWS S3 service not available")
            return None
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analytics_{timestamp}.json"
            
            filename = secure_filename(filename)
            s3_key = f"analytics/{filename}"
            
            # Convert data to JSON
            json_data = json.dumps(data, indent=2, default=str)
            
            # Upload JSON data
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'upload_time': datetime.now().isoformat(),
                    'data_type': 'analytics'
                }
            )
            
            # Generate URL
            s3_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"
            
            logger.info(f"Analytics data uploaded successfully to S3: {s3_key}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Failed to upload analytics data to S3: {e}")
            return None
    
    def upload_processed_frame(self, frame_data: bytes, frame_id: str) -> Optional[str]:
        """
        Upload processed frame image to S3
        
        Args:
            frame_data: Binary image data
            frame_id: Unique identifier for the frame
            
        Returns:
            S3 URL of uploaded image or None if failed
        """
        if not self.is_available():
            logger.warning("AWS S3 service not available")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"processed_frames/{timestamp}_{frame_id}.jpg"
            
            # Upload image data
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=frame_data,
                ContentType='image/jpeg',
                Metadata={
                    'upload_time': timestamp,
                    'frame_id': frame_id,
                    'data_type': 'processed_frame'
                }
            )
            
            # Generate URL
            s3_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"
            
            logger.info(f"Processed frame uploaded successfully to S3: {s3_key}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Failed to upload processed frame to S3: {e}")
            return None
    
    def list_files(self, prefix: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket
        
        Args:
            prefix: Prefix to filter files (e.g., 'videos/', 'analytics/')
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        if not self.is_available():
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=limit
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{obj['Key']}"
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            s3_key: S3 object key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"File deleted successfully from S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file from S3: {e}")
            return False
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for secure file access
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if failed
        """
        if not self.is_available():
            return None
        
        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            return presigned_url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        if not self.is_available():
            return {'available': False}
        
        try:
            # Get bucket size and object count
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            total_size = 0
            object_count = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    total_size += obj['Size']
                    object_count += 1
            
            # Get objects by type
            videos = len([obj for obj in response.get('Contents', []) if obj['Key'].startswith('videos/')])
            analytics = len([obj for obj in response.get('Contents', []) if obj['Key'].startswith('analytics/')])
            frames = len([obj for obj in response.get('Contents', []) if obj['Key'].startswith('processed_frames/')])
            
            return {
                'available': True,
                'bucket_name': self.bucket_name,
                'region': self.region_name,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_objects': object_count,
                'videos_count': videos,
                'analytics_count': analytics,
                'frames_count': frames
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {'available': False, 'error': str(e)}

# Global instance
aws_storage = AWSStorageService()
