"""
Simple tests for AWSStorageService
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from app.services.aws_service import AWSStorageService


class TestAWSStorageService:
    """Test cases for AWSStorageService"""
    
    @patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret',
        'AWS_REGION': 'us-east-1',
        'AWS_S3_BUCKET_NAME': 'test-bucket'
    })
    @patch('app.services.aws_service.boto3.client')
    def test_aws_service_initialization_success(self, mock_boto_client):
        """Test successful AWS service initialization"""
        # Setup mock
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.return_value = True
        
        # Test
        service = AWSStorageService()
        
        # Assertions
        assert service.s3_client is not None
        assert service.bucket_name == 'test-bucket'
        assert service.region_name == 'us-east-1'
        mock_boto_client.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_aws_service_initialization_no_credentials(self):
        """Test AWS service initialization without credentials"""
        # Test
        service = AWSStorageService()
        
        # Assertions
        assert service.s3_client is None
    
    @patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret'
    })
    @patch('app.services.aws_service.boto3.client')
    def test_is_available(self, mock_boto_client):
        """Test availability check"""
        # Test with valid client
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_bucket.return_value = True
        
        service = AWSStorageService()
        assert service.is_available() is True
        
        # Test with no client
        service.s3_client = None
        assert service.is_available() is False
