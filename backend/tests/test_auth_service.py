"""
Simple tests for AuthService
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.auth_service import AuthService


class TestAuthService:
    """Test cases for AuthService"""
    
    @patch('app.services.auth_service.Session')
    @patch('app.services.auth_service.create_access_token')
    def test_authenticate_user_success(self, mock_create_token, mock_session_class):
        """Test successful user authentication"""
        # Setup mocks
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        mock_user = Mock()
        mock_user.check_password.return_value = True
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_user
        
        mock_create_token.return_value = "test_token_123"
        
        # Test
        success, token, error = AuthService.authenticate_user("testuser", "password123")
        
        # Assertions
        assert success is True
        assert token == "test_token_123"
        assert error is None
        mock_session.close.assert_called_once()
    
    @patch('app.services.auth_service.Session')
    def test_authenticate_user_invalid_credentials(self, mock_session_class):
        """Test authentication with invalid credentials"""
        # Setup mocks
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        
        # Test
        success, token, error = AuthService.authenticate_user("wronguser", "wrongpass")
        
        # Assertions
        assert success is False
        assert token is None
        assert error == "Invalid credentials"
        mock_session.close.assert_called_once()
    
    def test_authenticate_user_empty_credentials(self):
        """Test authentication with empty credentials"""
        # Test empty username
        success, token, error = AuthService.authenticate_user("", "password")
        assert success is False
        assert token is None
        assert error == "Username and password are required"
        
        # Test empty password
        success, token, error = AuthService.authenticate_user("user", "")
        assert success is False
        assert token is None
        assert error == "Username and password are required"
