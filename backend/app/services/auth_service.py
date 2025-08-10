"""Authentication service for handling user login/logout operations."""
import logging
from typing import Optional, Tuple
from flask_jwt_extended import create_access_token
from app.models.user import User
from app import Session


class AuthService:
    """Service class for authentication operations."""
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Authenticate user credentials.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            Tuple of (success, token, error_message)
        """
        if not username or not password:
            return False, None, "Username and password are required"
            
        session = Session()
        try:
            user = session.query(User).filter_by(username=username).first()
            if user and user.check_password(password):
                access_token = create_access_token(identity=username)
                logging.info(f"User {username} authenticated successfully")
                return True, access_token, None
            else:
                logging.warning(f"Failed authentication for username: {username}")
                return False, None, "Invalid credentials"
        except Exception as e:
            logging.error(f"Error during authentication for {username}: {e}")
            return False, None, "Authentication failed, please try again"
        finally:
            session.close()
    
    @staticmethod
    def validate_session(username: str) -> bool:
        """Validate if user session is still valid."""
        session = Session()
        try:
            user = session.query(User).filter_by(username=username).first()
            return user is not None
        except Exception as e:
            logging.error(f"Error validating session for {username}: {e}")
            return False
        finally:
            session.close()
