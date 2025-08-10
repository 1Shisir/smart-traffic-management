"""
Smart Traffic Management System - Application Factory

This module contains the Flask application factory and initialization logic.
"""

import os
import logging
from datetime import datetime
from typing import Tuple, Optional

from flask import Flask, redirect, url_for
from flask_socketio import SocketIO, disconnect
from flask_jwt_extended import JWTManager, get_jwt_identity, verify_jwt_in_request
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import config_map, Config
from app.models.user import User, Base as UserBase
from app.models.traffic_data import Base as TrafficBase

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global session factory - initialized during app creation
Session: Optional[sessionmaker] = None


def create_app(config_name: Optional[str] = None) -> Tuple[Flask, SocketIO]:
    """
    Application factory function.
    
    Args:
        config_name: Configuration environment name ('development', 'production', 'testing')
        
    Returns:
        Tuple of (Flask app, SocketIO instance)
    """
    global Session
    
    # Determine configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    config_class = config_map.get(config_name, config_map['default'])
    
    # Create Flask app
    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static'
    )
    
    # Configure app
    app.config.from_object(config_class)
    
    # Initialize CORS
    CORS(app, 
         origins=['http://localhost:5173', 'http://localhost:5174', 'http://localhost:3000'],  # Allow React dev server
         supports_credentials=True,  # Allow cookies/credentials
         allow_headers=['Content-Type', 'Authorization'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
    
    # Validate configuration
    config_issues = config_class.validate_config()
    if config_issues:
        for issue in config_issues:
            logger.warning(f"Config issue: {issue}")
    
    # Initialize extensions
    socketio = SocketIO(
        app, 
        cors_allowed_origins=app.config.get('CORS_ORIGINS', '*'),
        cookie=True,  # Enable cookie-based authentication
        logger=True,
        engineio_logger=True
    )
    jwt = JWTManager(app)
    
    # Setup JWT handlers
    @jwt.unauthorized_loader
    def unauthorized_callback(error):
        """Redirect unauthorized users to login page."""
        logger.warning(f"Unauthorized access attempt: {error}")
        return redirect(url_for('api.login'))
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        """Handle expired tokens."""
        logger.info("Expired token detected, redirecting to login")
        return redirect(url_for('api.login'))
    
    # Setup SocketIO handlers
    @socketio.on('connect')
    def handle_connect(auth):
        """Handle WebSocket connection with JWT verification."""
        try:
            # Try to verify JWT from cookies
            verify_jwt_in_request()
            user = get_jwt_identity()
            logger.info(f"User {user} connected via WebSocket")
            
            # Join user to a room for targeted messaging
            from flask_socketio import join_room
            join_room(f"user_{user}")
            
            # Send connection confirmation
            socketio.emit('connection_status', {
                'status': 'connected',
                'user': user,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"WebSocket connection rejected: {e}")
            # Don't disconnect immediately, let frontend handle authentication
            socketio.emit('auth_required', {
                'message': 'Authentication required',
                'redirect': '/login'
            })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        try:
            user = get_jwt_identity()
            if user:
                logger.info(f"User {user} disconnected from WebSocket")
            else:
                logger.info("Anonymous user disconnected from WebSocket")
        except Exception:
            logger.info("User disconnected from WebSocket")
    
    # Initialize database
    _initialize_database(app)
    
    # Register blueprints
    _register_blueprints(app)
    
    # Register SocketIO handlers
    from app.routes.api import register_socketio_handlers
    register_socketio_handlers(socketio)
    
    # Set socketio instance for video processor
    from app.utils.video_processor import set_socketio
    set_socketio(socketio)
    
    logger.info(f"Application created with config: {config_name}")
    return app, socketio


def _initialize_database(app: Flask) -> None:
    """Initialize database and create test users."""
    global Session
    
    try:
        # Create engine and session factory
        engine = create_engine(
            app.config['SQLALCHEMY_DATABASE_URI'],
            echo=app.config.get('SQLALCHEMY_ECHO', False)
        )
        
        # Create all tables
        UserBase.metadata.create_all(engine)
        TrafficBase.metadata.create_all(engine)
        
        # Create session factory
        Session = sessionmaker(bind=engine)
        
        # Create test users if they don't exist
        _create_test_users(app)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def _create_test_users(app: Flask) -> None:
    """Create test users if they don't exist."""
    session = Session()
    try:
        for user_data in app.config.get('TEST_USERS', []):
            existing_user = session.query(User).filter_by(
                username=user_data['username']
            ).first()
            
            if not existing_user:
                user = User(username=user_data['username'])
                user.set_password(user_data['password'])
                session.add(user)
                logger.info(f"Created test user: {user_data['username']}")
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to create test users: {e}")
        raise
    finally:
        session.close()


def _register_blueprints(app: Flask) -> None:
    """Register application blueprints."""
    try:
        from app.routes.api import api
        from app.utils.health import health
        
        app.register_blueprint(api, url_prefix='/api')
        app.register_blueprint(health)
        
        logger.info("Blueprints registered successfully")
    except Exception as e:
        logger.error(f"Failed to register blueprints: {e}")
        raise