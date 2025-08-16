#!/usr/bin/env python3
"""
Smart Traffic Management System - Main Application Entry Point

This module serves as the main entry point for the Flask application.
It handles application initialization, database setup, and server startup.
"""

import os
import sys
import signal
import logging
import threading
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

from app import create_app
from app.config import Config
from app.models.traffic_data import Base
from app.models.user import User
from app.utils.realtime_polling import clear_stale_data
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
app_instance: Optional[object] = None
socketio_instance: Optional[object] = None


def setup_database() -> None:
    """Initialize database and create tables if they don't exist."""
    try:
        logger.info("Setting up database...")
        engine = create_engine(
            Config.SQLALCHEMY_DATABASE_URI, 
            echo=Config.SQLALCHEMY_ECHO
        )
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    
    # Add cleanup logic here if needed
    # For example: stop background threads, close database connections, etc.
    
    sys.exit(0)


def validate_environment() -> bool:
    """Validate that all required environment variables and files exist."""
    required_files = [
        Config.VIDEO_PATH,
        Config.YOLO_MODEL
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    # Validate configuration
    if not Config.JWT_SECRET_KEY:
        logger.error("JWT secret key is not set")
        return False
    
    return True


def main() -> None:
    """Main application entry point."""
    global app_instance, socketio_instance
    
    try:
        logger.info("Starting Smart Traffic Management System...")
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Setup database
        setup_database()
        
        # Clear any stale realtime data from previous runs
        clear_stale_data()
        
        # Create Flask app and SocketIO instance
        app_instance, socketio_instance = create_app()
        
        # Get configuration
        host = os.getenv('FLASK_HOST', '127.0.0.1')
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_ENV') == 'development'
        
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        # Start the server
        socketio_instance.run(
            app_instance,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False  # Prevent double execution in debug mode
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ == '__main__':
    main()