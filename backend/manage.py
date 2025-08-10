#!/usr/bin/env python3
"""
Management CLI for Smart Traffic System

This module provides command-line management tools for the application.
"""

import click
import logging
import os
import sys
from typing import Optional

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, Session
from app.models.user import User
from app.models.traffic_data import TrafficData
from app.config import config_map


@click.group()
def cli():
    """Smart Traffic Management System CLI."""
    pass


@cli.command()
@click.option('--config', default='development', help='Configuration environment')
def init_db(config: str):
    """Initialize the database."""
    try:
        app, _ = create_app(config)
        click.echo(f"Database initialized successfully with {config} configuration!")
    except Exception as e:
        click.echo(f"Database initialization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('username')
@click.argument('password')
@click.option('--config', default='development', help='Configuration environment')
def create_user(username: str, password: str, config: str):
    """Create a new user."""
    try:
        app, _ = create_app(config)
        
        session = Session()
        
        # Check if user already exists
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            click.echo(f"User '{username}' already exists!", err=True)
            return
        
        # Create new user
        user = User(username=username)
        user.set_password(password)
        session.add(user)
        session.commit()
        
        click.echo(f"User '{username}' created successfully!")
        
    except Exception as e:
        click.echo(f"Failed to create user: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.option('--config', default='development', help='Configuration environment')
def list_users(config: str):
    """List all users."""
    try:
        app, _ = create_app(config)
        
        session = Session()
        users = session.query(User).all()
        
        if not users:
            click.echo("No users found.")
            return
        
        click.echo("Users:")
        for user in users:
            click.echo(f"  - {user.username} (ID: {user.id})")
            
    except Exception as e:
        click.echo(f"Failed to list users: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.option('--config', default='development', help='Configuration environment')
def clear_traffic_data(config: str):
    """Clear all traffic data."""
    try:
        app, _ = create_app(config)
        
        if not click.confirm('Are you sure you want to delete all traffic data?'):
            click.echo('Operation cancelled.')
            return
        
        session = Session()
        deleted_count = session.query(TrafficData).delete()
        session.commit()
        
        click.echo(f"Deleted {deleted_count} traffic data records.")
        
    except Exception as e:
        click.echo(f"Failed to clear traffic data: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.option('--config', default='development', help='Configuration environment')
def validate_config(config: str):
    """Validate application configuration."""
    try:
        config_class = config_map.get(config, config_map['default'])
        issues = config_class.validate_config()
        
        if not issues:
            click.echo("✅ Configuration is valid!")
        else:
            click.echo("⚠️  Configuration issues found:")
            for issue in issues:
                click.echo(f"  - {issue}")
                
    except Exception as e:
        click.echo(f"Configuration validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=5000, help='Port to bind to')
@click.option('--config', default='development', help='Configuration environment')
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
def run(host: str, port: int, config: str, debug: bool):
    """Run the application server."""
    try:
        os.environ['FLASK_HOST'] = host
        os.environ['FLASK_PORT'] = str(port)
        if debug:
            os.environ['FLASK_ENV'] = 'development'
        
        # Import and run main
        from main import main
        main()
        
    except Exception as e:
        click.echo(f"Failed to start server: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
