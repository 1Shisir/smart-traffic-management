# Smart Traffic Management System - Deployment Guide

This guide covers how to deploy and run the Smart Traffic Management System without Docker.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Quick Start

### Windows

1. **Download or clone the project**
   ```bash
   git clone <repository-url>
   cd smart-traffic-system/backend
   ```

2. **Run the startup script**
   ```bash
   start.bat
   ```
   
   This script will:
   - Create a virtual environment
   - Install dependencies
   - Initialize the database
   - Start the application

3. **Access the application**
   - Open your browser and go to: http://localhost:5000
   - Default login: `admin` / `admin123`

### Linux/Mac

1. **Download or clone the project**
   ```bash
   git clone <repository-url>
   cd smart-traffic-system/backend
   ```

2. **Make the startup script executable and run it**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

3. **Access the application**
   - Open your browser and go to: http://localhost:5000
   - Default login: `admin` / `admin123`

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional)

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
JWT_SECRET_KEY=your-super-secret-jwt-key
LOG_LEVEL=INFO
```

### 4. Initialize Database

```bash
python manage.py init-db
```

### 5. Start the Application

```bash
python main.py
```

## Required Files

Make sure these files are in the backend directory:
- `traffic_sample1.mp4` - Sample traffic video
- `yolov8n.pt` - YOLO model file

If missing, the application will log warnings but may still run with limited functionality.

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | development | Application environment |
| `FLASK_HOST` | 127.0.0.1 | Host to bind to |
| `FLASK_PORT` | 5000 | Port to listen on |
| `JWT_SECRET_KEY` | (default) | JWT signing key |
| `LOG_LEVEL` | INFO | Logging level |
| `SQLALCHEMY_DATABASE_URI` | sqlite:///traffic.db | Database connection |

### Config File

You can also modify `app/config.py` directly for persistent configuration changes.

## Management Commands

The application includes a CLI for management tasks:

```bash
# Initialize database
python manage.py init-db

# Create a new user
python manage.py create-user <username> <password>

# List all users
python manage.py list-users

# Clear traffic data
python manage.py clear-traffic-data

# Validate configuration
python manage.py validate-config

# Run with custom settings
python manage.py run --host 0.0.0.0 --port 8080 --debug
```

## Health Monitoring

The application includes health check endpoints:

- **Basic health check:** http://localhost:5000/health
- **Detailed health check:** http://localhost:5000/health/detailed

## Troubleshooting

### Common Issues

1. **Python not found**
   - Make sure Python 3.8+ is installed and in your PATH
   - Try using `python3` instead of `python` on Linux/Mac

2. **Permission errors**
   - Make sure you have write permissions in the project directory
   - Try running as administrator (Windows) or with sudo (Linux/Mac)

3. **Port already in use**
   - Change the port using: `python main.py` with `FLASK_PORT=8080` environment variable
   - Or kill the process using the port

4. **Missing video/model files**
   - Download the required files and place them in the backend directory
   - Check the file paths in `app/config.py`

5. **Database errors**
   - Delete the `traffic.db` file and run `python manage.py init-db` again

### Logs

Check the application logs in:
- Console output (when running)
- `app.log` file (if configured)

## Production Deployment

For production deployment without Docker:

1. **Use a proper WSGI server**
   ```bash
   pip install gunicorn  # Linux/Mac
   pip install waitress  # Windows
   ```

2. **Set production environment**
   ```bash
   export FLASK_ENV=production
   ```

3. **Use a reverse proxy**
   - Nginx (recommended)
   - Apache
   - Cloudflare

4. **Use a proper database**
   - PostgreSQL
   - MySQL
   - SQLite (for small deployments)

5. **Set up monitoring**
   - Use the built-in health endpoints
   - Set up log aggregation
   - Monitor system resources

## Security Considerations

1. **Change default passwords**
   - Change the default admin password
   - Use strong JWT secret keys

2. **Use HTTPS**
   - Set up SSL certificates
   - Configure secure cookie settings

3. **Firewall configuration**
   - Only expose necessary ports
   - Use proper network security

4. **Regular updates**
   - Keep dependencies updated
   - Monitor for security vulnerabilities
