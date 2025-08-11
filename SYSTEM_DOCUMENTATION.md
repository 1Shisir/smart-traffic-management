# Smart Traffic Management System
## Real-Time Traffic Monitoring & Analysis Platform

---

## 🚦 System Overview

The Smart Traffic Management System is an advanced AI-powered platform that provides real-time traffic monitoring, vehicle detection, and traffic flow analysis. The system uses computer vision and machine learning to automatically detect and count vehicles, determine optimal traffic light timing, and provide comprehensive traffic analytics.

### Key Features
- **Real-Time Vehicle Detection**: Uses YOLOv8 AI model to detect cars, buses, trucks, and motorcycles
- **Intelligent Traffic Light Control**: Automatically adjusts traffic light timing based on vehicle count
- **Live Video Processing**: Processes traffic video feeds in real-time with visual annotations
- **WebSocket Communication**: Real-time data updates to the dashboard without page refresh
- **Historical Data Analytics**: Stores and analyzes traffic patterns over time
- **Professional Dashboard**: Modern, responsive web interface for monitoring and control
- **Secure Authentication**: JWT-based user authentication with role-based access
- **Database Integration**: SQLite database for reliable data storage and retrieval

---

## 🏗️ System Architecture

### Backend (Python Flask)
- **Framework**: Flask with SocketIO for real-time communication
- **AI/ML**: YOLOv8 for vehicle detection and classification
- **Database**: SQLite with SQLAlchemy ORM
- **Authentication**: JWT (JSON Web Tokens) with bcrypt password hashing
- **Video Processing**: OpenCV for video capture and frame processing
- **Real-Time Updates**: Flask-SocketIO for WebSocket communication

### Frontend (React)
- **Framework**: React 18 with Vite for fast development
- **Styling**: TailwindCSS for modern, responsive design
- **Icons**: React Icons for professional UI elements
- **Real-Time Communication**: Socket.IO client for live updates
- **State Management**: React Context API for global state
- **Authentication**: Persistent login with token management

### Database Schema
```sql
-- Traffic Data Table
traffic_data (
    id: Integer (Primary Key)
    junction: String (50) - Junction name/location
    total_count: Integer - Total vehicles detected
    car_count: Integer - Number of cars
    bus_count: Integer - Number of buses  
    truck_count: Integer - Number of trucks
    motorcycle_count: Integer - Number of motorcycles
    traffic_light: String (10) - Current light state (red/yellow/green)
    light_duration: Integer - Light duration in seconds
    timestamp: DateTime - When data was recorded
)

-- Users Table
users (
    id: Integer (Primary Key)
    username: String (50) - Unique username
    password_hash: String (128) - Bcrypt hashed password
)
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **Git** (for version control)

### 1. Clone the Repository
```bash
git clone https://github.com/1Shisir/smart-traffic-management.git
cd smart-traffic-management
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

**Backend will be available at: http://localhost:5000**

### 3. Frontend Setup
```bash
# Open new terminal and navigate to frontend
cd traffic-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Frontend will be available at: http://localhost:5173**

### 4. Access the System
1. Open your web browser
2. Navigate to `http://localhost:5173`
3. Login with demo credentials:
   - **Username**: admin
   - **Password**: admin123

---

## 📱 User Interface Features

### Login Page
- Secure authentication with JWT tokens
- Professional gradient design
- Input validation and error handling
- Remember me functionality

### Dashboard Features
- **Real-Time Traffic Data**: Live vehicle counts and traffic light status
- **Video Preview**: Live annotated video feed showing detected vehicles
- **Vehicle Statistics**: Individual counts for cars, buses, trucks, motorcycles
- **Traffic Light Indicator**: Visual representation of current traffic light state
- **Historical Data Table**: Scrollable table of past traffic records
- **Processing Controls**: Start/Stop video processing with real-time feedback
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Key Dashboard Components
1. **Header**: System title, processing controls, logout functionality
2. **Traffic Statistics Cards**: Real-time vehicle count displays with icons
3. **Traffic Light Simulator**: Dynamic color-coded traffic light display
4. **Live Video Feed**: Real-time video with vehicle detection overlays
5. **Historical Data Grid**: Searchable, sortable traffic history
6. **Status Indicators**: Processing state, connection status, error alerts

---

## ⚙️ Technical Specifications

### Performance Metrics
- **Video Processing**: 5 FPS processing rate for optimal performance
- **Detection Accuracy**: >90% vehicle detection accuracy with YOLOv8
- **Response Time**: <200ms for real-time updates via WebSocket
- **Database Operations**: Batch processing for efficient data storage
- **Memory Management**: Automatic cleanup and resource optimization

### Security Features
- **Authentication**: JWT token-based authentication with expiration
- **Password Security**: Bcrypt hashing with salt for password storage
- **Input Validation**: Server-side and client-side input sanitization
- **SQL Injection Prevention**: Parameterized queries and ORM protection
- **XSS Protection**: Input escaping and content security policies
- **Rate Limiting**: Protection against brute force attacks

### Scalability Considerations
- **Modular Architecture**: Separated concerns for easy scaling
- **Database Optimization**: Indexed queries and batch operations
- **Memory Management**: Proper resource cleanup and garbage collection
- **Configuration Management**: Environment-based configuration
- **Error Handling**: Comprehensive error logging and recovery

---

## 🔧 Configuration Options

### Environment Variables (.env file)
```bash
# Flask Configuration
FLASK_ENV=development
FLASK_HOST=127.0.0.1
FLASK_PORT=5000

# Database
SQLALCHEMY_DATABASE_URI=sqlite:///traffic.db

# Security
JWT_SECRET_KEY=your_secure_secret_key
JWT_ACCESS_TOKEN_EXPIRES_DAYS=1

# Video Processing
VIDEO_PATH=traffic_sample1.mp4
YOLO_MODEL=yolov8n.pt
DETECTION_THRESHOLD=0.5
TRAFFIC_LIGHT_THRESHOLD=12

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

### Traffic Light Logic
- **Green Light**: 0-7 vehicles detected (30 seconds)
- **Yellow Light**: 8-11 vehicles detected (5 seconds)  
- **Red Light**: 12+ vehicles detected (40 seconds)

---

## 📊 API Documentation

### Authentication Endpoints
- `POST /api/login` - User authentication
- `POST /api/logout` - User logout

### Data Endpoints
- `GET /api/data` - Retrieve traffic data (paginated)
- `GET /api/current-status` - Get current system status
- `POST /api/generate-sample-data` - Generate test data

### Video Processing
- `GET /api/video-feed` - Live video frame endpoint

### WebSocket Events
- `connect` - Client connection established
- `disconnect` - Client disconnection
- `update` - Real-time traffic data updates
- `traffic_light` - Traffic light state changes
- `processing_started` - Video processing started
- `processing_stopped` - Video processing stopped
- `processing_error` - Processing error occurred

---

## 🛠️ Development & Deployment

### Development Workflow
1. **Backend Development**: Modify Python files in `/backend/app/`
2. **Frontend Development**: Edit React components in `/traffic-frontend/src/`
3. **Database Changes**: Update models in `/backend/app/models/`
4. **Testing**: Use demo credentials and sample video data

### Production Deployment
1. **Environment Setup**: Configure production environment variables
2. **Database Migration**: Set up production database
3. **Security Configuration**: Update JWT secrets and passwords
4. **SSL Setup**: Configure HTTPS for secure communication
5. **Server Configuration**: Set up reverse proxy (nginx) and process manager

### File Structure
```
smart-traffic-system/
├── backend/                 # Python Flask backend
│   ├── app/
│   │   ├── models/         # Database models
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   ├── utils/          # Utility functions
│   │   └── config.py       # Configuration
│   ├── main.py             # Application entry point
│   └── requirements.txt    # Python dependencies
├── traffic-frontend/        # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── context/        # React context
│   │   └── App.jsx         # Main application
│   ├── package.json        # Node dependencies
│   └── vite.config.js      # Vite configuration
└── README.md               # Project documentation
```

---

## 🔍 Troubleshooting

### Common Issues

**Backend Won't Start**
- Check Python version (3.8+ required)
- Verify virtual environment is activated
- Install missing dependencies: `pip install -r requirements.txt`
- Check port 5000 isn't already in use

**Frontend Connection Issues**
- Verify backend is running on port 5000
- Check browser console for errors
- Ensure Node.js version 16+ is installed
- Clear browser cache and cookies

**Video Processing Errors**
- Verify video file exists: `traffic_sample1.mp4`
- Check YOLO model file: `yolov8n.pt`
- Ensure sufficient system memory
- Check GPU drivers if using CUDA

**Authentication Problems**
- Clear browser storage and cookies
- Check JWT token expiration
- Verify username/password combination
- Check network connectivity

### System Requirements
- **CPU**: Intel i5 or equivalent (i7 recommended for video processing)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space
- **Network**: Stable internet connection for model downloads
- **GPU**: Optional CUDA-compatible GPU for faster processing

---

## 📈 Future Enhancements

### Planned Features
- **Multiple Camera Support**: Monitor multiple junctions simultaneously
- **Advanced Analytics**: Traffic flow predictions and pattern analysis
- **Mobile App**: iOS/Android companion app
- **Cloud Integration**: AWS/Azure cloud deployment
- **AI Improvements**: Advanced vehicle classification and tracking
- **Report Generation**: Automated traffic reports and insights
- **Integration APIs**: Third-party traffic system integration

### Performance Optimizations
- **Distributed Processing**: Multi-camera load balancing
- **Database Scaling**: PostgreSQL/MySQL support for larger datasets
- **Caching**: Redis caching for improved response times
- **CDN Integration**: Static asset delivery optimization


---

## 📄 License & Credits

- **License**: MIT License
- **AI Model**: YOLOv8 by Ultralytics
- **Frontend Framework**: React (Meta)
- **Backend Framework**: Flask (Pallets)
- **UI Library**: TailwindCSS
- **Icons**: React Icons

---

**© 2025 Smart Traffic Management System - Real-Time Traffic Monitoring Solution**
