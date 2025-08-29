# 📖 Smart Traffic Management System - User Manual

## Welcome to the Smart Traffic Management System!

This comprehensive guide will help you navigate and use all features of the Smart Traffic Management System effectively. Whether you're a traffic manager, system administrator, or analyst, this manual provides step-by-step instructions for optimal system usage.

---

## 🚀 Getting Started

### System Requirements
- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Internet Connection**: Required for real-time updates
- **Screen Resolution**: Minimum 1024x768 (recommended: 1920x1080)

### Accessing the System
1. Open your web browser
2. Navigate to the system URL (default: `http://localhost:5174`)
3. You'll be directed to the login page

---

## 🔐 Login & Authentication

### First Time Login
1. **Enter Credentials**:
   - Username: `admin`
   - Password: `password123`
2. Click **"Login"** button
3. Upon successful login, you'll be redirected to the main dashboard

### System Status Indicators
- **🟢 Green Header**: System is online and connected
- **🟠 Orange "Server Unavailable" Bar**: Backend is offline
  - Click **"Retry"** to check connection again
  - System will automatically reconnect when backend is available

---

## 📊 Main Dashboard Overview

The dashboard is divided into several key sections:

### 1. **Header Section**
- **System Title**: "Smart Traffic Management System"
- **Network Status**: Connection indicator
- **Current Time**: Live system clock

### 2. **Control Panel** (Top Section)
- **Start/Stop Processing**: Control video analysis
- **Status Indicators**: Processing state and system health

### 3. **Real-Time Data** (Left Side)
- **Current Traffic Count**: Live vehicle detection numbers
- **Vehicle Breakdown**: Cars, buses, trucks, motorcycles
- **Traffic Light Status**: Current signal state and timing

### 4. **Historical Data** (Right Side)
- **Recent Records**: Table of past traffic data
- **Time Stamps**: When each record was captured
- **Data Export**: Download historical information

### 5. **Video Feed** (Center)
- **Live Processing View**: Real-time video with AI detection overlay
- **Vehicle Annotations**: Colored boxes around detected vehicles
- **Processing Controls**: Play/pause/stop functionality

### 6. **AI Detection Video** (New Feature)
- **Detection Video Access**: Button below live video feed
- **Frame Preview**: View individual frames with detection boundaries
- **Full Video Download**: Process and download complete video with AI annotations
- **Progress Tracking**: Monitor video processing status

---

## 🎮 Core Functions

### Starting Traffic Analysis
1. **Start Processing**:
   - Click **"Start Processing"** button
   - System will begin analyzing video feed
   - You'll see real-time vehicle detection overlays

2. **Monitor Results**:
   - Watch live vehicle counts update
   - Observe traffic light timing adjustments
   - View detection boxes on video feed

### Stopping Traffic Analysis

1. Click **"Stop Processing"** button
2. Video analysis will cease
3. Last recorded data remains visible
4. System returns to standby mode

### Understanding Vehicle Detection

The AI system detects and counts:
- **🚗 Cars**: Standard passenger vehicles (Blue boxes)
- **🚌 Buses**: Public transportation vehicles (Green boxes)
- **🚛 Trucks**: Commercial/freight vehicles (Red boxes)
- **🏍️ Motorcycles**: Two-wheeled vehicles (Yellow boxes)

### Traffic Light Intelligence

The system automatically:
- **Analyzes Traffic Density**: Counts vehicles in real-time
- **Adjusts Signal Timing**: Optimizes green/red light duration
- **Reduces Wait Times**: Minimizes traffic congestion
- **Provides Recommendations**: Suggests optimal timing settings

### Using AI Vehicle Detection Video

**Accessing the Feature**:
1. Look for **"🤖 View AI Detection Video"** button below the live video feed
2. Click the button to navigate to the AI Detection Video page

**Preview Detection on Single Frames**:
1. **Enter Frame Number**: Input the specific frame you want to analyze
2. **Load Preview**: Click "Load Preview Frame" to see detection results
3. **View Results**: Observe vehicle detection boundaries and counts
4. **Frame Navigation**: Try different frame numbers to explore the video

**Download Full Video with Detection**:
1. **Start Processing**: Click "📥 Download Video with Detection" button
2. **Wait for Processing**: AI will analyze every frame (this may take several minutes)
3. **Progress Monitoring**: Watch for processing status updates
4. **Automatic Download**: Processed video will download when complete
5. **File Format**: Video saves as MP4/AVI with detection overlays

**What You'll Get in the Processed Video**:
- ✅ **Vehicle Boundaries**: Colored boxes around all detected vehicles
- ✅ **Vehicle Counts**: Real-time count display on each frame
- ✅ **Detection Confidence**: Accuracy percentages for each detection
- ✅ **Frame Information**: Frame number and total frame count
- ✅ **Vehicle Categories**: Separate counts for cars, buses, trucks, motorcycles

---

## 📈 Data Management

### Viewing Historical Data

1. **Data Table**: Located on the right side of dashboard
2. **Columns Include**:
   - Timestamp
   - Total vehicles
   - Vehicle type breakdown
   - Traffic light state
   - Signal duration

3. **Sorting**: Click column headers to sort data
4. **Filtering**: Use date/time filters if available

### Real-Time Status

- **Live Updates**: Data refreshes every few seconds
- **Processing Status**: Shows if analysis is active
- **Last Update**: Timestamp of most recent data
- **System Health**: Overall system status

---

## ☁️ AWS Cloud Storage

### Accessing Cloud Features

1. Navigate to **AWS Storage** tab in dashboard
2. Features include:
   - **File Backup**: Video and data storage
   - **Analytics Export**: Download traffic reports
   - **Cloud Sync**: Automatic data backup

### Downloading Files

1. **View Files**: Browse uploaded videos and data
2. **Download**: Click download button next to any file
3. **New Tab**: Downloads open in new browser tab
4. **Analytics**: Export traffic analysis reports

### File Management

- **Upload Videos**: Add new traffic footage for analysis
- **Backup Data**: Store traffic records in cloud
- **Export Reports**: Generate and download analytics

---

## 🔧 Troubleshooting

### Common Issues

#### "Server Unavailable" Message
**Problem**: Orange bar at top of screen
**Solution**: 
- Check if backend server is running
- Click "Retry" button
- Contact system administrator if problem persists

#### Video Not Loading
**Problem**: Black screen or error in video area
**Solution**:
- Ensure video file exists and is accessible
- Check video format compatibility (MP4 recommended)
- Restart video processing

#### AI Detection Video Processing Fails
**Problem**: Download fails or processing errors
**Solution**:
- Check available disk space (processing requires temporary storage)
- Ensure stable internet connection during processing
- Try processing a shorter video segment first
- Contact administrator if YOLO model fails to load

#### Detection Video Download Issues
**Problem**: Processed video fails to download or is corrupted
**Solution**:
- Clear browser downloads and try again
- Check browser permissions for file downloads
- Ensure sufficient storage space on local device
- Try using a different browser if issues persist

#### No Vehicle Detection
**Problem**: Video plays but no detection boxes appear
**Solution**:
- Verify AI model is loaded (YOLOv8)
- Check processing is actually started
- Ensure adequate lighting in video

#### Login Issues
**Problem**: Cannot access system
**Solution**:
- Verify username and password
- Check network connection
- Clear browser cache and cookies

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "Invalid credentials" | Wrong username/password | Use correct login details |
| "Processing already active" | Another session is running | Stop current processing first |
| "Video file not found" | Missing video source | Upload or select valid video |
| "Database connection failed" | Backend database issue | Contact administrator |
| "Failed to load preview frame" | AI detection preview error | Check video file and try different frame |
| "Failed to process video" | AI video processing failed | Ensure sufficient resources and try again |
| "Video processing timeout" | Processing took too long | Try shorter video or contact support |

---

## 💡 Best Practices

### For Optimal Performance

1. **Regular Monitoring**: Check system status frequently
2. **Data Management**: Export and backup data regularly
3. **Video Quality**: Use high-resolution, well-lit footage
4. **Browser Maintenance**: Keep browser updated and clear cache
5. **Network Stability**: Ensure stable internet connection
6. **AI Video Processing**: Allow sufficient time for detection video processing
7. **Storage Management**: Maintain adequate free disk space for video processing

### For Accurate Results

1. **Camera Positioning**: Ensure clear view of intersection
2. **Lighting Conditions**: Adequate illumination for detection
3. **Video Quality**: Minimum 720p resolution recommended
4. **Processing Duration**: Allow sufficient time for analysis
5. **Regular Calibration**: Verify detection accuracy periodically
6. **Frame Selection**: Test different frames for optimal detection preview
7. **Video Format**: Use MP4 format for best compatibility

---

## 📋 Quick Reference

### Keyboard Shortcuts
- **Ctrl + R**: Refresh dashboard
- **Ctrl + L**: Focus on login fields
- **Esc**: Close modal dialogs
- **F5**: Reload page
- **←**: Navigate back from AI Detection Video page

### Status Icons
- 🟢 **Green**: System operational
- 🟡 **Yellow**: Warning/processing
- 🔴 **Red**: Error/offline
- 🔵 **Blue**: Information/neutral
- 🤖 **Robot**: AI processing active
- 📥 **Download**: File download available

### Default Settings
- **Junction**: Main St & 1st Ave
- **Traffic Light Duration**: 30 seconds
- **Data Refresh**: Every 3 seconds
- **Video Format**: MP4
- **Detection Model**: YOLOv8

---

## 🆘 Support & Contact

### Getting Help

1. **System Issues**: Check troubleshooting section first
2. **Technical Support**: Contact your system administrator
3. **Feature Requests**: Submit feedback through appropriate channels
4. **Documentation**: Refer to technical documentation for advanced features

### Additional Resources

- **System Documentation**: `SYSTEM_DOCUMENTATION.md`
- **Deployment Guide**: `DEPLOYMENT.md`
- **AWS Setup**: `AWS_SETUP.md`
- **API Reference**: Available in system documentation

---

## 🔄 System Updates

The Smart Traffic Management System is regularly updated with:
- **New Features**: Enhanced functionality and capabilities
- **Bug Fixes**: Resolution of known issues
- **Performance Improvements**: Faster processing and better accuracy
- **Security Updates**: Enhanced protection and authentication

### Update Notifications
- System will display notifications for important updates
- Check with administrator for update schedules
- Backup important data before major updates

---

## 📝 Conclusion

The Smart Traffic Management System provides powerful tools for monitoring and optimizing traffic flow. By following this user manual, you'll be able to:

✅ **Successfully log in and navigate the system**  
✅ **Start and monitor traffic analysis**  
✅ **Interpret vehicle detection and traffic data**  
✅ **Manage historical data and exports**  
✅ **Troubleshoot common issues**  
✅ **Utilize cloud storage features**  
✅ **Access AI vehicle detection video processing**  
✅ **Preview individual frames with detection boundaries**  
✅ **Download complete videos with AI annotations**  
✅ **Navigate between dashboard and detection video pages**  

For additional support or advanced features, consult the technical documentation or contact your system administrator.

---

*This manual is current as of August 2025. For the latest updates and features, refer to the system documentation.*