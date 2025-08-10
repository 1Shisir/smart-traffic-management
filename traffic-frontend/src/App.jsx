import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import { useAuth } from './context/AuthContext.jsx';
import Login from './pages/Login.jsx';
import Dashboard from './components/Dashboard.jsx';
import LoadingScreen from './components/LoadingScreen.jsx';
import './App.css';

function App() {
  const { isAuthenticated, isLoading, token, logout: authLogout } = useAuth();
  const [socket, setSocket] = useState(null);
  const [trafficData, setTrafficData] = useState({
    junction: 'Main St & 1st Ave',
    time: 'Loading...',
    count: 0,
    car: 0,
    bus: 0,
    truck: 0,
    motorcycle: 0,
    traffic_light: 'red',
    light_duration: 30
  });
  const [history, setHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Debug effect to track isProcessing state changes
  useEffect(() => {
    console.log('🔄 isProcessing state changed to:', isProcessing);
    console.log('🔄 Will pass isProcessing =', isProcessing, 'to Dashboard component');
  }, [isProcessing]);

  // Load initial data when authenticated
  useEffect(() => {
    if (isAuthenticated && token) {
      loadInitialData();
    }
  }, [isAuthenticated, token]);

  const loadInitialData = async () => {
    try {
      console.log('📊 Loading initial traffic data...');
      
      // First, try to get current status
      const statusResponse = await fetch('http://localhost:5000/api/current-status', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        console.log('📊 Current status loaded:', statusData);
        
        // Update current traffic data
        if (statusData.current_data) {
          setTrafficData(prev => ({
            ...prev,
            junction: statusData.current_data.junction,
            time: statusData.current_data.time,
            count: statusData.current_data.count,
            car: statusData.current_data.car,
            bus: statusData.current_data.bus,
            truck: statusData.current_data.truck,
            motorcycle: statusData.current_data.motorcycle,
            traffic_light: statusData.current_data.traffic_light,
            light_duration: statusData.current_data.light_duration
          }));
        }
        
        // Update processing state from backend
        if (statusData.system_status) {
          setIsProcessing(statusData.system_status.processing_active);
          console.log('📊 Processing state from backend:', statusData.system_status.processing_active);
        }
      }
      
      // Load historical data
      const historyResponse = await fetch('http://localhost:5000/api/data?limit=20', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (historyResponse.ok) {
        const historyData = await historyResponse.json();
        console.log('📊 History data loaded:', historyData.length, 'records');
        
        if (historyData.length > 0) {
          setHistory(historyData);
          
          // If no current data was available, use the latest historical record
          if (!statusResponse.ok) {
            const latest = historyData[0];
            setTrafficData(prev => ({
              ...prev,
              junction: latest.junction || 'Main St & 1st Ave',
              time: latest.time || new Date().toLocaleTimeString(),
              count: latest.total_count || 0,
              car: latest.car_count || 0,
              bus: latest.bus_count || 0,
              truck: latest.truck_count || 0,
              motorcycle: latest.motorcycle_count || 0,
              traffic_light: latest.traffic_light || 'red',
              light_duration: latest.light_duration || 30
            }));
          }
        } else {
          console.log('📊 No historical data found, generating sample data...');
          await generateSampleData();
        }
      } else {
        console.error('📊 Failed to load history data:', historyResponse.status);
        // Try to generate sample data if no history exists
        await generateSampleData();
      }
    } catch (error) {
      console.error('📊 Error loading initial data:', error);
      // Try to generate sample data as fallback
      await generateSampleData();
    }
  };

  const generateSampleData = async () => {
    try {
      console.log('🎲 Generating sample data...');
      const response = await fetch('http://localhost:5000/api/generate-sample-data', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ count: 20 })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('🎲 Sample data generated:', result.message);
        // Reload data after generation
        setTimeout(() => loadInitialData(), 1000);
      } else {
        console.error('🎲 Failed to generate sample data');
      }
    } catch (error) {
      console.error('🎲 Error generating sample data:', error);
    }
  };

  // Initialize socket connection after authentication
  useEffect(() => {
    if (isAuthenticated && token && !socket) {
      console.log('🔌 Initializing socket connection with authenticated token...');
      const newSocket = io('http://localhost:5000', {
        auth: { token },
        transports: ['websocket', 'polling'],
        upgrade: true,
        rememberUpgrade: true
      });

        newSocket.on('connect', () => {
          console.log('🔌 Socket connected successfully');
          console.log('🔌 Socket ID:', newSocket.id);
        });

        newSocket.on('disconnect', () => {
          console.log('🔌 Socket disconnected');
        });

        newSocket.on('connect_error', (error) => {
          console.error('🔌 Socket connection error:', error);
          // If authentication error, clear token and logout
          if (error.message && error.message.includes('authentication')) {
            console.log('Socket authentication failed, logging out');
            logout();
          }
        });

        // Listen for real-time traffic data updates
        newSocket.on('update', (data) => {
          console.log('Received real-time update:', data);
          setTrafficData(data);
          setHistory(prev => [data, ...prev.slice(0, 49)]); // Keep last 50 records
        });

        // Listen for traffic light state changes
        newSocket.on('traffic_light', (data) => {
          console.log('Received traffic light update:', data);
          setTrafficData(prev => ({
            ...prev,
            traffic_light: data.state,
            light_duration: data.duration
          }));
        });

        // Listen for processing state changes
        newSocket.on('processing_started', (data) => {
          console.log('✅ Processing started event received:', data);
          setIsProcessing(prevState => {
            console.log('✅ Setting isProcessing from', prevState, 'to true');
            return true;
          });
        });

        newSocket.on('processing_stopped', (data) => {
          console.log('🛑 Processing stopped event received:', data);
          setIsProcessing(prevState => {
            console.log('🛑 Setting isProcessing from', prevState, 'to false');
            return false;
          });
        });

        // Listen for processing errors
        newSocket.on('processing_error', (data) => {
          console.error('❌ Processing error event received:', data);
          setIsProcessing(prevState => {
            console.log('❌ Setting isProcessing from', prevState, 'to false due to error');
            return false;
          });
        });

        // Debug: Add a catch-all listener to see what events are received
        newSocket.onAny((eventName, ...args) => {
          console.log('📥 RECEIVED WebSocket event:', eventName, args);
        });

        setSocket(newSocket);
    }
  }, [isAuthenticated, token, socket]);

  // Ensure socket is disconnected when not authenticated
  useEffect(() => {
    if (!isAuthenticated && socket) {
      console.log('User not authenticated, disconnecting socket...');
      socket.disconnect();
      setSocket(null);
    }
  }, [isAuthenticated, socket]);

  const startProcessing = () => {
    if (socket) {
      if (token) {
        console.log('🚀 Starting video processing...');
        console.log('Current isProcessing state before start:', isProcessing);
        socket.emit('start_processing', { token });
        return true;
      } else {
        console.error('No token available for start_processing');
        return false;
      }
    } else {
      console.error('No socket connection available for start_processing');
      return false;
    }
  };

  const stopProcessing = () => {
    console.log('🛑 stopProcessing function called');
    console.log('🛑 Socket exists:', !!socket);
    console.log('🛑 Socket connected:', socket?.connected);
    console.log('🛑 Socket ID:', socket?.id);
    
    if (socket) {
      console.log('🛑 Token exists:', !!token);
      
      if (token) {
        console.log('🛑 Stopping video processing...');
        console.log('🛑 Current isProcessing state before stop:', isProcessing);
        console.log('🛑 About to emit stop_processing event with token');
        socket.emit('stop_processing', { token });
        console.log('🛑 stop_processing event emitted successfully');
        
        // Add timeout to check if backend responds
        setTimeout(() => {
          console.log('⏰ Checking if stop processing response was received after 3 seconds...');
          console.log('⏰ Current isProcessing state:', isProcessing);
          if (isProcessing) {
            console.warn('⚠️ No response received from backend after 3 seconds');
            console.log('🔄 Force-setting isProcessing to false as fallback');
            // Force set processing to false as fallback
            setIsProcessing(prevState => {
              console.log('🔄 Timeout fallback: Setting isProcessing from', prevState, 'to false');
              return false;
            });
          }
        }, 3000);
        
        return true;
      } else {
        console.error('❌ No token available for stop_processing');
        return false;
      }
    } else {
      console.error('❌ No socket connection available for stop_processing');
      return false;
    }
  };

  // Cleanup socket on unmount
  useEffect(() => {
    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, [socket]);

  // Show loading screen while checking authentication
  if (isLoading) {
    return <LoadingScreen />;
  }

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <Login />;
  }

  // Show dashboard if authenticated
  return (
    <Dashboard 
      data={trafficData}
      history={history}
      isProcessing={isProcessing}
      onStartProcessing={startProcessing}
      onStopProcessing={stopProcessing}
      onLogout={authLogout}
    />
  );
}

export default App;