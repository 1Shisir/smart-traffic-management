import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import io from 'socket.io-client';
import { useAuth } from './context/AuthContext.jsx';
import Login from './pages/Login.jsx';
import VideoDetection from './pages/VideoDetection.jsx';
import Dashboard from './components/Dashboard.jsx';
import LoadingScreen from './components/LoadingScreen.jsx';
import NetworkStatus from './components/NetworkStatus.jsx';
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
  const [pollingInterval, setPollingInterval] = useState(null);
  const [networkStatus, setNetworkStatus] = useState({ backend: true, network: true });

  // Load initial data when authenticated and connect socket
  useEffect(() => {
    if (isAuthenticated && token) {
      loadInitialData();
    }
    // Load chart data regardless of authentication (for public access)
    loadChartData();
    // Connect socket immediately for real-time updates
    connectSocket();
    // Note: Don't start polling here - only start when processing begins
    
    // Cleanup on unmount
    return () => {
      stopPolling();
    };
  }, [isAuthenticated, token]);

  // Additional effect to check initial status on app load (single check, not continuous polling)
  useEffect(() => {
    // Check current processing status on app load
    const checkInitialStatus = async () => {
      try {
        const response = await fetch('/api/realtime-status');
        if (response.ok) {
          const data = await response.json();
          console.log('🔄 Initial status check:', data);
          const backendProcessingActive = data.processing_active === true;
          setIsProcessing(backendProcessingActive);
          
          // If processing is already active (e.g., after page refresh during processing), start polling
          if (backendProcessingActive) {
            console.log('🔄 Processing detected on load, starting polling...');
            startPolling();
          }
        }
      } catch (error) {
        console.log('🔄 Initial status check failed:', error);
        // Default to false if can't reach server
        setIsProcessing(false);
      }
    };
    
    checkInitialStatus();
  }, []);

  const loadInitialData = async () => {
    try {
      const statusResponse = await fetch('/api/current-status', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        
        // Update current traffic data
        if (statusData.current_data) {
          setTrafficData(prev => ({
            ...prev,
            ...statusData.current_data
          }));
        }

        // Set processing state
        if (statusData.system_status) {
          setIsProcessing(statusData.system_status.processing_active || false);
        }
      }

      await loadChartData();
      
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  };

  // Handle network status changes
  const handleNetworkStatusChange = (status) => {
    setNetworkStatus(status);
    
    // If backend comes back online, reload data
    if (status.backend && isAuthenticated) {
      loadInitialData();
    }
  };

  const loadChartData = async () => {
    try {
      const response = await fetch('/api/data?limit=50', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const chartData = await response.json();
        setHistory(chartData);
      }
    } catch (error) {
      // Silently handle error
    }
  };

  const generateSampleData = async () => {
    try {
      const response = await fetch('/api/generate-sample-data', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ count: 20 })
      });

      if (response.ok) {
        setTimeout(() => loadChartData(), 1000);
      }
    } catch (error) {
      // Silently handle error
    }
  };

  // 🔄 HTTP Polling for real-time updates (more reliable than Socket.IO)
  const startPolling = () => {
    // Don't start polling if it's already running
    if (pollingInterval) {
      console.log('🔄 Polling is already active, skipping start request');
      return;
    }

    const poll = async () => {
      try {
        console.log('🔄 Polling for real-time updates...');
        const response = await fetch('/api/realtime-status');
        
        if (response.ok) {
          const realtimeData = await response.json();
          console.log('🔄 📥 Received polling data:', realtimeData);
          
          // Always update processing status based on backend data
          const backendProcessingActive = realtimeData.processing_active === true;
          
          if (backendProcessingActive) {
            // Update traffic data when processing is active
            setTrafficData(prevData => ({
              ...prevData,
              junction: realtimeData.junction || prevData.junction,
              count: realtimeData.total_vehicles || 0,
              car: realtimeData.car_count || 0,
              bus: realtimeData.bus_count || 0,
              truck: realtimeData.truck_count || 0,
              motorcycle: realtimeData.motorcycle_count || 0,
              traffic_light: realtimeData.traffic_light_state || 'red',
              light_duration: realtimeData.traffic_light_duration || 30,
              time: new Date(realtimeData.timestamp).toLocaleTimeString() || new Date().toLocaleTimeString()
            }));
            
            // Update processing status to active
            setIsProcessing(true);
            console.log('🔄 ✅ Updated traffic data from polling - processing active');
          } else {
            // Processing is not active or has stopped on the backend
            setIsProcessing(false);
            console.log('🔄 🛑 Processing stopped or inactive detected via polling');
            
            // Stop polling since processing is no longer active
            stopPolling();
            console.log('🔄 Stopped polling because backend processing is inactive');
          }
        }
      } catch (error) {
        console.error('🔄 ❌ Polling error:', error);
      }
    };

    // Start polling every 1 second for real-time updates
    const intervalId = setInterval(poll, 1000);
    setPollingInterval(intervalId);
    
    // Also poll immediately
    poll();
    
    console.log('🔄 ✅ Started HTTP polling for real-time updates (1-second intervals)');
  };

  const stopPolling = () => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
      console.log('🔄 🛑 Stopped HTTP polling - no longer checking for real-time updates');
    }
  };

  // Socket connection management - simplified without authentication
  const connectSocket = () => {
    if (socket) return socket; // Already connected
    
    console.log('🔌 Connecting socket without authentication...');
    const newSocket = io('/', {
      transports: ['websocket', 'polling'],
      upgrade: true,
      rememberUpgrade: true,
      forceNew: false, // Changed to false to reuse connections
      reconnection: true, // Allow auto-reconnect
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      timeout: 20000
    });

    newSocket.on('connect', () => {
      console.log('🔌 Socket connected successfully (no auth)');
      console.log('🔌 Socket ID:', newSocket.id);
      console.log('🔌 Socket connected:', newSocket.connected);
      // Test connection immediately
      newSocket.emit('test_connection', {});
    });

    newSocket.on('test_response', (data) => {
      console.log('✅ Socket test response:', data);
    });

    newSocket.on('disconnect', (reason) => {
      console.log('🔌 Socket disconnected:', reason);
    });

    newSocket.on('connect_error', (error) => {
      console.log('❌ Socket connection error:', error);
    });

    newSocket.on('error', (error) => {
      console.log('❌ Socket error:', error);
    });

    // Add catch-all event listener to debug what events are received
    newSocket.onAny((eventName, ...args) => {
      console.log('🎯 Received Socket.IO event:', eventName, args);
    });

    // Listen for real-time traffic data updates
    newSocket.on('update', (data) => {
      console.log('📊 Received update event:', data);
      setTrafficData(prev => ({
        ...prev,
        junction: data.junction || prev.junction,
        time: data.time || prev.time,
        count: data.count || 0,
        car: data.car || 0,
        bus: data.bus || 0,
        truck: data.truck || 0,
        motorcycle: data.motorcycle || 0,
        traffic_light: data.traffic_light || prev.traffic_light,
        light_duration: data.light_duration || prev.light_duration
      }));
    });

    // Listen for traffic light updates
    newSocket.on('traffic_light', (data) => {
      console.log('🚦 Received traffic_light event:', data);
      setTrafficData(prev => ({
        ...prev,
        traffic_light: data.state || prev.traffic_light,
        light_duration: data.duration || prev.light_duration
      }));
    });

    // Listen for processing state changes
    newSocket.on('processing_started', (data) => {
      console.log('🚀 Received processing_started event:', data);
      setIsProcessing(true);
    });

    newSocket.on('processing_stopped', (data) => {
      console.log('✅ Received processing_stopped event:', data);
      setIsProcessing(false);
      // Keep socket connected for continuous updates
    });

    newSocket.on('processing_stopped_global', (data) => {
      console.log('🌍 Received processing_stopped_global event:', data);
      setIsProcessing(false);
      // Keep socket connected for continuous updates
    });

    newSocket.on('processing_error', (data) => {
      console.log('❌ Received processing_error event:', data);
      setIsProcessing(false);
    });

    newSocket.on('connect_error', (error) => {
      console.log('❌ Socket connection error:', error);
    });

    setSocket(newSocket);
    return newSocket;
  };

  const disconnectSocket = () => {
    if (socket) {
      console.log('🔌 Disconnecting socket...');
      socket.disconnect();
      setSocket(null);
    }
  };

  // Ensure socket is disconnected when not authenticated
  useEffect(() => {
    if (!isAuthenticated && socket) {
      disconnectSocket();
    }
  }, [isAuthenticated, socket]);

  // Periodic chart data refresh from database
  useEffect(() => {
    if (!isAuthenticated || !token) return;

    loadChartData();

    const refreshInterval = isProcessing ? 30000 : 120000;
    const intervalId = setInterval(() => {
      loadChartData();
    }, refreshInterval);

    return () => {
      clearInterval(intervalId);
    };
  }, [isAuthenticated, token, isProcessing]);

  const startProcessing = () => {
    console.log('🚀 Frontend: Starting processing without auth - connecting socket first');
    
    // Immediately update UI state to show processing is starting
    setIsProcessing(true);
    
    // Start polling for real-time updates now that processing is beginning
    startPolling();
    console.log('🔄 Started polling because processing is beginning');
    
    // Connect socket first, then start processing
    const socketConnection = connectSocket();
    
    // Wait a moment for connection, then start processing
    setTimeout(() => {
      if (socketConnection) {
        console.log('🚀 Frontend: Sending start_processing event (no auth)');
        socketConnection.emit('start_processing', {
          junction: 'main_junction'
        });
      }
    }, 1000);
    
    return true;
  };

  const stopProcessing = () => {
    console.log('🛑 Frontend: Sending stop_processing event (no auth)');
    
    // Immediately update UI to show processing is stopping
    setIsProcessing(false);
    
    // Stop polling since processing is ending
    stopPolling();
    console.log('🔄 Stopped polling because processing is ending');
    
    if (socket) {
      socket.emit('stop_processing', {});
      return true;
    }
    console.log('❌ Frontend: No socket available');
    return false;
  };

  // Cleanup socket and polling on unmount
  useEffect(() => {
    return () => {
      disconnectSocket();
      stopPolling();
    };
  }, []);

  // Show loading screen while checking authentication
  if (isLoading) {
    return (
      <Router>
        <NetworkStatus onStatusChange={handleNetworkStatusChange} />
        <LoadingScreen />
      </Router>
    );
  }

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return (
      <Router>
        <NetworkStatus onStatusChange={handleNetworkStatusChange} />
        <Login />
      </Router>
    );
  }

  const logout = () => {
    authLogout();
    disconnectSocket();
  };

  const refreshChart = () => {
    // Placeholder function for chart refresh
    // This could be implemented to refresh historical data
  };

  // Authenticated user sees the dashboard
  return (
    <Router>
      <NetworkStatus onStatusChange={handleNetworkStatusChange} />
      <Routes>
        <Route 
          path="/" 
          element={
            <Dashboard
              data={trafficData}
              history={history}
              isProcessing={isProcessing}
              isPolling={pollingInterval !== null}
              networkStatus={networkStatus}
              onStartProcessing={startProcessing}
              onStopProcessing={stopProcessing}
              onRefreshChart={refreshChart}
              onLogout={logout}
            />
          } 
        />
        <Route path="/video-detection" element={<VideoDetection />} />
      </Routes>
    </Router>
  );
}

export default App;
