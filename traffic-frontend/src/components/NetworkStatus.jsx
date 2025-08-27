import React, { useState, useEffect } from 'react';
import { FiWifi, FiWifiOff, FiAlertTriangle } from 'react-icons/fi';

const NetworkStatus = ({ onStatusChange = () => {} }) => {
  const [isOnline, setIsOnline] = useState(true);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online', 'offline', 'checking'
  const [lastChecked, setLastChecked] = useState(null);

  // Check backend connectivity
  const checkBackendStatus = async () => {
    console.log('🔍 NetworkStatus: Checking backend connectivity...');
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      const response = await fetch('/health', {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Cache-Control': 'no-cache',
        },
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        console.log('✅ NetworkStatus: Backend is online');
        setBackendStatus('online');
        setLastChecked(new Date());
        onStatusChange({ backend: true, network: navigator.onLine });
        return true;
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.log('❌ NetworkStatus: Backend connectivity check failed:', error.message);
      setBackendStatus('offline');
      setLastChecked(new Date());
      onStatusChange({ backend: false, network: navigator.onLine });
      return false;
    }
  };

  // Monitor network connectivity
  useEffect(() => {
    console.log('🚀 NetworkStatus: Component initialized');
    
    const handleOnline = () => {
      console.log('🌐 NetworkStatus: Network came online');
      setIsOnline(true);
      checkBackendStatus(); // Re-check backend when network comes back
    };

    const handleOffline = () => {
      console.log('🌐 NetworkStatus: Network went offline');
      setIsOnline(false);
      setBackendStatus('offline');
      onStatusChange({ backend: false, network: false });
    };

    // Set initial status
    setIsOnline(navigator.onLine);
    console.log('🌐 NetworkStatus: Initial network status:', navigator.onLine);

    // Add event listeners
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Initial backend check
    console.log('🔍 NetworkStatus: Starting initial backend check...');
    checkBackendStatus();

    // Set up periodic backend health checks (every 30 seconds)
    const healthCheckInterval = setInterval(() => {
      if (navigator.onLine) {
        console.log('🔍 NetworkStatus: Periodic backend check...');
        checkBackendStatus();
      }
    }, 30000);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      clearInterval(healthCheckInterval);
    };
  }, []);

  // Don't show anything if everything is working fine, or if we're still doing the initial check
  if (isOnline && backendStatus === 'online') {
    return null;
  }

  // Don't show status during initial check unless it takes too long
  if (backendStatus === 'checking' && !lastChecked) {
    return null;
  }

  const getStatusInfo = () => {
    if (!isOnline) {
      return {
        icon: FiWifiOff,
        title: 'No Internet Connection',
        message: 'Please check your network connection and try again.',
        color: 'bg-red-500',
        textColor: 'text-red-100'
      };
    } else if (backendStatus === 'offline') {
      return {
        icon: FiAlertTriangle,
        title: 'Server Unavailable',
        message: 'Unable to connect to the traffic management server. Please ensure the backend is running.',
        color: 'bg-orange-500',
        textColor: 'text-orange-100'
      };
    } else if (backendStatus === 'checking') {
      return {
        icon: FiWifi,
        title: 'Checking Connection...',
        message: 'Verifying server connectivity...',
        color: 'bg-blue-500',
        textColor: 'text-blue-100'
      };
    }
  };

  const statusInfo = getStatusInfo();
  const IconComponent = statusInfo.icon;

  const formatLastChecked = () => {
    if (!lastChecked) return '';
    return `Last checked: ${lastChecked.toLocaleTimeString()}`;
  };

  const handleRetry = async () => {
    setBackendStatus('checking');
    const success = await checkBackendStatus();
    if (!success) {
      // Could add toast notification here if needed
      console.log('Retry failed - backend still unavailable');
    }
  };

  return (
    <div className={`fixed top-0 left-0 right-0 z-50 ${statusInfo.color} ${statusInfo.textColor} shadow-lg`}>
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <IconComponent className="h-5 w-5 animate-pulse" />
            <div>
              <div className="font-semibold text-sm">{statusInfo.title}</div>
              <div className="text-xs opacity-90">{statusInfo.message}</div>
              {lastChecked && (
                <div className="text-xs opacity-75 mt-1">{formatLastChecked()}</div>
              )}
            </div>
          </div>
          
          {backendStatus === 'offline' && (
            <div className="flex items-center space-x-2">
              <button
                onClick={handleRetry}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1 rounded text-xs text-black font-medium transition-colors"
              >
                Retry
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default NetworkStatus;
