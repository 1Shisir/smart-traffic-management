import React, { useState, useEffect } from 'react';
import TrafficLight from './TrafficLight.jsx';
import VideoPreview from './VideoPreview.jsx';
import TrafficChart from './TrafficChart.jsx';
import ChartControls from './ChartControls.jsx';
import { FaCar, FaBus, FaTruck, FaMotorcycle, FaEye, FaClock } from 'react-icons/fa';

const Dashboard = ({ data, history, isProcessing, isPolling, onStartProcessing, onStopProcessing, onRefreshChart, onLogout }) => {
  // Default data structure to prevent undefined errors
  const defaultData = {
    count: 0,
    car: 0,
    bus: 0,
    truck: 0,
    motorcycle: 0,
    traffic_light: 'red',
    light_duration: 30,
    junction: 'Main St & 1st Ave',
    time: new Date().toLocaleTimeString()
  };

  // Use provided data or default data
  const trafficData = data || defaultData;

  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [lastClickTime, setLastClickTime] = useState(0);
  
  // Chart settings state
  const [chartType, setChartType] = useState('bar'); // 'bar' or 'line'
  const [viewMode, setViewMode] = useState('breakdown'); // 'total' or 'breakdown'
  
  // Clear button states when processing state changes
  useEffect(() => {
    if (isProcessing) {
      // Processing started - clear the starting state
      setIsStarting(false);
    } else {
      // Processing stopped - clear the stopping state  
      setIsStopping(false);
    }
  }, [isProcessing]);
  
  const handleStartClick = async () => {
    // Debouncing: prevent clicks within 1 second of each other
    const now = Date.now();
    if (now - lastClickTime < 1000) {
      console.log('🔘 Dashboard: Click ignored - too soon after last click');
      return;
    }
    setLastClickTime(now);
    
    if (isStarting || isProcessing) return;
    
    setIsStarting(true);
    console.log('🔘 Dashboard: Start button clicked');
    
    try {
      const success = await onStartProcessing();
      if (!success) {
        setIsStarting(false);
        console.log('❌ Dashboard: Start processing failed');
      } else {
        console.log('✅ Dashboard: Start processing initiated');
        // Shorter timeout since polling will update isProcessing when backend confirms
        setTimeout(() => {
          setIsStarting(false);
        }, 1500);
      }
    } catch (error) {
      console.error('❌ Dashboard: Start processing error:', error);
      setIsStarting(false);
    }
  };

  const handleStopClick = async () => {
    // Debouncing: prevent clicks within 1 second of each other
    const now = Date.now();
    if (now - lastClickTime < 1000) {
      console.log('🔘 Dashboard: Click ignored - too soon after last click');
      return;
    }
    setLastClickTime(now);
    
    if (isStopping || !isProcessing) return;
    
    setIsStopping(true);
    console.log('🔘 Dashboard: Stop button clicked');
    
    try {
      const success = await onStopProcessing();
      if (!success) {
        setIsStopping(false);
        console.log('❌ Dashboard: Stop processing failed');
      } else {
        console.log('✅ Dashboard: Stop processing initiated');
        // Shorter timeout since polling will detect when processing actually stops
        setTimeout(() => {
          setIsStopping(false);
        }, 1500);
      }
    } catch (error) {
      console.error('❌ Dashboard: Stop processing error:', error);
      setIsStopping(false);
    }
  };
  
  const vehicleIcons = {
    car: <FaCar className="text-blue-500" />,
    bus: <FaBus className="text-green-500" />,
    truck: <FaTruck className="text-orange-500" />,
    motorcycle: <FaMotorcycle className="text-purple-500" />
  };

  const VehicleCountCard = ({ type, count, icon }) => (
    <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-blue-500">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 capitalize">{type}s</p>
          <p className="text-2xl font-bold text-gray-900">{count}</p>
        </div>
        <div className="text-2xl">{icon}</div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center mr-3">
                <FaEye className="text-white text-sm" />
              </div>
              <h1 className="text-xl font-semibold text-gray-900">Traffic Monitoring Dashboard</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Processing Controls */}
              <div className="flex space-x-2">
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    handleStartClick();
                  }}
                  disabled={isProcessing || isStarting}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                    (isProcessing || isStarting)
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                      : 'bg-green-500 hover:bg-green-600 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                  }`}
                >
                  {isStarting ? '🔄 Starting...' : isProcessing ? '⏸️ Processing...' : '▶️ Start Processing'}
                </button>
                
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    handleStopClick();
                  }}
                  disabled={!isProcessing || isStopping}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                    (!isProcessing || isStopping)
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                      : 'bg-red-500 hover:bg-red-600 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                  }`}
                >
                  {isStopping ? '🔄 Stopping...' : '⏹️ Stop Processing'}
                </button>
              </div>

              {/* Status Indicator */}
              <div className="flex items-center space-x-4">
                <div className="flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${isProcessing ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
                  <span className="text-sm text-gray-600">
                    {isProcessing ? 'Active' : 'Inactive'}
                  </span>
                </div>
                
                {/* Polling Status */}
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${isPolling ? 'bg-blue-500 animate-ping' : 'bg-gray-300'}`}></div>
                  <span className="text-xs text-gray-500">
                    {isPolling ? 'Real-time' : 'Static'}
                  </span>
                </div>
              </div>

              {/* Logout Button */}
              <button
                onClick={onLogout}
                className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-all duration-200 hover:shadow-lg transform hover:scale-105"
              >
                🚪 Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Status Banner */}
        <div className={`mb-8 p-4 rounded-lg ${
          isProcessing ? 'bg-green-50 border border-green-200' : 'bg-yellow-50 border border-yellow-200'
        }`}>
          <div className="flex items-center">
            <div className="text-2xl mr-3">
              {isProcessing ? '🟢' : '🟡'}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">
                {isProcessing ? 'System Active' : 'System Standby'}
              </h3>
              <p className="text-sm text-gray-600">
                {isProcessing 
                  ? 'Real-time traffic monitoring is active and collecting data' 
                  : 'Click "Start Processing" to begin monitoring traffic'}
              </p>
            </div>
          </div>
        </div>

        {/* Real-time Stats Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Left Column - Vehicle Counts */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <h2 className="text-lg font-semibold mb-6 text-gray-800 flex items-center">
                <FaCar className="mr-2 text-blue-500" />
                Vehicle Detection
              </h2>
              
              {/* Total Count */}
              <div className="bg-gradient-to-r from-blue-400 to-blue-50 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <p className="text-sm opacity-90">Total Vehicles</p>
                  <p className="text-3xl font-bold">{trafficData.count}</p>
                </div>
              </div>

              {/* Individual Counts */}
              <div className="grid grid-cols-2 gap-3">
                <VehicleCountCard type="car" count={trafficData.car} icon={vehicleIcons.car} />
                <VehicleCountCard type="bus" count={trafficData.bus} icon={vehicleIcons.bus} />
                <VehicleCountCard type="truck" count={trafficData.truck} icon={vehicleIcons.truck} />
                <VehicleCountCard type="motorcycle" count={trafficData.motorcycle} icon={vehicleIcons.motorcycle} />
              </div>

              {/* Junction & Time Info */}
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between text-sm">
                  <div>
                    <span className="text-gray-600">Junction:</span>
                    <span className="ml-2 font-medium">{trafficData.junction}</span>
                  </div>
                  <div className="flex items-center">
                    <FaClock className="text-gray-400 mr-1" />
                    <span className="text-gray-600">{trafficData.time}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Middle Column - Traffic Light */}
          <div className="lg:col-span-1">
            <TrafficLight 
              state={trafficData.traffic_light} 
              duration={trafficData.light_duration}
              vehicleCount={trafficData.count}
              isProcessing={isProcessing}
            />
          </div>

          {/* Right Column - Video Preview */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-lg font-semibold mb-4 text-gray-800 flex items-center">
                <FaEye className="mr-2 text-indigo-500" />
                Live Video Feed
              </h2>
              <VideoPreview />
            </div>
          </div>
        </div>

        {/* Traffic Analytics Chart */}
        <div className="mb-8">
          <ChartControls 
            chartType={chartType}
            viewMode={viewMode}
            onChartTypeChange={setChartType}
            onViewModeChange={setViewMode}
            onRefresh={onRefreshChart}
          />
          
          <TrafficChart 
            data={history}
            chartType={chartType}
            showVehicleTypes={viewMode === 'breakdown'}
          />
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
