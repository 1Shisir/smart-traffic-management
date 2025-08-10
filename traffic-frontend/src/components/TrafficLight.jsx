import React, { useState, useEffect } from 'react';

const TrafficLight = ({ state, duration, vehicleCount = 0 }) => {
  const [timeLeft, setTimeLeft] = useState(duration || 0);
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    setTimeLeft(duration || 0);
  }, [duration]);

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => {
        setTimeLeft(timeLeft - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [timeLeft]);

  useEffect(() => {
    // Blink when time is almost up (last 3 seconds)
    if (timeLeft <= 3 && timeLeft > 0) {
      setIsBlinking(true);
      const blinkTimer = setInterval(() => {
        setIsBlinking(prev => !prev);
      }, 500);
      return () => clearInterval(blinkTimer);
    } else {
      setIsBlinking(false);
    }
  }, [timeLeft]);

  const getLightColor = (lightType) => {
    const currentState = state?.toLowerCase() || 'red';
    
    if (lightType === currentState) {
      if (isBlinking && timeLeft <= 3) {
        return isBlinking ? getLightActiveColor(lightType) : getLightInactiveColor(lightType);
      }
      return getLightActiveColor(lightType);
    }
    return getLightInactiveColor(lightType);
  };

  const getLightActiveColor = (lightType) => {
    switch (lightType) {
      case 'red': return 'bg-red-500 shadow-red-500/50';
      case 'yellow': return 'bg-yellow-500 shadow-yellow-500/50';
      case 'green': return 'bg-green-500 shadow-green-500/50';
      default: return 'bg-gray-400';
    }
  };

  const getLightInactiveColor = (lightType) => {
    switch (lightType) {
      case 'red': return 'bg-red-900';
      case 'yellow': return 'bg-yellow-900';
      case 'green': return 'bg-green-900';
      default: return 'bg-gray-400';
    }
  };

  const getTrafficStatus = () => {
    const currentState = state?.toLowerCase() || 'red';
    switch (currentState) {
      case 'red': return { text: 'STOP', color: 'text-red-500', icon: '🛑' };
      case 'yellow': return { text: 'CAUTION', color: 'text-yellow-500', icon: '⚠️' };
      case 'green': return { text: 'GO', color: 'text-green-500', icon: '✅' };
      default: return { text: 'UNKNOWN', color: 'text-gray-500', icon: '❓' };
    }
  };

  const status = getTrafficStatus();

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-lg font-semibold mb-4 text-center text-gray-800">
        Traffic Light Control
      </h3>
      
      <div className="flex flex-col items-center space-y-6">
        {/* Traffic Light Housing */}
        <div className="bg-gray-800 rounded-lg p-4 shadow-2xl">
          <div className="space-y-3">
            {/* Red Light */}
            <div className={`w-16 h-16 rounded-full transition-all duration-300 ${getLightColor('red')} ${
              state?.toLowerCase() === 'red' ? 'shadow-lg' : ''
            }`}></div>
            
            {/* Yellow Light */}
            <div className={`w-16 h-16 rounded-full transition-all duration-300 ${getLightColor('yellow')} ${
              state?.toLowerCase() === 'yellow' ? 'shadow-lg' : ''
            }`}></div>
            
            {/* Green Light */}
            <div className={`w-16 h-16 rounded-full transition-all duration-300 ${getLightColor('green')} ${
              state?.toLowerCase() === 'green' ? 'shadow-lg' : ''
            }`}></div>
          </div>
        </div>

        {/* Status Display */}
        <div className="text-center">
          <div className={`text-2xl font-bold ${status.color} mb-2`}>
            {status.icon} {status.text}
          </div>
          
          {timeLeft > 0 && (
            <div className="bg-gray-100 rounded-lg p-3 min-w-[120px]">
              <div className="text-sm text-gray-600 mb-1">Time Remaining</div>
              <div className={`text-2xl font-mono font-bold ${timeLeft <= 3 ? 'text-red-500' : 'text-gray-800'}`}>
                {timeLeft}s
              </div>
            </div>
          )}
        </div>

        {/* Vehicle Count Display */}
        <div className="bg-blue-50 rounded-lg p-4 w-full">
          <div className="text-center">
            <div className="text-sm text-gray-600 mb-1">Vehicles Detected</div>
            <div className="text-3xl font-bold text-blue-600">
              🚗 {vehicleCount}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {vehicleCount === 0 ? 'No traffic' : 
               vehicleCount < 5 ? 'Light traffic' :
               vehicleCount < 10 ? 'Moderate traffic' : 'Heavy traffic'}
            </div>
          </div>
        </div>

        {/* Traffic Flow Indicator */}
        <div className="w-full bg-gray-200 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-700">Traffic Flow</span>
            <div className="flex space-x-1">
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  className={`w-2 h-6 rounded ${
                    i < Math.min(5, Math.ceil(vehicleCount / 3)) 
                      ? vehicleCount < 5 ? 'bg-green-400' :
                        vehicleCount < 10 ? 'bg-yellow-400' : 'bg-red-400'
                      : 'bg-gray-300'
                  }`}
                ></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrafficLight;