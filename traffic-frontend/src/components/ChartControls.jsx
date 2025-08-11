import React, { useState } from 'react';
import { FaChartBar, FaChartLine, FaCar, FaList, FaSync } from 'react-icons/fa';

const ChartControls = ({ onChartTypeChange, onViewModeChange, chartType, viewMode, onRefresh }) => {
  return (
    <div className="bg-white rounded-lg shadow-md border p-4 mb-4">
      <div className="flex flex-wrap gap-4 items-center justify-between">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold text-gray-800">Chart Settings</h3>
        </div>
        
        <div className="flex flex-wrap gap-2">
          {/* Chart Type Toggle */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => onChartTypeChange('bar')}
              className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                chartType === 'bar'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
              }`}
            >
              <FaChartBar className="mr-2" />
              Bar Chart
            </button>
            <button
              onClick={() => onChartTypeChange('line')}
              className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                chartType === 'line'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
              }`}
            >
              <FaChartLine className="mr-2" />
              Line Chart
            </button>
          </div>

          {/* View Mode Toggle */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => onViewModeChange('total')}
              className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'total'
                  ? 'bg-green-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
              }`}
            >
              <FaList className="mr-2" />
              Total Count
            </button>
            <button
              onClick={() => onViewModeChange('breakdown')}
              className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'breakdown'
                  ? 'bg-green-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
              }`}
            >
              <FaCar className="mr-2" />
              By Vehicle Type
            </button>
          </div>
          
          {/* Refresh Button */}
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="flex items-center px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors shadow-sm hover:shadow-md"
              title="Refresh chart data from database"
            >
              <FaSync className="mr-2" />
              Refresh Data
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChartControls;
