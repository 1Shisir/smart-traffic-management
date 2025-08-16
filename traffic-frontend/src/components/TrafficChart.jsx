import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

const TrafficChart = ({ data, chartType = 'bar', showVehicleTypes = true }) => {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        labels: [],
        datasets: []
      };
    }

    // Sort data by timestamp for proper chronological display
    const sortedData = [...data].sort((a, b) => {
      const dateA = new Date(a.timestamp || a.created_at || 0);
      const dateB = new Date(b.timestamp || b.created_at || 0);
      return dateA - dateB;
    });

    // Extract labels (timestamps) - show only time for recent data
    const labels = sortedData.map(item => {
      const timestamp = item.timestamp || item.created_at;
      if (timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit',
          second: '2-digit'
        });
      }
      return 'Unknown';
    });

    if (showVehicleTypes) {
      // Show breakdown by vehicle type
      return {
        labels,
        datasets: [
          {
            label: 'Cars',
            data: sortedData.map(item => item.car_count || item.car || 0),
            backgroundColor: 'rgba(59, 130, 246, 0.8)', // Blue
            borderColor: 'rgba(59, 130, 246, 1)',
            borderWidth: chartType === 'line' ? 2 : 1,
            fill: chartType === 'line' ? false : true,
          },
          {
            label: 'Buses',
            data: sortedData.map(item => item.bus_count || item.bus || 0),
            backgroundColor: 'rgba(16, 185, 129, 0.8)', // Green
            borderColor: 'rgba(16, 185, 129, 1)',
            borderWidth: chartType === 'line' ? 2 : 1,
            fill: chartType === 'line' ? false : true,
          },
          {
            label: 'Trucks',
            data: sortedData.map(item => item.truck_count || item.truck || 0),
            backgroundColor: 'rgba(245, 158, 11, 0.8)', // Orange
            borderColor: 'rgba(245, 158, 11, 1)',
            borderWidth: chartType === 'line' ? 2 : 1,
            fill: chartType === 'line' ? false : true,
          },
          {
            label: 'Motorcycles',
            data: sortedData.map(item => item.motorcycle_count || item.motorcycle || 0),
            backgroundColor: 'rgba(139, 69, 19, 0.8)', // Brown
            borderColor: 'rgba(139, 69, 19, 1)',
            borderWidth: chartType === 'line' ? 2 : 1,
            fill: chartType === 'line' ? false : true,
          }
        ]
      };
    } else {
      // Show total vehicle count only
      return {
        labels,
        datasets: [
          {
            label: 'Total Vehicles',
            data: sortedData.map(item => item.total_count || item.total || 0),
            backgroundColor: 'rgba(99, 102, 241, 0.8)', // Indigo
            borderColor: 'rgba(99, 102, 241, 1)',
            borderWidth: chartType === 'line' ? 3 : 1,
            fill: chartType === 'line' ? false : true,
            tension: chartType === 'line' ? 0.4 : 0, // Smooth line curves
          }
        ]
      };
    }
  }, [data, showVehicleTypes, chartType]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
        }
      },
      title: {
        display: true,
        text: showVehicleTypes ? 'Vehicle Count by Type Over Time' : 'Total Vehicle Count Over Time',
        font: {
          size: 16,
          weight: 'bold'
        },
        padding: 20
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: ${value} vehicle${value !== 1 ? 's' : ''}`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      },
      y: {
        title: {
          display: true,
          text: 'Number of Vehicles',
          font: {
            weight: 'bold'
          }
        },
        beginAtZero: true,
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          stepSize: 1,
          callback: function(value) {
            return Number.isInteger(value) ? value : '';
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    elements: {
      point: {
        radius: chartType === 'line' ? 4 : 0,
        hoverRadius: chartType === 'line' ? 6 : 0,
        backgroundColor: 'white',
        borderWidth: 2,
      }
    }
  };

  const ChartComponent = chartType === 'line' ? Line : Bar;

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center">
          <div className="text-gray-400 text-4xl mb-2">📊</div>
          <p className="text-gray-500 text-lg font-medium">No Data Available</p>
          <p className="text-gray-400 text-sm">Start processing to see traffic analytics</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md border">
      <div className="p-4 border-b">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-800">Traffic Analytics</h3>
          <div className="text-sm text-gray-600">
            {data.length} record{data.length !== 1 ? 's' : ''}
          </div>
        </div>
      </div>
      <div className="p-4">
        <div className="h-64 md:h-80">
          <ChartComponent data={chartData} options={chartOptions} />
        </div>
      </div>
    </div>
  );
};

export default TrafficChart;
