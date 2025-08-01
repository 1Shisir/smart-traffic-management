import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const VehicleChart = ({ data }) => {
  const chartData = {
    labels: ['Car', 'Bus', 'Truck', 'Motorcycle'],
    datasets: [
      {
        label: 'Vehicles',
        data: [data.car, data.bus, data.truck, data.motorcycle],
        backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0'],
      },
    ],
  };

  const options = {
    responsive: true,
    scales: {
      y: { beginAtZero: true },
    },
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Vehicle Counts' },
    },
  };

  return <Bar data={chartData} options={options} />;
};

export default VehicleChart;