import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import axios from 'axios';
import Dashboard from './components/Dashboard';
import './App.css'; // Assuming you have some global styles

const socket = io('http://localhost:5000', { transports: ['websocket'] });

const App = () => {
  const [data, setData] = useState({
    junction: 'Loading...',
    count: 0,
    car: 0,
    bus: 0,
    truck: 0,
    motorcycle: 0,
    time: 'Loading...',
    traffic_light: 'Green',
    light_duration: 20,
  });
  const [history, setHistory] = useState([]);

  useEffect(() => {
    // Socket.IO listeners
    socket.on('update', (newData) => {
      setData((prev) => ({ ...prev, ...newData }));
    });
    socket.on('traffic_light', ({ state, duration }) => {
      setData((prev) => ({ ...prev, traffic_light: state, light_duration: duration }));
    });

    // Fetch historical data
    axios.get('http://localhost:5000/api/data')
      .then((response) => setHistory(response.data))
      .catch((error) => console.error('Error fetching history:', error));

    return () => {
      socket.off('update');
      socket.off('traffic_light');
      socket.disconnect();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      <Dashboard data={data} history={history} />
    </div>
  );
};

export default App;