import React from 'react';
import VehicleChart from './VehicleChart.jsx';
import TrafficLight from './TrafficLight.jsx';
import VideoPreview from './VideoPreview.jsx';
import HistoryTable from './HistoryTable.jsx';

const Dashboard = ({ data, history }) => {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6 text-center">Traffic Monitoring Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Real-Time Data</h2>
          <p className="mb-2"><strong>Junction:</strong> {data.junction}</p>
          <p className="mb-2"><strong>Timestamp:</strong> {data.time}</p>
          <p className="mb-2"><strong>Total Vehicles:</strong> {data.count}</p>
          <p className="mb-2"><strong>Cars:</strong> {data.car}</p>
          <p className="mb-2"><strong>Buses:</strong> {data.bus}</p>
          <p className="mb-2"><strong>Trucks:</strong> {data.truck}</p>
          <p className="mb-2"><strong>Motorcycles:</strong> {data.motorcycle}</p>
          <TrafficLight state={data.traffic_light} duration={data.light_duration} />
        </div>
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Vehicle Counts</h2>
          <VehicleChart data={data} />
        </div>
      </div>
      <div className="bg-white p-6 rounded-lg shadow-lg mt-6">
        <h2 className="text-xl font-semibold mb-4">Video Preview</h2>
        <VideoPreview />
      </div>
      <div className="bg-white p-6 rounded-lg shadow-lg mt-6">
        <h2 className="text-xl font-semibold mb-4">Historical Data</h2>
        <HistoryTable history={history} />
      </div>
    </div>
  );
};

export default Dashboard;