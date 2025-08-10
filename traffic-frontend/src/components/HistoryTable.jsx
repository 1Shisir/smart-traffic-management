import React from 'react';

const HistoryTable = ({ history }) => {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-gray-200">
            <th className="border p-2">Junction</th>
            <th className="border p-2">Timestamp</th>
            <th className="border p-2">Total</th>
            <th className="border p-2">Car</th>
            <th className="border p-2">Bus</th>
            <th className="border p-2">Truck</th>
            <th className="border p-2">Motorcycle</th>
            <th className="border p-2">Traffic Light</th>
          </tr>
        </thead>
        <tbody>
          {history.length === 0 ? (
            <tr>
              <td colSpan="8" className="border p-2 text-center">Loading...</td>
            </tr>
          ) : (
            history.map((row, index) => (
              <tr key={index}>
                <td className="border p-2">{row.junction || 'Main St & 1st Ave'}</td>
                <td className="border p-2">{row.time || row.timestamp || new Date(row.created_at).toLocaleString()}</td>
                <td className="border p-2">{row.total_count || row.total || 0}</td>
                <td className="border p-2">{row.car_count || row.car || 0}</td>
                <td className="border p-2">{row.bus_count || row.bus || 0}</td>
                <td className="border p-2">{row.truck_count || row.truck || 0}</td>
                <td className="border p-2">{row.motorcycle_count || row.motorcycle || 0}</td>
                <td className="border p-2">{row.traffic_light || 'green'}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

export default HistoryTable;