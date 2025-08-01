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
                <td className="border p-2">{row.junction}</td>
                <td className="border p-2">{row.timestamp}</td>
                <td className="border p-2">{row.total}</td>
                <td className="border p-2">{row.car}</td>
                <td className="border p-2">{row.bus}</td>
                <td className="border p-2">{row.truck}</td>
                <td className="border p-2">{row.motorcycle}</td>
                <td className="border p-2">{row.traffic_light}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

export default HistoryTable;