import React from 'react';

const TrafficLight = ({ state, duration }) => {
  return (
    <div className="text-center">
      <p className="mb-2"><strong>Traffic Light:</strong> {state} ({duration}s)</p>
      <div
        className="w-12 h-12 rounded-full mx-auto"
        style={{
          backgroundColor: state === 'Green' ? 'green' : state === 'Yellow' ? 'yellow' : 'red',
        }}
      ></div>
    </div>
  );
};

export default TrafficLight;