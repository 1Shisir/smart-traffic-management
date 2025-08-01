import React, { useState, useEffect } from 'react';

const VideoPreview = () => {
  const [src, setSrc] = useState('/video-preview');

  useEffect(() => {
    const interval = setInterval(() => {
      setSrc(`/video-preview?${new Date().getTime()}`);
    }, 1000); // Refresh every second
    return () => clearInterval(interval);
  }, []);

  return (
    <img
      src={src}
      alt="Video Preview"
      className="max-w-full rounded-lg"
      onError={(e) => (e.target.src = '/video-preview')}
    />
  );
};

export default VideoPreview;