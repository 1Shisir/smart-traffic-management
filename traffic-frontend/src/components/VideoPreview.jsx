import React, { useState, useEffect, useRef } from 'react';

const VideoPreview = () => {
  const [videoSrc, setVideoSrc] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const videoRef = useRef(null);

  const setupVideoStream = async () => {
    try {
      // Create a video URL without authentication (public access)
      const videoUrl = 'http://localhost:5000/api/video-stream';
      setVideoSrc(videoUrl);
      setError(false);
      setLoading(false);
    } catch (err) {
      console.error('Error setting up video stream:', err);
      setError(true);
      setLoading(false);
    }
  };

  useEffect(() => {
    // Setup video stream
    setupVideoStream();
  }, []);

  const handleVideoError = () => {
    console.error('Video failed to load');
    setError(true);
    setLoading(false);
  };

  const handleVideoLoad = () => {
    setLoading(false);
    setError(false);
    // Auto-play and loop the video
    if (videoRef.current) {
      videoRef.current.play();
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center animate-pulse">
        <div className="text-gray-400 mb-2 text-2xl">🔄</div>
        <p className="text-gray-600">Loading video preview...</p>
      </div>
    );
  }

  if (error || !videoSrc) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center">
        <div className="text-gray-500 mb-2 text-2xl">📹</div>
        <p className="text-gray-600 font-medium">Video preview unavailable</p>
        <p className="text-sm text-gray-500 mt-1">Check server connection or video file availability</p>
        <button 
          onClick={setupVideoStream}
          className="mt-3 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <video
        ref={videoRef}
        src={videoSrc}
        className="w-full h-auto rounded-lg shadow-lg"
        controls
        autoPlay
        loop
        muted
        onError={handleVideoError}
        onLoadedData={handleVideoLoad}
        onCanPlay={handleVideoLoad}
      />
      <div className="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-xs">
        � SAMPLE
      </div>
      {loading && (
        <div className="absolute inset-0 bg-black bg-opacity-50 rounded-lg flex items-center justify-center">
          <div className="text-white text-lg">Loading video...</div>
        </div>
      )}
    </div>
  );
};

export default VideoPreview;