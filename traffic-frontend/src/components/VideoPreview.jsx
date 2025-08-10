import React, { useState, useEffect } from 'react';

const VideoPreview = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchFrame = async () => {
    // Get token from multiple possible locations
    const token = localStorage.getItem('authToken') || 
                 sessionStorage.getItem('authToken') || 
                 getCookie('authToken');
    
    if (!token) {
      setError(true);
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/video-feed', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setImageSrc(prev => {
          if (prev) URL.revokeObjectURL(prev); // Clean up previous URL
          return imageUrl;
        });
        setError(false);
      } else {
        console.log('No video frame available');
        setError(true);
      }
    } catch (err) {
      console.error('Error fetching video frame:', err);
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to get cookie
  const getCookie = (name) => {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
  };

  useEffect(() => {
    // Initial fetch
    fetchFrame();

    // Refresh every 2 seconds
    const interval = setInterval(fetchFrame, 2000);
    
    return () => {
      clearInterval(interval);
      // Clean up blob URL on unmount
      if (imageSrc) URL.revokeObjectURL(imageSrc);
    };
  }, []);

  if (loading) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center animate-pulse">
        <div className="text-gray-400 mb-2 text-2xl">🔄</div>
        <p className="text-gray-600">Loading video preview...</p>
      </div>
    );
  }

  if (error || !imageSrc) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center">
        <div className="text-gray-500 mb-2 text-2xl">📹</div>
        <p className="text-gray-600 font-medium">Video preview unavailable</p>
        <p className="text-sm text-gray-500 mt-1">Start video processing to see live feed</p>
      </div>
    );
  }

  return (
    <div className="relative">
      <img
        src={imageSrc}
        alt="Live Traffic Feed"
        className="w-full h-auto rounded-lg shadow-lg"
        onError={() => setError(true)}
      />
      <div className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded text-xs">
        🔴 LIVE
      </div>
    </div>
  );
};

export default VideoPreview;