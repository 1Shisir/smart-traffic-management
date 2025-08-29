import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const VideoDetection = () => {
  const navigate = useNavigate();
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewFrame, setPreviewFrame] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [detectionStats, setDetectionStats] = useState({
    total: 0,
    car: 0,
    bus: 0,
    truck: 0,
    motorcycle: 0
  });
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  console.log('VideoDetection component is rendering...');

  const loadPreviewFrame = async (frameNumber = 0) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`/api/video-detection-preview?frame=${frameNumber}`);
      
      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setPreviewFrame(imageUrl);
        setCurrentFrame(frameNumber);
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to load preview frame');
      }
    } catch (err) {
      console.error('Error loading preview frame:', err);
      setError('Failed to load preview frame');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadDetectedVideo = async () => {
    try {
      setIsProcessing(true);
      setError(null);
      
      // Show processing message
      alert('Processing video with AI detection... This may take several minutes.');
      
      const response = await fetch('/api/video-with-detection');
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `traffic_detected_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.mp4`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        
        alert('Video with detection downloaded successfully!');
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to process video');
      }
    } catch (err) {
      console.error('Error downloading detected video:', err);
      setError('Failed to download video with detection');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate('/')}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition-colors"
            >
              ← Back to Dashboard
            </button>
            <h1 className="text-2xl font-bold text-gray-800">
              🤖 AI Vehicle Detection Video
            </h1>
            <div></div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <strong>Error:</strong> {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Preview Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">👁️ Preview Detection</h2>
            
            {/* Preview Frame */}
            <div className="mb-4">
              {previewFrame ? (
                <img 
                  src={previewFrame} 
                  alt="Detection Preview" 
                  className="w-full h-64 object-cover rounded border"
                />
              ) : (
                <div className="w-full h-64 bg-gray-200 rounded border flex items-center justify-center">
                  <span className="text-gray-500">No preview loaded</span>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Frame Number: {currentFrame}
                </label>
                <input
                  type="number"
                  min="0"
                  value={currentFrame}
                  onChange={(e) => setCurrentFrame(parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <button
                onClick={() => loadPreviewFrame(currentFrame)}
                disabled={isLoading}
                className="w-full bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors"
              >
                {isLoading ? 'Loading...' : 'Load Preview Frame'}
              </button>
            </div>
          </div>

          {/* Download Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">📥 Download Full Video</h2>
            
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded">
                <h3 className="font-medium text-blue-800">What you'll get:</h3>
                <ul className="text-blue-700 text-sm mt-2 space-y-1">
                  <li>• Vehicle detection boundaries</li>
                  <li>• Real-time vehicle counts</li>
                  <li>• Detection confidence scores</li>
                  <li>• Frame-by-frame analysis</li>
                </ul>
              </div>

              {/* Detection Stats */}
              {(detectionStats.total > 0) && (
                <div className="bg-gray-50 p-4 rounded">
                  <h3 className="font-medium mb-2">Current Frame Stats:</h3>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>🚗 Cars: {detectionStats.car}</div>
                    <div>🚌 Buses: {detectionStats.bus}</div>
                    <div>🚛 Trucks: {detectionStats.truck}</div>
                    <div>🏍️ Motorcycles: {detectionStats.motorcycle}</div>
                  </div>
                  <div className="mt-2 font-medium">
                    Total Vehicles: {detectionStats.total}
                  </div>
                </div>
              )}

              <button
                onClick={downloadDetectedVideo}
                disabled={isProcessing}
                className="w-full bg-purple-500 hover:bg-purple-600 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors"
              >
                {isProcessing ? (
                  <span>🔄 Processing Video...</span>
                ) : (
                  <span>📥 Download Video with Detection</span>
                )}
              </button>

              {isProcessing && (
                <div className="text-center text-sm text-gray-600">
                  <p>This process may take several minutes depending on video length.</p>
                  <p>Please wait while AI analyzes each frame...</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">📋 How to Use</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h3 className="font-medium mb-2">Preview Detection:</h3>
              <ol className="space-y-1 text-gray-600">
                <li>1. Enter a frame number</li>
                <li>2. Click "Load Preview Frame"</li>
                <li>3. View detection boundaries</li>
                <li>4. See vehicle counts</li>
              </ol>
            </div>
            <div>
              <h3 className="font-medium mb-2">Download Full Video:</h3>
              <ol className="space-y-1 text-gray-600">
                <li>1. Click "Download Video with Detection"</li>
                <li>2. Wait for AI processing</li>
                <li>3. Video will download automatically</li>
                <li>4. Play with your preferred video player</li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoDetection;
