import React, { useState, useEffect } from 'react';
import { FaCloud, FaUpload, FaDownload, FaTrash, FaDatabase, FaChartBar } from 'react-icons/fa';

const AWSStorageManager = () => {
  const [awsStatus, setAwsStatus] = useState(null);
  const [files, setFiles] = useState([]);
  const [uploadFile, setUploadFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [activeTab, setActiveTab] = useState('status');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAWSStatus();
    loadFiles();
  }, []);

  const loadAWSStatus = async () => {
    try {
      const response = await fetch('/api/aws/status');
      const data = await response.json();
      setAwsStatus(data);
    } catch (error) {
      console.error('Failed to load AWS status:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadFiles = async (prefix = '') => {
    try {
      const response = await fetch(`/api/aws/files?prefix=${prefix}&limit=50`);
      const data = await response.json();
      if (data.success) {
        setFiles(data.files);
      }
    } catch (error) {
      console.error('Failed to load files:', error);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('video', uploadFile);

    try {
      const response = await fetch('/api/aws/upload/video', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (data.success) {
        alert('Video uploaded successfully!');
        setUploadFile(null);
        loadFiles();
        loadAWSStatus(); // Refresh stats
      } else {
        alert(`Upload failed: ${data.error}`);
      }
    } catch (error) {
      alert(`Upload error: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const downloadFile = async (s3Key, fileName) => {
    try {
      // Get presigned URL for secure download
      const response = await fetch(`/api/aws/files/${encodeURIComponent(s3Key)}/url?expiration=3600`);
      const data = await response.json();
      
      if (data.success) {
        // Open download in new tab
        const link = document.createElement('a');
        link.href = data.presigned_url;
        link.download = fileName || s3Key.split('/').pop();
        link.target = '_blank';
        link.rel = 'noopener noreferrer'; // Security best practice
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        alert(`Download failed: ${data.error}`);
      }
    } catch (error) {
      alert(`Download error: ${error.message}`);
    }
  };

  const exportAnalytics = async () => {
    try {
      // Get current traffic data for analytics export
      const response = await fetch('/api/data?limit=100');
      const data = await response.json();
      
      if (data && data.length > 0) {
        // Prepare analytics data
        const analyticsData = {
          export_date: new Date().toISOString(),
          total_records: data.length,
          date_range: {
            from: data[data.length - 1].timestamp,
            to: data[0].timestamp
          },
          summary: {
            total_vehicles: data.reduce((sum, record) => sum + (record.total_count || 0), 0),
            total_cars: data.reduce((sum, record) => sum + (record.car_count || 0), 0),
            total_buses: data.reduce((sum, record) => sum + (record.bus_count || 0), 0),
            total_trucks: data.reduce((sum, record) => sum + (record.truck_count || 0), 0),
            total_motorcycles: data.reduce((sum, record) => sum + (record.motorcycle_count || 0), 0),
            peak_hour: data.reduce((peak, record) => 
              (record.total_count || 0) > (peak.total_count || 0) ? record : peak, data[0]
            )
          },
          traffic_data: data
        };

        // Upload analytics to S3
        const uploadResponse = await fetch('/api/aws/upload/analytics', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(analyticsData)
        });

        const uploadData = await uploadResponse.json();
        if (uploadData.success) {
          alert('Analytics exported to S3 successfully!');
          loadFiles(); // Refresh file list
          loadAWSStatus(); // Refresh stats
        } else {
          alert(`Analytics export failed: ${uploadData.error}`);
        }
      } else {
        alert('No traffic data available to export');
      }
    } catch (error) {
      alert(`Analytics export error: ${error.message}`);
    }
  };

  const deleteFile = async (s3Key) => {
    if (!confirm(`Are you sure you want to delete ${s3Key}?`)) return;

    try {
      const response = await fetch(`/api/aws/files/${encodeURIComponent(s3Key)}`, {
        method: 'DELETE'
      });

      const data = await response.json();
      if (data.success) {
        alert('File deleted successfully!');
        loadFiles();
        loadAWSStatus(); // Refresh stats
      } else {
        alert(`Delete failed: ${data.error}`);
      }
    } catch (error) {
      alert(`Delete error: ${error.message}`);
    }
  };

  const backupDatabase = async () => {
    if (!confirm('Are you sure you want to backup the database to AWS S3?')) return;

    try {
      const response = await fetch('/api/aws/backup/database', {
        method: 'POST'
      });

      const data = await response.json();
      if (data.success) {
        alert('Database backup completed successfully!');
        loadFiles();
        loadAWSStatus(); // Refresh stats
      } else {
        alert(`Backup failed: ${data.error}`);
      }
    } catch (error) {
      alert(`Backup error: ${error.message}`);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileType = (key) => {
    if (key.startsWith('videos/')) return 'Video';
    if (key.startsWith('analytics/')) return 'Analytics';
    if (key.startsWith('processed_frames/')) return 'Frame';
    return 'Other';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center mb-6">
        <FaCloud className="text-3xl text-blue-500 mr-3" />
        <h2 className="text-2xl font-bold text-gray-800">AWS Storage Manager</h2>
      </div>

      {/* Tab Navigation */}
      <div className="flex mb-6 border-b">
        <button
          onClick={() => setActiveTab('status')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'status'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-blue-600'
          }`}
        >
          Status
        </button>
        <button
          onClick={() => setActiveTab('upload')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'upload'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-blue-600'
          }`}
        >
          Upload
        </button>
        <button
          onClick={() => setActiveTab('files')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'files'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-blue-600'
          }`}
        >
          Files
        </button>
        <button
          onClick={() => setActiveTab('backup')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'backup'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-blue-600'
          }`}
        >
          Backup
        </button>
      </div>

      {/* Status Tab */}
      {activeTab === 'status' && (
        <div className="space-y-4">
          <div className={`p-4 rounded-lg ${
            awsStatus?.aws_available ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${
                awsStatus?.aws_available ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <span className="font-medium">
                AWS S3 Status: {awsStatus?.aws_available ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          {awsStatus?.aws_available && awsStatus.stats && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center">
                  <FaDatabase className="text-blue-500 mr-2" />
                  <div>
                    <p className="text-sm text-gray-600">Total Storage</p>
                    <p className="text-lg font-semibold">{awsStatus.stats.total_size_mb} MB</p>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div className="flex items-center">
                  <FaChartBar className="text-purple-500 mr-2" />
                  <div>
                    <p className="text-sm text-gray-600">Total Files</p>
                    <p className="text-lg font-semibold">{awsStatus.stats.total_objects}</p>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div>
                  <p className="text-sm text-gray-600">Bucket</p>
                  <p className="text-lg font-semibold">{awsStatus.stats.bucket_name}</p>
                  <p className="text-xs text-gray-500">{awsStatus.stats.region}</p>
                </div>
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div>
                  <p className="text-sm text-gray-600">Videos</p>
                  <p className="text-lg font-semibold">{awsStatus.stats.videos_count}</p>
                </div>
              </div>

              <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                <div>
                  <p className="text-sm text-gray-600">Analytics</p>
                  <p className="text-lg font-semibold">{awsStatus.stats.analytics_count}</p>
                </div>
              </div>

              <div className="bg-pink-50 border border-pink-200 rounded-lg p-4">
                <div>
                  <p className="text-sm text-gray-600">Frames</p>
                  <p className="text-lg font-semibold">{awsStatus.stats.frames_count}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div className="space-y-4">
          {!awsStatus?.aws_available ? (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-yellow-800">AWS S3 is not available. Please check your configuration.</p>
            </div>
          ) : (
            <form onSubmit={handleFileUpload} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Video File
                </label>
                <input
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv,.wmv"
                  onChange={(e) => setUploadFile(e.target.files[0])}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  disabled={isUploading}
                />
              </div>

              <button
                type="submit"
                disabled={!uploadFile || isUploading}
                className={`flex items-center px-4 py-2 rounded-lg font-medium ${
                  uploadFile && !isUploading
                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                <FaUpload className="mr-2" />
                {isUploading ? 'Uploading...' : 'Upload Video'}
              </button>
            </form>
          )}
        </div>
      )}

      {/* Files Tab */}
      {activeTab === 'files' && (
        <div className="space-y-4">
          <div className="flex space-x-2">
            <button
              onClick={() => loadFiles('')}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
            >
              All
            </button>
            <button
              onClick={() => loadFiles('videos/')}
              className="px-3 py-1 text-sm bg-blue-100 hover:bg-blue-200 rounded"
            >
              Videos
            </button>
            <button
              onClick={() => loadFiles('analytics/')}
              className="px-3 py-1 text-sm bg-green-100 hover:bg-green-200 rounded"
            >
              Analytics
            </button>
            <button
              onClick={() => loadFiles('processed_frames/')}
              className="px-3 py-1 text-sm bg-purple-100 hover:bg-purple-200 rounded"
            >
              Frames
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    File Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Modified
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {files.map((file, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {file.key.split('/').pop()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {getFileType(file.key)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatFileSize(file.size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(file.last_modified).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => downloadFile(file.key, file.key.split('/').pop())}
                          className="text-blue-600 hover:text-blue-900 p-1"
                          title="Download file"
                        >
                          <FaDownload />
                        </button>
                        <button
                          onClick={() => deleteFile(file.key)}
                          className="text-red-600 hover:text-red-900 p-1"
                          title="Delete file"
                        >
                          <FaTrash />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {files.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No files found
              </div>
            )}
          </div>
        </div>
      )}

      {/* Backup Tab */}
      {activeTab === 'backup' && (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-lg font-medium text-blue-900 mb-2">Database Backup</h3>
            <p className="text-blue-700 mb-4">
              Create a backup of your traffic database and store it securely in AWS S3.
            </p>
            <button
              onClick={backupDatabase}
              disabled={!awsStatus?.aws_available}
              className={`flex items-center px-4 py-2 rounded-lg font-medium ${
                awsStatus?.aws_available
                  ? 'bg-blue-500 text-white hover:bg-blue-600'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              <FaDatabase className="mr-2" />
              Backup Database
            </button>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h3 className="text-lg font-medium text-green-900 mb-2">Analytics Export</h3>
            <p className="text-green-700 mb-4">
              Export current traffic analytics data to AWS S3 for backup and analysis.
            </p>
            <button
              onClick={exportAnalytics}
              disabled={!awsStatus?.aws_available}
              className={`flex items-center px-4 py-2 rounded-lg font-medium mb-4 ${
                awsStatus?.aws_available
                  ? 'bg-green-500 text-white hover:bg-green-600'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              <FaChartBar className="mr-2" />
              Export Analytics
            </button>
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <h3 className="text-lg font-medium text-yellow-900 mb-2">Automatic Backups</h3>
            <p className="text-yellow-700 mb-2">
              The system automatically backs up:
            </p>
            <ul className="text-yellow-700 text-sm space-y-1">
              <li>• Processed frames every 10 frames during video processing</li>
              <li>• Analytics data every 50 frames during video processing</li>
              <li>• All data includes timestamps and metadata for easy retrieval</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default AWSStorageManager;
