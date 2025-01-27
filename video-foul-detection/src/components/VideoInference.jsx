import React, { useState, useRef } from 'react';
import { Upload, Play, Pause, AlertCircle } from 'lucide-react';

const VideoInference = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
      setError(null);
      setVideoUrl(URL.createObjectURL(selectedFile));
    } else {
      setError('Please select a valid video file');
      setFile(null);
      setVideoUrl(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a video file');
      return;
    }

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:5000/api/inference', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to process video');
      }

      setResults(data.predictions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900">AI Football Referee</h1>
        <p className="mt-2 text-gray-600">Upload a football video to detect fouls and violations</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="relative">
          {!videoUrl ? (
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 bg-gray-50">
              <div className="flex flex-col items-center">
                <Upload className="w-16 h-16 text-gray-400 mb-4" />
                <label className="cursor-pointer">
                  <span className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    Choose Video
                  </span>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
                <p className="mt-2 text-sm text-gray-500">
                  Drag and drop or click to select
                </p>
              </div>
            </div>
          ) : (
            <div className="relative rounded-xl overflow-hidden bg-black">
              <video
                ref={videoRef}
                src={videoUrl}
                className="w-full h-96 object-contain"
                onEnded={() => setIsPlaying(false)}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <button
                  type="button"
                  onClick={togglePlayPause}
                  className="p-4 rounded-full bg-black/50 hover:bg-black/70 transition-colors"
                >
                  {isPlaying ? (
                    <Pause className="w-8 h-8 text-white" />
                  ) : (
                    <Play className="w-8 h-8 text-white" />
                  )}
                </button>
              </div>
              <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center">
                <p className="text-white text-sm bg-black/50 px-3 py-1 rounded-full">
                  {file?.name}
                </p>
                <label className="cursor-pointer">
                  <span className="px-3 py-1 bg-white/90 text-gray-900 rounded-full hover:bg-white transition-colors text-sm">
                    Change Video
                  </span>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={loading || !file}
          className="w-full py-3 px-4 rounded-lg shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Processing Video...
            </span>
          ) : (
            'Analyze for Fouls'
          )}
        </button>
      </form>

      {error && (
        <div className="flex items-center gap-2 p-4 text-red-800 bg-red-50 border border-red-200 rounded-lg">
          <AlertCircle className="h-4 w-4" />
          <p>{error}</p>
        </div>
      )}

      {results && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-gray-900">Detected Fouls</h2>
            <p className="text-sm text-gray-500 mt-1">Analysis results from the video</p>
          </div>
          <div className="divide-y divide-gray-200">
            {results.map((category, index) => (
              <div key={index} className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  {category.category}
                </h3>
                <div className="space-y-3">
                  {category.details.map((detail, i) => (
                    <div key={i} className="flex justify-between items-center bg-gray-50 p-3 rounded-lg">
                      <span className="text-gray-700">{detail.prediction}</span>
                      <div className="flex items-center">
                        <div className="w-32 h-2 bg-gray-200 rounded-full mr-3">
                          <div
                            className={`h-full rounded-full ${
                              detail.probability < 0.4
                                ? 'bg-red-500'
                                : detail.probability < 0.7
                                ? 'bg-yellow-500'
                                : 'bg-green-500'
                            }`}
                            style={{ width: `${detail.probability * 100}%` }}
                          />
                        </div>
                        <span className={`font-medium min-w-[60px] text-right ${
                          detail.probability < 0.4
                            ? 'text-red-700'
                            : detail.probability < 0.7
                            ? 'text-yellow-700'
                            : 'text-green-700'
                        }`}>
                          {(detail.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoInference;