"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { AlertCircle, RefreshCw, Upload } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useVideoStore } from '@/lib/store/use-video-store';
import { uploadVideo } from '@/lib/api/football-api';

export const VideoProcessor: React.FC = () => {
  const { 
    videoFile, 
    setIsUploading, 
    setIsProcessing, 
    setResultData, 
    setError,
    isUploading,
    isProcessing,
    resultData,
    reset
  } = useVideoStore();

  const handleProcess = async () => {
    if (!videoFile) {
      setError('Please upload a video first');
      return;
    }

    try {
      setIsUploading(true);
      setError(null);
      
      // Start processing
      setIsProcessing(true);
      
      const result = await uploadVideo(videoFile);
      
      if (result.success && result.data) {
        setResultData(result.data);
      } else {
        setError(result.error || 'An error occurred during video processing');
      }
    } catch (error) {
      setError('Failed to process video');
      console.error(error);
    } finally {
      setIsUploading(false);
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    reset();
  };

  if (isUploading || isProcessing) {
    return (
      <Button disabled className="w-full">
        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
        {isUploading ? 'Uploading...' : 'Processing...'}
      </Button>
    );
  }

  if (resultData) {
    return (
      <Button variant="outline" onClick={handleReset} className="w-full">
        <Upload className="mr-2 h-4 w-4" />
        Analyze Another Video
      </Button>
    );
  }

  return (
    <div className="w-full space-y-4">
      <Button 
        onClick={handleProcess} 
        disabled={!videoFile}
        className="w-full"
      >
        Process Video
      </Button>
      
      {!videoFile && (
        <Alert variant="default" className="bg-muted">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Upload a video of a football incident to get started
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};