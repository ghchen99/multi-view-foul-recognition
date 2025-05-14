"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { AlertCircle, RefreshCw, Upload, ArrowUpCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useVideoStore } from '@/lib/store/use-video-store';
import { uploadVideo } from '@/lib/api/football-api';
import { Card, CardContent } from '@/components/ui/card';

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
      <Card className="w-full shadow-sm border bg-secondary/20">
        <CardContent className="p-4 text-center">
          <Button disabled className="w-full py-6">
            <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
            {isUploading ? 'Uploading Video...' : 'Analyzing Incident...'}
          </Button>
          <p className="mt-2 text-xs text-muted-foreground">
            {isUploading 
              ? 'Please wait while we upload your video...' 
              : 'Our AI is analyzing the football incident...'}
          </p>
        </CardContent>
      </Card>
    );
  }

  if (resultData) {
    return (
      <Button 
        variant="outline" 
        onClick={handleReset} 
        className="w-full shadow-sm bg-secondary/30 hover:bg-secondary flex items-center justify-center py-6 text-base"
      >
        <ArrowUpCircle className="mr-2 h-5 w-5" />
        Analyze Another Video
      </Button>
    );
  }

  return (
    <div className="w-full space-y-4">
      <Button 
        onClick={handleProcess} 
        disabled={!videoFile}
        className="w-full py-6 text-base shadow-md flex items-center justify-center"
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