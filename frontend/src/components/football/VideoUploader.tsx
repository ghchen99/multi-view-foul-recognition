"use client";

import React, { useCallback, useState } from 'react';
import { Upload, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useVideoStore } from '@/lib/store/use-video-store';

export const VideoUploader = () => {
  const { setVideoFile, setVideoUrl, setError } = useVideoStore();
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const validateVideoFile = (file: File): boolean => {
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-ms-wmv'];
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    if (!validTypes.includes(file.type)) {
      setError(`Invalid file type. Allowed types: mp4, avi, mov, wmv`);
      return false;
    }
    
    if (file.size > maxSize) {
      setError(`File too large. Maximum size is 100MB`);
      return false;
    }
    
    return true;
  };

  const handleFile = useCallback((file: File) => {
    if (validateVideoFile(file)) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setError(null);
    }
  }, [setVideoFile, setVideoUrl, setError]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  }, [handleFile]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full">
      <div
        className={`relative flex flex-col items-center justify-center w-full h-64 p-4 border-2 border-dashed rounded-lg transition-colors ${
          dragActive ? 'border-primary bg-muted/50' : 'border-gray-300'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          className="hidden"
          type="file"
          accept="video/mp4,video/avi,video/quicktime,video/x-ms-wmv"
          onChange={handleFileChange}
        />
        <Upload className="w-10 h-10 mb-3 text-gray-400" />
        <p className="mb-2 text-sm text-gray-500 text-center">
          <span className="font-semibold">Click to upload</span> or drag and drop
        </p>
        <p className="text-xs text-gray-500 text-center">
          MP4, AVI, MOV, or WMV (Max 100MB)
        </p>
      </div>
      
      <ErrorMessage />
    </div>
  );
};

const ErrorMessage = () => {
  const { error } = useVideoStore();
  
  if (!error) return null;
  
  return (
    <Alert variant="destructive" className="mt-4">
      <AlertCircle className="h-4 w-4" />
      <AlertDescription>{error}</AlertDescription>
    </Alert>
  );
};