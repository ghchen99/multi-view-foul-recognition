"use client";

import React, { useCallback, useState } from 'react';
import { Upload, AlertCircle, Film } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useVideoStore } from '@/lib/store/use-video-store';
import { Card, CardContent } from '@/components/ui/card';

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
    <Card className="w-full overflow-hidden shadow-lg border-2">
      <CardContent className="p-8">
        <div className="text-center mb-6">
          <Film className="h-16 w-16 mx-auto mb-4 text-primary" />
          <h2 className="text-2xl font-bold mb-2">Football Incident Analysis</h2>
          <p className="text-muted-foreground">
            Upload a video clip of a football incident to receive an instant AI referee decision
          </p>
        </div>
        
        <div
          className={`relative flex flex-col items-center justify-center w-full h-64 p-4 border-2 border-dashed rounded-lg transition-all cursor-pointer ${
            dragActive 
              ? 'border-primary bg-primary/5 scale-[1.02]' 
              : 'border-gray-300 hover:border-primary/50 hover:bg-secondary/50'
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
          <div className={`transition-all duration-200 ${dragActive ? 'scale-110' : ''}`}>
            <Upload className="w-12 h-12 mb-4 mx-auto text-primary" />
            <p className="mb-2 text-sm font-medium text-center">
              <span className="font-bold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-500 text-center">
              MP4, AVI, MOV, or WMV (Max 100MB)
            </p>
          </div>
        </div>

        <div className="mt-6 space-y-3">
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <div className="w-10 h-px bg-border"></div>
            <span>HOW IT WORKS</span>
            <div className="w-10 h-px bg-border"></div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="p-3">
              <div className="font-bold text-lg mb-1">1</div>
              <p className="text-xs text-muted-foreground">Upload football incident video</p>
            </div>
            <div className="p-3">
              <div className="font-bold text-lg mb-1">2</div>
              <p className="text-xs text-muted-foreground">AI analyzes player actions</p>
            </div>
            <div className="p-3">
              <div className="font-bold text-lg mb-1">3</div>
              <p className="text-xs text-muted-foreground">Get referee decision & explanation</p>
            </div>
          </div>
        </div>
      </CardContent>
      
      <ErrorMessage />
    </Card>
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