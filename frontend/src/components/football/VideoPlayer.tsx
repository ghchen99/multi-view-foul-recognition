"use client";

import React, { useRef, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Play, Pause, RefreshCw, Film } from 'lucide-react';
import { useVideoStore } from '@/lib/store/use-video-store';

export const VideoPlayer: React.FC = () => {
  const { videoUrl } = useVideoStore();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [isLoaded, setIsLoaded] = React.useState(false);

  useEffect(() => {
    setIsLoaded(false);
    
    // Reset video state when URL changes
    if (videoRef.current) {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  }, [videoUrl]);

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleReplay = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleLoadedData = () => {
    setIsLoaded(true);
  };

  if (!videoUrl) {
    return (
      <Card className="w-full shadow-md">
        <CardContent className="p-8 flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <Film className="h-10 w-10 mb-3 mx-auto text-muted" />
            <p>Upload a video to analyze</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full overflow-hidden shadow-md">
      <CardContent className="p-0 relative">
        {!isLoaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-muted/20">
            <Skeleton className="w-full h-full" />
          </div>
        )}
        <div className="aspect-video bg-black relative overflow-hidden">
          <video 
            ref={videoRef}
            src={videoUrl}
            className="w-full h-full object-contain"
            controls={false}
            onLoadedData={handleLoadedData}
          />
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-3">
            <Button 
              variant="secondary" 
              size="sm" 
              onClick={handlePlayPause}
              className="bg-black/60 hover:bg-black/80 text-white shadow-lg"
            >
              {isPlaying ? <Pause className="h-4 w-4 mr-1" /> : <Play className="h-4 w-4 mr-1" />}
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
            <Button 
              variant="secondary" 
              size="sm" 
              onClick={handleReplay}
              className="bg-black/60 hover:bg-black/80 text-white shadow-lg"
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Replay
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};