"use client";

import React from 'react';
import { VideoUploader } from './VideoUploader';
import { VideoPlayer } from './VideoPlayer';
import { VideoProcessor } from './VideoProcessor';
import { PredictionResults } from './PredictionResults';
import { useVideoStore } from '@/lib/store/use-video-store';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// Helper function for decision color
const getDecisionColorClass = (decision: string): string => {
  const lowerDecision = decision.toLowerCase();
  
  if (lowerDecision.includes('red')) return 'bg-red-100 text-red-800 hover:bg-red-100';
  if (lowerDecision.includes('yellow')) return 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100';
  if (lowerDecision.includes('no card')) return 'bg-green-100 text-green-800 hover:bg-green-100';
  
  return '';
};

export const FootballAnalyzer: React.FC = () => {
  const { videoUrl, resultData } = useVideoStore();

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="flex flex-col space-y-8">
        <header className="text-center">
          <h1 className="text-3xl font-bold tracking-tight mb-2">
            AI Football Referee
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload a video of a football incident to get an AI-powered referee decision. 
            The system analyzes player actions and provides a ruling based on football regulations.
          </p>
        </header>

        {!videoUrl && (
          <div className="max-w-xl mx-auto w-full">
            <VideoUploader />
          </div>
        )}

        {videoUrl && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left side - Model Predictions */}
            <div className="lg:col-span-3 lg:order-first order-last">
              {resultData && (
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold">Model Analysis</h2>
                  <div className="lg:block">
                    <PredictionResults data={resultData} />
                  </div>
                </div>
              )}
            </div>

            {/* Center - Video Player */}
            <div className="lg:col-span-6">
              <div className="space-y-4">
                <VideoPlayer />
                <VideoProcessor />
              </div>
            </div>

            {/* Right side - AI Decision */}
            <div className="lg:col-span-3">
              {resultData && (
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold">Referee Decision</h2>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">AI Referee Decision</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">Decision:</span>
                        <Badge 
                          variant="outline" 
                          className={getDecisionColorClass(resultData.ai_decision || 'Unknown')}
                        >
                          {resultData.ai_decision || 'Unknown'}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <span className="font-medium">Explanation:</span>
                        <p className="text-sm whitespace-pre-line">
                          {resultData.explanation || 'No explanation available'}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};