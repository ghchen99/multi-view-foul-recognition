"use client";

import React from 'react';
import { VideoUploader } from './VideoUploader';
import { VideoPlayer } from './VideoPlayer';
import { VideoProcessor } from './VideoProcessor';
import { PredictionResults } from './PredictionResults';
import { RefereeDecisionCard } from './RefereeDecisionCard';
import { useVideoStore } from '@/lib/store/use-video-store';
import { Separator } from '@/components/ui/separator';
import { Activity } from 'lucide-react';

export const FootballAnalyzer: React.FC = () => {
  const { videoUrl, resultData } = useVideoStore();

  return (
    <div className="container mx-auto py-12 px-4 max-w-4xl">
      <div className="flex flex-col space-y-8">
        <header className="text-center mb-4">
          <div className="flex justify-center mb-3">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10">
              <Activity className="h-8 w-8 text-primary" />
            </div>
          </div>
          <h1 className="text-4xl font-bold tracking-tight mb-3">
            AI Football Referee
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload a video of a football incident to get an AI-powered referee decision. 
            The system analyzes player actions and provides a ruling based on football regulations.
          </p>
        </header>

        {!videoUrl && (
          <div className="mx-auto w-full max-w-2xl">
            <VideoUploader />
          </div>
        )}

        {videoUrl && (
          <div className="flex flex-col space-y-8">
            {/* Video Player */}
            <div className="w-full">
              <VideoPlayer />
              <div className="mt-4">
                <VideoProcessor />
              </div>
            </div>

            {resultData && (
              <>
                <Separator className="my-2" />
                
                {/* Results Section */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Referee Decision - Now using our new component */}
                  <RefereeDecisionCard 
                    decision={resultData.ai_decision} 
                    explanation={resultData.explanation} 
                  />

                  {/* Model Analysis */}
                  <PredictionResults data={resultData} />
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};