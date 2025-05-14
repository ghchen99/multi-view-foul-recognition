"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useVideoStore } from '@/lib/store/use-video-store';
import { PredictionResponse } from '@/lib/types/api';
import { AlertCircle, ThumbsUp, Award, Info } from 'lucide-react';

// Helper function for decision color
const getDecisionColorClass = (decision: string): string => {
  const lowerDecision = decision.toLowerCase();
  
  if (lowerDecision.includes('red')) return 'bg-red-100 text-red-800 border-red-200';
  if (lowerDecision.includes('yellow')) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
  if (lowerDecision.includes('no card')) return 'bg-green-100 text-green-800 border-green-200';
  
  return '';
};

interface RefereeDecisionCardProps {
  decision: string | null;
  explanation: string | null;
}

export const RefereeDecisionCard: React.FC<RefereeDecisionCardProps> = ({ decision, explanation }) => {
  const decisionText = decision || 'Unknown';
  const colorClass = getDecisionColorClass(decisionText);
  const icon = decisionText.toLowerCase().includes('no card') 
    ? <ThumbsUp className="h-5 w-5 text-green-600" /> 
    : <AlertCircle className="h-5 w-5 text-orange-600" />;
  
  return (
    <Card className="w-full h-full shadow-md overflow-hidden flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Award className="h-5 w-5 text-primary" />
          Referee Decision
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 flex-grow flex flex-col">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-medium text-sm">Decision:</span>
            <Badge 
              variant="outline" 
              className={`px-3 py-1 text-sm ${colorClass}`}
            >
              {decisionText}
            </Badge>
          </div>
          
          <div className="bg-secondary/30 rounded-lg p-4 flex gap-3 items-start">
            {icon}
            <div className="flex-1">
              <h4 className="font-medium text-sm mb-1">Explanation:</h4>
              <p className="text-sm text-muted-foreground">
                {explanation || 'No explanation available'}
              </p>
            </div>
          </div>
        </div>
      
        <div className="flex-grow"></div>
        
        <div className="mt-auto pt-3 border-t border-border/50">
          <div className="flex items-center text-xs text-muted-foreground gap-1.5">
            <Info className="h-3.5 w-3.5" />
            <span>AI referee decisions are generated based on similar incidents and official guidelines</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};