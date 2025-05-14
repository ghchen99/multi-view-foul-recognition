"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { PredictionResponse, CategoryResult } from '@/lib/types/api';

interface PredictionResultsProps {
  data: PredictionResponse;
}

export const PredictionResults: React.FC<PredictionResultsProps> = ({ data }) => {
  return (
    <div className="flex flex-col space-y-4 w-full">
      <ModelPredictions predictions={data.predictions} />
    </div>
  );
};

interface ModelPredictionsProps {
  predictions: CategoryResult[];
}

export const ModelPredictions: React.FC<ModelPredictionsProps> = ({ predictions }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Model Predictions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {predictions.map((category, idx) => (
          <div key={idx} className="space-y-2">
            <h3 className="font-medium text-sm">{category.category}</h3>
            {category.details.map((detail, detailIdx) => (
              <div key={detailIdx} className="space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-sm">{detail.prediction}</span>
                  <span className="text-sm font-mono">
                    {(detail.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress
                  value={detail.probability * 100}
                  className="h-2"
                  indicatorClassName={getColorClass(detail.probability)}
                />
              </div>
            ))}
            {idx < predictions.length - 1 && <Separator className="my-2" />}
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

interface AIDecisionProps {
  decision: string | null;
  explanation: string | null;
}

export const AIDecision: React.FC<AIDecisionProps> = ({ decision, explanation }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">AI Referee Decision</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center space-x-2">
          <span className="font-medium">Decision:</span>
          <Badge 
            variant="outline" 
            className={getDecisionColorClass(decision || 'Unknown')}
          >
            {decision || 'Unknown'}
          </Badge>
        </div>
        <div className="space-y-2">
          <span className="font-medium">Explanation:</span>
          <p className="text-sm whitespace-pre-line">
            {explanation || 'No explanation available'}
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

// Helper functions for coloring
const getColorClass = (probability: number): string => {
  if (probability >= 0.8) return 'bg-red-500';
  if (probability >= 0.6) return 'bg-orange-500';
  if (probability >= 0.4) return 'bg-yellow-500';
  if (probability >= 0.2) return 'bg-blue-500';
  return 'bg-gray-500';
};

const getDecisionColorClass = (decision: string): string => {
  const lowerDecision = decision.toLowerCase();
  
  if (lowerDecision.includes('red')) return 'bg-red-100 text-red-800 hover:bg-red-100';
  if (lowerDecision.includes('yellow')) return 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100';
  if (lowerDecision.includes('no card')) return 'bg-green-100 text-green-800 hover:bg-green-100';
  
  return '';
};