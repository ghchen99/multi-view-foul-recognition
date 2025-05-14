"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { PredictionResponse, CategoryResult } from '@/lib/types/api';
import { BarChart3, Info } from 'lucide-react';

interface PredictionResultsProps {
  data: PredictionResponse;
}

export const PredictionResults: React.FC<PredictionResultsProps> = ({ data }) => {
  return (
    <Card className="w-full h-full shadow-md flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          Model Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 flex-grow flex flex-col">
        <div className="space-y-5 flex-grow">
          {data.predictions.map((category, idx) => (
            <div key={idx} className="space-y-3">
              <h3 className="font-semibold text-sm">{category.category}</h3>
              {category.details.map((detail, detailIdx) => (
                <div key={detailIdx} className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">{detail.prediction}</span>
                    <Badge 
                      variant="outline" 
                      className={`text-xs font-mono ${getProbabilityColorClass(detail.probability)}`}
                    >
                      {(detail.probability * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <Progress
                    value={detail.probability * 100}
                    className="h-2"
                    indicatorClassName={getColorClass(detail.probability)}
                  />
                </div>
              ))}
              {idx < data.predictions.length - 1 && <Separator className="my-3" />}
            </div>
          ))}
        </div>
        
        <div className="mt-auto pt-3 border-t border-border/50">
          <div className="flex items-center text-xs text-muted-foreground gap-1.5">
            <Info className="h-3.5 w-3.5" />
            <span>Percentages indicate confidence level of the AI model</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Helper functions for coloring
const getProbabilityColorClass = (probability: number): string => {
  if (probability >= 0.8) return 'bg-red-50 text-red-800 border-red-200';
  if (probability >= 0.6) return 'bg-orange-50 text-orange-800 border-orange-200';
  if (probability >= 0.4) return 'bg-yellow-50 text-yellow-800 border-yellow-200';
  if (probability >= 0.2) return 'bg-blue-50 text-blue-800 border-blue-200';
  return 'bg-gray-50 text-gray-800 border-gray-200';
};

const getColorClass = (probability: number): string => {
  if (probability >= 0.8) return 'bg-red-500';
  if (probability >= 0.6) return 'bg-orange-500';
  if (probability >= 0.4) return 'bg-yellow-500';
  if (probability >= 0.2) return 'bg-blue-500';
  return 'bg-gray-500';
};