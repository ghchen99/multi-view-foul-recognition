// src/lib/types/api.ts

export interface PredictionDetail {
    prediction: string;
    probability: number;
  }
  
  export interface CategoryResult {
    category: string;
    details: PredictionDetail[];
  }
  
  export interface PredictionResponse {
    status: string;
    predictions: CategoryResult[];
    ai_decision: string | null;
    explanation: string | null;
  }
  
  export interface ErrorResponse {
    error: string;
  }
  
  export interface UploadVideoResponse {
    success: boolean;
    data?: PredictionResponse;
    error?: string;
  }