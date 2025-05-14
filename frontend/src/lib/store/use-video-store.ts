// src/lib/store/use-video-store.ts

import { create } from 'zustand';
import { PredictionResponse } from '../types/api';

interface VideoState {
  videoFile: File | null;
  videoUrl: string | null;
  isUploading: boolean;
  isProcessing: boolean;
  resultData: PredictionResponse | null;
  error: string | null;
  setVideoFile: (file: File | null) => void;
  setVideoUrl: (url: string | null) => void;
  setIsUploading: (isUploading: boolean) => void;
  setIsProcessing: (isProcessing: boolean) => void;
  setResultData: (data: PredictionResponse | null) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useVideoStore = create<VideoState>((set) => ({
  videoFile: null,
  videoUrl: null,
  isUploading: false,
  isProcessing: false,
  resultData: null,
  error: null,
  setVideoFile: (file) => set({ videoFile: file }),
  setVideoUrl: (url) => set({ videoUrl: url }),
  setIsUploading: (isUploading) => set({ isUploading }),
  setIsProcessing: (isProcessing) => set({ isProcessing }),
  setResultData: (resultData) => set({ resultData }),
  setError: (error) => set({ error }),
  reset: () => set({
    videoFile: null,
    videoUrl: null,
    isUploading: false,
    isProcessing: false,
    resultData: null,
    error: null,
  }),
}));