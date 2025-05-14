// src/lib/api/football-api.ts

import axios from 'axios';
import { UploadVideoResponse } from '../types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export const uploadVideo = async (videoFile: File): Promise<UploadVideoResponse> => {
  try {
    const formData = new FormData();
    formData.append('video', videoFile);

    const response = await axios.post(`${API_BASE_URL}/api/inference`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      return {
        success: false,
        error: error.response.data.detail || 'An error occurred during video processing',
      };
    }
    
    return {
      success: false,
      error: 'Failed to connect to the server',
    };
  }
};