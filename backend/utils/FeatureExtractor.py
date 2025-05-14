import os
import json
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms
from torchvision.models.video import (
    r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights,
    mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights
)
from typing import List, Dict, Union, Optional
from models.ActionData import ActionData
from utils.HDF5Reader import save_to_hdf5

class FeatureExtractor:
    def __init__(self, base_dir: str = 'data/dataset/', model_type: str = 'r3d_18'):
        self.base_dir = Path(base_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model(model_type).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                               std=[0.22803, 0.22145, 0.216989])
        ])
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _initialize_model(self, model_type: str):
        """Initialize the model based on the specified model type."""
        model_mapping = {
            'r3d_18': (r3d_18, R3D_18_Weights.DEFAULT),
            'mc3_18': (mc3_18, MC3_18_Weights.DEFAULT),
            'r2plus1d_18': (r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
            's3d': (s3d, S3D_Weights.DEFAULT),
            'mvit_v2_s': (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
            'mvit_v1_b': (mvit_v1_b, MViT_V1_B_Weights.DEFAULT)
        }

        if model_type not in model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class, model_weights = model_mapping[model_type]
        model = model_class(weights=model_weights)
        
        # Remove the final classification layer
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = torch.nn.Identity()
        elif hasattr(model, 'head'):
            model.head = torch.nn.Identity()

        return model

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        return self.transform(Image.fromarray(frame))

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames concurrently."""
        with ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(self.preprocess_frame, frames))

        frames_tensor = torch.stack(processed_frames)
        return frames_tensor.permute(1, 0, 2, 3)

    def extract_video_features(self, video_path: str) -> Optional[torch.Tensor]:
        """Extract features from a single video file."""
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return None

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video file: {video_path}")
                return None

            frames = []
            # Capture frames 63 to 87
            for frame_idx in range(63, 88):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame {frame_idx}")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) != 25:
                logging.error(f"Expected 25 frames, but got {len(frames)}")
                return None

            frames_tensor = self.preprocess_frames(frames).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(frames_tensor)

            return features

        except Exception as e:
            logging.error(f"Error extracting features for {video_path}: {str(e)}")
            return None

    def extract_clip_features(self, action: ActionData) -> None:
        """Extract features from all clips of an action concurrently."""
        logging.info(f"Starting feature extraction for action with {len(action.clips)} clips")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            for clip in action.clips:
                video_path = os.path.join('data', clip['Url'].lower() + '.mp4')
                futures.append(executor.submit(self._process_single_clip, video_path, clip))
            
            # Wait for all clips to complete and handle results
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error during feature extraction: {str(e)}")

    def _process_single_clip(self, video_path: str, clip: Dict) -> None:
        """Process a single clip and store its features."""
        logging.info(f"Extracting features for video: {video_path}")
        clip['video_features'] = self.extract_video_features(video_path)

    def extract_features(self, split: str, max_actions: Optional[int] = None) -> str:
        """Extract features for a specific dataset split."""
        input_file = self.base_dir / split / 'annotations.json'
        output_file = self.base_dir / split / f'{split}_features.h5'
        
        if not input_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {input_file}")
            
        # Load annotations
        with open(input_file, 'r') as f:
            annotations = json.load(f)
            
        logging.info(f"Dataset Set: {annotations['Set']}")
        logging.info(f"Total Actions: {annotations['Number of actions']}")
        
        # Process one action at a time
        actions = []
        action_count = 0
        
        for action_id, action_data in annotations['Actions'].items():
            if max_actions and action_count >= max_actions:
                break
                
            logging.info(f"Processing Action ID: {action_id}")
            action = ActionData(action_data)
            
            if action.valid:
                # Process all clips for this action concurrently
                self.extract_clip_features(action)
                actions.append(action)
            else:
                logging.info(f"Skipped Action ID: {action_id}")
            
            action_count += 1
        
        # Save extracted features
        save_to_hdf5(actions, str(output_file))
        logging.info(f"Saved features to {output_file}")
        
        return str(output_file)

def main():
    """Extract features for training and validation sets."""
    extractor = FeatureExtractor()
    
    try:
        # Extract features for training and validation
        train_features = extractor.extract_features('train')
        valid_features = extractor.extract_features('valid')
        test_features = extractor.extract_features('test')
        
        logging.info("Feature extraction completed successfully!")
        
    except Exception as e:
        logging.error(f"Feature extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()