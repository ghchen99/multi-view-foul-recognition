import os
import json
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms
from torchvision.models.video import (
    r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights,
    mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights
)
from typing import List, Dict, Union, Optional
from ActionData import ActionData
from HDF5Reader import save_to_hdf5

class FeatureExtractor:
    def __init__(self, model_type: str = 'r3d_18', device: str = 'cpu') -> None:
        self.device = device
        self.model = self._initialize_model(model_type).to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                                 std=[0.22803, 0.22145, 0.216989])
        ])
    
    def _initialize_model(self, model_type: str):
        """
        Initializes the model based on the specified model type.
        """
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
        """
        Preprocesses a single frame.
        """
        return self.transform(Image.fromarray(frame))

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses the frames concurrently.
        """
        with ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(self.preprocess_frame, frames))

        frames_tensor = torch.stack(processed_frames)
        return frames_tensor.permute(1, 0, 2, 3)

    def extract_features(self, video_path: str) -> torch.Tensor:
        """
        Extracts features from the video file.
        """
        logging.info(f"Starting feature extraction for video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []

        # Capture frames 63 to 87 sequentially (avoid concurrent access to VideoCapture)
        for frame_idx in range(63, 88):
            logging.debug(f"Processing frame {frame_idx}...")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logging.warning(f"Failed to read frame {frame_idx}")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        logging.info("Released video capture.")

        if len(frames) != 25:
            logging.error(f"Expected 25 frames, but got {len(frames)}")
            raise ValueError(f"Expected 25 frames, but got {len(frames)}")

        logging.debug(f"Preprocessing {len(frames)} frames...")
        frames_tensor = self.preprocess_frames(frames).unsqueeze(0).to(self.device)

        # logging.info(f"Shape of frames_tensor: {frames_tensor.shape}")

        with torch.no_grad():
            logging.debug("Extracting features using the model...")
            features = self.model(frames_tensor)

        logging.info("Feature extraction completed successfully.")

        return features


def extract_clip_features(action: ActionData) -> None:
    """
    Extracts motion features from the clips associated with the action with enhanced logging and concurrency.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Starting feature extraction for action with {len(action.clips)} clips")

    # Use ThreadPoolExecutor for concurrent extraction
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for clip in action.clips:
            video_path = os.path.join('data', clip['Url'].lower() + '.mp4')
            futures.append(executor.submit(extract_video_features, video_path, clip, device))
        
        # Wait for all tasks to complete and handle results
        for future in as_completed(futures):
            try:
                future.result()  # Raises exception if occurred during video feature extraction
            except Exception as e:
                logging.error(f"Error during feature extraction: {str(e)}")

def extract_video_features(video_path: str, clip: Dict, device: torch.device) -> None:
    """
    Helper function to extract features for a single video and assign them to a clip.
    """
    extractor = FeatureExtractor(model_type='r3d_18', device=device)

    if os.path.exists(video_path):
        logging.info(f"Extracting features for video: {video_path}")
        try:
            clip['video_features'] = extractor.extract_features(video_path)
            logging.info(f"Features extracted for video: {video_path}")
        except Exception as e:
            logging.error(f"Error extracting features for {video_path}: {str(e)}")
            clip['video_features'] = None
    else:
        logging.error(f"Video file not found: {video_path}")
        clip['video_features'] = None

def process_annotations(annotations: Dict[str, Union[int, Dict]], max_actions: Optional[int] = None) -> List[ActionData]:
    """
    Processes dataset annotations and extracts video features.
    If `max_actions` is specified, processes only the first `max_actions` actions.
    """
    result = []
    action_count = 0

    for action_id, action_data in annotations['Actions'].items():
        if max_actions and action_count >= max_actions:
            break
        
        logging.info(f"Processing Action ID: {action_id}")
        action = ActionData(action_data)
        if action.valid: 
            extract_clip_features(action)
            result.append(action)
        else:
            logging.info(f"Skipped Action ID: {action_id}")
        
        action_count += 1

    return result


def main() -> None:
    """
    Main function to load, process, and save data.
    """
    with open('data/dataset/train/annotations.json', 'r') as f:
        annotations = json.load(f)
    
    logging.info(f"Dataset Set: {annotations['Set']}")
    logging.info(f"Total Actions: {annotations['Number of actions']}")
    actions = process_annotations(annotations, max_actions=3)
    
    output_file = 'data/dataset/train/train_features.h5'
    save_to_hdf5(actions, output_file)

    logging.info(f"Done: Extracted features for {len(actions)} actions.")


if __name__ == "__main__":
    main()
