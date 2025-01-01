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
from typing import List, Dict, Union
from ActionData import ActionData
from HDF5 import save_to_hdf5


class FeatureExtractor:
    def __init__(self, model_type: str = 'r3d_18', device: str = 'cpu') -> None:
        self.model_type = model_type
        self.device = device
        
        # Initialize model based on model type
        self.model = self._initialize_model(model_type)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Define transformation for frames
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

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses the frames before passing them to the model.
        """
        processed_frames = [self.transform(Image.fromarray(frame)) for frame in frames]
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

        # Capture frames 63 to 87
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
        
        logging.info(f"Shape of frames_tensor: {frames_tensor.shape}")

        with torch.no_grad():
            logging.debug("Extracting features using the model...")
            features = self.model(frames_tensor)
        
        logging.info("Feature extraction completed successfully.")

        return features

    def extract_features_legacy(self, video_path: str) -> Dict[str, float]:
        """
        Extracts motion-related features using legacy methods.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count, height, width, fps = map(int, [
            cap.get(cv2.CAP_PROP_FRAME_COUNT),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FPS)
        ])

        logging.info(f"Video: {video_path}, Frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

        # Frame range for analysis (63 to 87)
        start_frame, end_frame = 63, 87
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                  for frame in self._load_frames(cap, start_frame, end_frame)]

        cap.release()

        if len(frames) < 2:
            logging.error("Not enough frames loaded for processing.")
            return {}

        # Perform feature extraction using ThreadPoolExecutor for parallel processing
        results = self._process_frames_parallel(frames)

        # Normalize and return the extracted features
        return self._normalize_features(*results)

    def _load_frames(self, cap, start_frame, end_frame) -> List[np.ndarray]:
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def _process_frames_parallel(self, frames: List[np.ndarray]) -> tuple:
        """
        Process frames in parallel using ThreadPoolExecutor.
        """
        def process_frame(i: int) -> tuple:
            prev_gray, gray = frames[i], frames[i + 1]

            # Motion difference
            frame_diff = np.sum(cv2.absdiff(prev_gray, gray))

            # Optical Flow
            optical_flow_mag = self._compute_optical_flow(prev_gray, gray)

            # Keypoint detection
            keypoint_diff = self._compute_keypoint_diff(prev_gray, gray)

            return frame_diff, optical_flow_mag, keypoint_diff

        with ThreadPoolExecutor() as executor:
            return list(executor.map(process_frame, range(len(frames) - 1)))

    def _compute_optical_flow(self, prev_gray: np.ndarray, gray: np.ndarray) -> List[float]:
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        if prev_pts is not None:
            next_pts, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            if next_pts is not None:
                flow = next_pts - prev_pts
                return list(np.linalg.norm(flow, axis=1))
        return []

    def _compute_keypoint_diff(self, prev_gray: np.ndarray, gray: np.ndarray) -> int:
        orb = cv2.ORB_create()
        kp_prev = orb.detect(prev_gray, None)
        kp_frame = orb.detect(gray, None)
        return len(kp_frame) - len(kp_prev)

    def _normalize_features(self, frame_diffs, optical_flow_magnitudes, keypoint_diffs):
        def normalize_feature(feature: np.ndarray) -> np.ndarray:
            return feature / np.max(feature) if len(feature) > 0 else feature

        return {
            'mean_motion': np.mean(normalize_feature(np.array(frame_diffs))),
            'std_motion': np.std(frame_diffs),
            'max_motion': np.max(frame_diffs),
            'mean_optical_flow': np.mean(optical_flow_magnitudes),
            'std_optical_flow': np.std(optical_flow_magnitudes),
            'max_optical_flow': np.max(optical_flow_magnitudes),
            'mean_keypoint_diff': np.mean(keypoint_diffs),
            'std_keypoint_diff': np.std(keypoint_diffs)
        }


def extract_clip_features(action: ActionData) -> None:
    """
    Extracts motion features from the clips associated with the action with enhanced logging and concurrency.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Starting feature extraction for action with {len(action.clips)} clips")

    # Use ThreadPoolExecutor for concurrent extraction
    with ThreadPoolExecutor(max_workers=4) as executor:
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


def process_annotations(annotations: Dict[str, Union[int, Dict]]) -> List[ActionData]:
    """
    Processes dataset annotations and extracts video features.
    """
    result = []
    
    for action_id, action_data in annotations['Actions'].items():
        logging.info(f"Processing Action ID: {action_id}")
        action = ActionData(action_data)
        if action.valid:  # Check if initialisation succeeded
            extract_clip_features(action)
            result.append(action)
        else:
            logging.info(f"Skipped Action ID: {action_id}")

    return result


def main() -> None:
    """
    Main function to load, process, and save data.
    """
    with open('data/dataset/train/annotations.json', 'r') as f:
        annotations = json.load(f)
    
    logging.info(f"Dataset Set: {annotations['Set']}")
    logging.info(f"Total Actions: {annotations['Number of actions']}")
    actions = process_annotations(annotations)
    
    output_file = 'data/dataset/train/train_features.h5'
    save_to_hdf5(actions, output_file)

    logging.info(f"Done: Extracted features for {len(actions)} actions.")


if __name__ == "__main__":
    main()
