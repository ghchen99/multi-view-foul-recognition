import cv2
import torch
import os
import json
import numpy as np
import logging
from HDF5 import save_to_hdf5
from concurrent.futures import ThreadPoolExecutor
from ActionData import ActionData
from PIL import Image
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights

class FeatureExtractor:
    def __init__(self, model_type='r3d_18', device='cpu'):
        self.model_type = model_type
        self.device = device
        
        if model_type == 'r3d_18':
            self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
        elif model_type == 'mc3_18':
            self.model = mc3_18(weights=MC3_18_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
        elif model_type == 'r2plus1d_18':
            self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
        elif model_type == 's3d':
            self.model = s3d(weights=S3D_Weights.DEFAULT)
            self.model.classifier = torch.nn.Identity()
        elif model_type == 'mvit_v2_s':
            self.model = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
            self.model.head = torch.nn.Identity()
        elif model_type == 'mvit_v1_b':
            self.model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
            self.model.head = torch.nn.Identity()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                              std=[0.22803, 0.22145, 0.216989])
        ])

    def preprocess_frames(self, frames):
        processed_frames = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            processed_frame = self.transform(pil_image)
            processed_frames.append(processed_frame)
        
        frames_tensor = torch.stack(processed_frames)

        return frames_tensor.permute(1, 0, 2, 3)

    def extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        for frame_idx in range(63, 88):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) != 25:
            raise ValueError(f"Expected 25 frames, got {len(frames)}")
        
        frames_tensor = self.preprocess_frames(frames).unsqueeze(0).to(self.device)
        
        # TODO: Fix for mvit models
        print(f"Shape of frames_tensor: {frames_tensor.shape}")
        
        with torch.no_grad():
            features = self.model(frames_tensor)
            
        return features
    
    # Extract motion-related features from a video
    def extract_features_legacy(self, video_path):
        """
        Extracts motion-related features from as sports video within the specified frame range.
        """
        cap = cv2.VideoCapture(video_path)

        # Retrieve video properties
        frame_count, height, width, fps = map(int, [
            cap.get(cv2.CAP_PROP_FRAME_COUNT),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FPS)
        ])

        logging.info(f"Video: {video_path}, Frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

        # Frame range for analysis (frames 63 to 87)
        start_frame = 63
        end_frame = 87

        # Initialize feature extraction variables
        frame_diffs, optical_flow_magnitudes, keypoint_diffs = [], [], []

        # Move to the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Preload frames into memory
        frames = []
        for frame_counter in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()

        # Check if enough frames were loaded
        if len(frames) < 2:
            logging.error("Not enough frames loaded for processing.")
            return {}

        # Optical flow calculation parameters
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Keypoint detection parameters
        feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=10)

        def process_frame(i):
            nonlocal frames

            prev_gray = frames[i]
            gray = frames[i + 1]

            # Motion difference based on grayscale frames
            frame_diff = np.sum(cv2.absdiff(prev_gray, gray))

            # Optical Flow computation
            optical_flow_mag = []
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if prev_pts is not None:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
                if next_pts is not None:
                    flow = next_pts - prev_pts
                    optical_flow_mag.extend(np.linalg.norm(flow, axis=1))

            # Keypoint detection
            orb = cv2.ORB_create()
            kp_prev = orb.detect(prev_gray, None)
            kp_frame = orb.detect(gray, None)
            keypoint_diff = len(kp_frame) - len(kp_prev)

            return frame_diff, optical_flow_mag, keypoint_diff

        # Parallel processing
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_frame, range(len(frames) - 1)))

        # Collect results
        for frame_diff, optical_flow_mag, keypoint_diff in results:
            frame_diffs.append(frame_diff)
            optical_flow_magnitudes.extend(optical_flow_mag)
            keypoint_diffs.append(keypoint_diff)

        # Normalize features
        def normalize_feature(feature):
            return feature / np.max(feature) if len(feature) > 0 else feature

        frame_diffs = normalize_feature(np.array(frame_diffs))
        optical_flow_magnitudes = np.array(optical_flow_magnitudes)

        return {
            'mean_motion': np.mean(frame_diffs),
            'std_motion': np.std(frame_diffs),
            'max_motion': np.max(frame_diffs),
            'mean_optical_flow': np.mean(optical_flow_magnitudes),
            'std_optical_flow': np.std(optical_flow_magnitudes),
            'max_optical_flow': np.max(optical_flow_magnitudes),
            'mean_keypoint_diff': np.mean(keypoint_diffs),
            'std_keypoint_diff': np.std(keypoint_diffs)
        }

def extract_clip_features(action: ActionData):
    """
    Extracts motion features from the clips associated with the action.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for clip in action.clips:
        extractor = FeatureExtractor(model_type='r3d_18', device=device)
        video_path = os.path.join('data', clip['Url'].lower() + '.mp4')
        clip['video_features'] = extractor.extract_features(video_path) if os.path.exists(video_path) else None
        if clip['video_features'] is None:
            logging.error(f"Video file not found: {video_path}")
    

# Main processing function to extract features for all actions
def process_annotations(annotations):
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

# Main function to load, process, and save data
def main():
    
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