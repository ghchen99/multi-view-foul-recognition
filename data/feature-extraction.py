import os
import cv2
import json
import h5py
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load annotation file
def load_annotations(file_path):
    """
    Loads annotations from a JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

# Extract general dataset information
def log_dataset_info(annotations):
    """
    Logs general information about the dataset.
    """
    logging.info(f"Dataset Set: {annotations['Set']}")
    logging.info(f"Total Actions: {annotations['Number of actions']}")

# ActionData class to store and process action parameters
class ActionData:
    """
    Represents a specific action in the dataset, including associated clips and features.
    """

    def __init__(self, action_data):
        """
        Initializes an ActionData object with the provided action data.
        """
        self.offence = action_data['Offence']
        self.contact = action_data['Contact']
        self.bodypart = action_data['Bodypart']
        self.upperbodypart = action_data['Upper body part']
        self.actionclass = action_data['Action class']
        self.severity = action_data['Severity'] # {1:"No card", 2:"Borderline No/Yellow", 3:"Yellow card", 4:"Borderline Yellow/Red", 5:"Red card"}
        self.multiplefouls = action_data['Multiple fouls']
        self.trytoplay = action_data['Try to play']
        self.touchball = action_data['Touch ball']
        self.handball = action_data['Handball']
        self.handballoffence = action_data['Handball offence']
        self.clips = action_data['Clips']

    def extract_clip_features(self):
        """
        Extracts motion features from the clips associated with the action.
        """
        for clip in self.clips:
            video_path = os.path.join('data', clip['Url'].lower() + '.mp4')
            clip['video_features'] = extract_features(video_path) if os.path.exists(video_path) else None
            if clip['video_features'] is None:
                logging.error(f"Video file not found: {video_path}")

# Extract motion-related features from a video
def extract_features(video_path, max_frames=None):
    """
    Extracts motion-related features from a sports video.
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

    # Set max_frames to the total frame count if not specified
    if max_frames is None:
        max_frames = frame_count

    # Initialize feature extraction variables
    frame_diffs, optical_flow_magnitudes, keypoint_diffs = [], [], []

    # Read the first frame and initialize previous frame for motion tracking
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        logging.error("Failed to read the first frame.")
        cap.release()
        return {}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Optical flow calculation parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Keypoint detection parameters
    feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=10)

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame_counter >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion difference based on grayscale frames
        frame_diffs.append(np.sum(cv2.absdiff(prev_gray, gray)))

        # Optical Flow computation
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if prev_pts is not None:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            if next_pts is not None:
                flow = next_pts - prev_pts
                optical_flow_magnitudes.extend(np.linalg.norm(flow, axis=1))

        # Keypoint detection
        orb = cv2.ORB_create()
        kp_prev = orb.detect(prev_gray, None)
        kp_frame = orb.detect(gray, None)
        keypoint_diffs.append(len(kp_frame) - len(kp_prev))

        prev_gray = gray
        frame_counter += 1

    cap.release()

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

# Main processing function to extract features for all actions
def process_annotations(annotations):
    """
    Processes dataset annotations and extracts video features.
    """
    result = []
    for action_id, action_data in annotations['Actions'].items():
        logging.info(f"Processing Action ID: {action_id}")
        action = ActionData(action_data)
        action.extract_clip_features()
        result.append(action)

    return result

# Save extracted data to HDF5 file
def save_to_hdf5(actions, output_file):
    """
    Saves action data and video features to an HDF5 file.
    """
    with h5py.File(output_file, 'w') as f:
        for idx, action in enumerate(actions):
            action_group = f.create_group(f"action_{idx}")

            # Save attributes (non-clip data)
            for attr in vars(action):
                if attr != "clips":
                    action_group.create_dataset(attr, data=getattr(action, attr))

            # Create a group for clips
            clips_group = action_group.create_group("clips")

            # Process and store each clip
            for clip_idx, clip in enumerate(action.clips):
                clip_group = clips_group.create_group(f"clip_{clip_idx}")

                # Save all clip attributes except video_features
                for key, value in clip.items():
                    if key != 'video_features':
                        clip_group.create_dataset(key, data=value)

                # Save video_features as a nested dataset
                if clip['video_features'] is not None:
                    video_features_group = clip_group.create_group('video_features')
                    for feature_key, feature_value in clip['video_features'].items():
                        video_features_group.create_dataset(feature_key, data=float(feature_value))

    logging.info(f"Action data saved to {output_file}")

# Main function to load, process, and save data
def main():
    annotations = load_annotations('data/dataset/test/annotations.json')
    
    log_dataset_info(annotations)
    actions = process_annotations(annotations)
    
    output_file = 'data/video_features.h5'
    save_to_hdf5(actions, output_file)

    logging.info(f"Done: Extracted features for {len(actions)} actions.")

if __name__ == "__main__":
    main()
