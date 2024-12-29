import os
import cv2
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load annotation file
def load_annotations(file_path):
    """
    Loads annotations from a JSON file.

    Args:
    file_path (str): Path to the annotation JSON file.

    Returns:
    dict: The parsed JSON content representing the dataset annotations.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

# Extract general dataset information
def log_dataset_info(annotations):
    """
    Logs general information about the dataset.

    Args:
    annotations (dict): The dataset annotations.
    """
    logging.info(f"Dataset Set: {annotations['Set']}")
    logging.info(f"Total Actions: {annotations['Number of actions']}")

# Action data class to store and process action parameters
class ActionData:
    """
    A class to represent the data of a specific action in the dataset, including 
    details about the action and associated clips.

    Attributes:
        offence (str): The offence type.
        contact (str): The type of contact made.
        bodypart (str): The body part involved in the action.
        upperbodypart (str): The upper body part involved in the action.
        actionclass (str): The class of action performed.
        severity (str): The severity of the action.
        multiplefouls (str): Whether multiple fouls were involved.
        trytoplay (str): Whether the player tried to play the ball.
        touchball (str): Whether the ball was touched.
        handball (str): Whether a handball occurred.
        handballoffence (str): Whether there was a handball offence.
        clips (list): List of clips associated with the action.

    Methods:
        extract_clip_features: Extracts motion features from the clips associated with the action.
    """
    
    def __init__(self, action_data):
        """
        Initializes an ActionData object with the provided action data.

        Args:
        action_data (dict): A dictionary containing action data.
        """
        self.offence = action_data['Offence']
        self.contact = action_data['Contact']
        self.bodypart = action_data['Bodypart']
        self.upperbodypart = action_data['Upper body part']
        self.actionclass = action_data['Action class']
        self.severity = action_data['Severity']
        self.multiplefouls = action_data['Multiple fouls']
        self.trytoplay = action_data['Try to play']
        self.touchball = action_data['Touch ball']
        self.handball = action_data['Handball']
        self.handballoffence = action_data['Handball offence']
        self.clips = action_data['Clips']

    def extract_clip_features(self):
        """
        Extracts motion features from the clips associated with the action.

        Returns:
        list: A list of extracted features from the clips, or an empty list if no valid clips.
        """
        video_features = []
        for clip in self.clips:
            video_path = os.path.join('data', clip['Url'].lower() + '.mp4')
            if os.path.exists(video_path):
                video_features.append(extract_features(video_path))
        return video_features

def extract_features(video_path):
    """
    Extracts more advanced motion-related features from a sports video.

    Args:
    video_path (str): Path to the video file.

    Returns:
    dict: A dictionary containing motion features such as mean, std, max motion, optical flow statistics, etc.
    """
    cap = cv2.VideoCapture(video_path)

    # Retrieve video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logging.info(f"Video: {video_path}")
    logging.info(f"Frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

    # Initialize feature extraction variables
    frame_diffs = []
    optical_flow_magnitudes = []
    keypoint_diffs = []

    # Read the first frame and initialize previous frame for motion tracking
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        logging.error("Failed to read the first frame.")
        cap.release()
        return {}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize optical flow calculation parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # For keypoint detection using goodFeaturesToTrack, you need to specify maxCorners, qualityLevel, and minDistance
    feature_params = dict(
        maxCorners=1000,  # Maximum number of corners to detect
        qualityLevel=0.01,  # Minimum quality of corners to be considered
        minDistance=10  # Minimum distance between detected corners
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion difference based on grayscale frames
        diff = cv2.absdiff(prev_gray, gray)
        frame_diffs.append(np.sum(diff))

        # Optical Flow computation (Lucas-Kanade method)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)  # Updated to use feature_params
        if prev_pts is not None:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            if next_pts is not None:
                flow = next_pts - prev_pts
                magnitude = np.linalg.norm(flow, axis=1)
                optical_flow_magnitudes.extend(magnitude)

        # Keypoint detection (ORB in this case)
        orb = cv2.ORB_create()
        kp_prev = orb.detect(prev_gray, None)
        kp_frame = orb.detect(gray, None)
        keypoint_diffs.append(len(kp_frame) - len(kp_prev))

        prev_gray = gray

    cap.release()

    # Normalize differences and optical flow
    if len(frame_diffs) > 0:
        frame_diffs = np.array(frame_diffs) / np.max(frame_diffs)
    if len(optical_flow_magnitudes) > 0:
        optical_flow_magnitudes = np.array(optical_flow_magnitudes)

    # Return statistics as potential features
    return {
        'mean_motion': np.mean(frame_diffs) if len(frame_diffs) > 0 else 0,
        'std_motion': np.std(frame_diffs) if len(frame_diffs) > 0 else 0,
        'max_motion': np.max(frame_diffs) if len(frame_diffs) > 0 else 0,
        'mean_optical_flow': np.mean(optical_flow_magnitudes) if len(optical_flow_magnitudes) > 0 else 0,
        'std_optical_flow': np.std(optical_flow_magnitudes) if len(optical_flow_magnitudes) > 0 else 0,
        'max_optical_flow': np.max(optical_flow_magnitudes) if len(optical_flow_magnitudes) > 0 else 0,
        'mean_keypoint_diff': np.mean(keypoint_diffs) if len(keypoint_diffs) > 0 else 0,
        'std_keypoint_diff': np.std(keypoint_diffs) if len(keypoint_diffs) > 0 else 0
    }


# Main processing function
def process_annotations(annotations):
    """
    Processes the dataset annotations and extracts video features from associated clips.

    Args:
    annotations (dict): The dataset annotations.

    Returns:
    list: A list of extracted video features from all actions in the dataset.
    """
    actions = annotations['Actions']
    video_features = []

    for action_id, action_data in actions.items():
        logging.info(f"Processing Action ID: {action_id}")
        
        # Create an ActionData object for each action
        action = ActionData(action_data)
        
        # Extract video features from action clips
        video_features.extend(action.extract_clip_features())

    return video_features

# Load annotations and process
annotations = load_annotations('data/dataset/test/annotations.json')
log_dataset_info(annotations)
video_features = process_annotations(annotations)

# Save features for model input
np.save('data/video_features.npy', video_features)
