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
    Extracts motion-related features from a video file.

    Args:
    video_path (str): Path to the video file.

    Returns:
    dict: A dictionary containing motion features such as mean, std, and max motion.
    """
    cap = cv2.VideoCapture(video_path)

    # Retrieve video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logging.info(f"Video: {video_path}")
    logging.info(f"Frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

    # Initialize motion intensity calculation
    frame_diffs = []
    
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        logging.error("Failed to read the first frame.")
        cap.release()
        return {}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        frame_diffs.append(np.sum(diff))
        prev_gray = gray

    cap.release()

    # Normalize differences
    if len(frame_diffs) > 0:
        frame_diffs = np.array(frame_diffs) / np.max(frame_diffs)

    # Return statistics as potential features
    return {
        'mean_motion': np.mean(frame_diffs) if len(frame_diffs) > 0 else 0,
        'std_motion': np.std(frame_diffs) if len(frame_diffs) > 0 else 0,
        'max_motion': np.max(frame_diffs) if len(frame_diffs) > 0 else 0
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
