import os
import json
import torch
import logging
from HDF5 import save_to_hdf5
from FeatureExtractor import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ActionData class to store and process action parameters
class ActionData:
    """
    Represents a specific action in the dataset, including associated clips and features.
    """

    def __init__(self, action_data):
        """
        Initialises an ActionData object with the provided action data.
        Sets self.valid to False if the action should be skipped, except for "No offence" cases
        where empty values are populated with "missing".

        Parameters:
        action_data (dict): A dictionary containing detailed information about the action. 
        """
        # Initialize validity flag
        self.valid = True
        
        # Special handling for "No offence" cases
        is_no_offence = action_data['Offence'] == "No offence"
        
        # Check and store offence - skip if empty string and not "No offence"
        if action_data['Offence'] == "" and not is_no_offence:
            logging.warning("Skipping action: Empty offence value")
            self.valid = False
            return
            
        # Check and store bodypart - skip if empty string and not "No offence"
        if action_data['Bodypart'] == "":
            logging.warning("Skipping action: Empty bodypart value")
            self.valid = False
            return
            
        # Check and store actionclass - skip if empty string or Dont know and not "No offence"
        if action_data['Action class'] in ["", "Dont know"]:
            logging.warning("Skipping action: Invalid action class value")
            self.valid = False
            return

        # Store offence value
        self.offence = action_data['Offence']
        
        # Handle bodypart and upperbodypart logic
        if action_data['Bodypart'] == "":
            self.bodypart = "" if is_no_offence else action_data['Bodypart']
        elif action_data['Bodypart'] == 'Upper body':
            upperbodypart = action_data['Upper body part']
            # Only update if upperbodypart is not empty string
            if upperbodypart != "":
                # Convert 'Use of shoulders' to 'Use of shoulder'
                if upperbodypart == 'Use of shoulders':
                    self.bodypart = 'Use of shoulder'
                else:
                    self.bodypart = upperbodypart
            else:
                self.bodypart = 'Upper body'  # Keep original value if upperbodypart is empty string
        else:
            self.bodypart = action_data['Bodypart']
        
        # Store actionclass - use "missing" for empty values in "No offence" cases
        self.actionclass = "" if (action_data['Action class'] == "" and is_no_offence) else action_data['Action class']
        
        # Store severity - use "missing" for empty values in "No offence" cases
        self.severity = "1.0" if (action_data['Severity'] == "") else action_data['Severity']
        
        # Store trytoplay - convert empty string to No or "missing" for "No offence" cases
        self.trytoplay = "No" if (action_data['Try to play'] == "" and is_no_offence) else ('No' if action_data['Try to play'] == "" else action_data['Try to play'])
        
        # Store touchball - convert empty string to No or "missing" for "No offence" cases
        self.touchball = "No" if (action_data['Touch ball'] == "" and is_no_offence) else ('No' if action_data['Touch ball'] == "" else action_data['Touch ball'])
        
        # Store clips
        self.clips = action_data['Clips']

    def extract_clip_features(self):
        """
        Extracts motion features from the clips associated with the action.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for clip in self.clips:
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
            action.extract_clip_features()
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
