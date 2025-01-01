import os
import h5py
import torch
import logging
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

            # Save clips
            clips_group = action_group.create_group("clips")
            for clip_idx, clip in enumerate(action.clips):
                clip_group = clips_group.create_group(f"clip_{clip_idx}")

                # Save non-video features
                for key, value in clip.items():
                    if key != 'video_features':
                        clip_group.create_dataset(key, data=value)

                # Save video features
                if clip['video_features'] is not None:
                    video_features_group = clip_group.create_group('video_features')
                    video_features_np = clip['video_features'].cpu().numpy()
                    video_features_group.create_dataset('data', data=video_features_np)

    logging.info(f"Saved action data to {output_file}")

def print_field_occurrences(actions):
    """
    Logs distinct occurrences and counts for each field in the actions data.
    """
    if not actions:
        logging.warning("No actions provided for analysis.")
        return

    fields = [field for field in actions[0].keys() if field != 'clips']
    logging.info("Field value distributions:")

    for field in sorted(fields):
        value_counts = Counter(str(action[field]) for action in actions)
        logging.info(f"\n{field} - Total entries: {len(actions)}")
        logging.info("-" * 50)

        for value, count in sorted(value_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(actions)) * 100
            logging.info(f"'{value}': {count} occurrences ({percentage:.1f}%)")

def read_from_hdf5(input_file):
    """
    Reads action data and video features from an HDF5 file.
    """
    with h5py.File(input_file, 'r') as f:
        actions = []
        for action_key in f.keys():
            action_group = f[action_key]
            action_data = {attr: action_group[attr][()] for attr in action_group if attr != 'clips'}

            # Read clips
            clips = []
            if 'clips' in action_group:
                clips_group = action_group['clips']
                for clip_key in clips_group.keys():
                    clip_group = clips_group[clip_key]
                    clip_data = {key: clip_group[key][()] for key in clip_group if key != 'video_features'}

                    # Read video features
                    video_features = None
                    if 'video_features' in clip_group:
                        features_group = clip_group['video_features']
                        if 'data' in features_group:
                            video_features_np = features_group['data'][()]
                            video_features = torch.tensor(video_features_np)

                    clip_data['video_features'] = video_features
                    clips.append(clip_data)

            action_data['clips'] = clips
            actions.append(action_data)

        logging.info(f"Loaded {len(actions)} actions from {input_file}")
        return actions

def main():
    """
    Main function to load and inspect HDF5 data.
    """
    input_file = 'data/dataset/train/train_features.h5'

    if os.path.exists(input_file):
        actions = read_from_hdf5(input_file)
        
        for index, action in enumerate(actions):
            logging.info(f"Action ID: {index}")
            logging.info(f"Body part: {action['bodypart']}")
            for clip in action['clips']:
                camera_angle = clip['Camera type']
                video_features = clip['video_features']
                
        print_field_occurrences(actions)
    else:
        logging.error(f"File not found: {input_file}")

if __name__ == "__main__":
    main()
