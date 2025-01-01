import os
import h5py
import torch
import logging
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                    video_features_tensor = clip['video_features']
                    video_features_np = video_features_tensor.cpu().numpy()  # Convert to NumPy
                    video_features_group.create_dataset('data', data=video_features_np)  # Save as NumPy array
                    
    logging.info(f"Action data saved to {output_file}")

def print_field_occurrences(actions):
    """
    Prints distinct occurrences and their counts for each field in the actions data.
    Excludes the 'clips' field.
    """
    if not actions:
        logging.warning("No actions provided to analyze")
        return
    
    # Get all fields from the first action (excluding clips)
    fields = [field for field in actions[0].keys() if field != 'clips']
    
    logging.info("Field value distributions:")
    for field in sorted(fields):
        # Count occurrences of each unique value
        value_counts = Counter(str(action[field]) for action in actions)
        
        # Print field name and total count
        logging.info(f"\n{field} - Total entries: {len(actions)}")
        logging.info("-" * 50)
        
        # Print each value and its count, sorted by count in descending order
        for value, count in sorted(value_counts.items(), key=lambda x: (-x[1], x[0])):
            percentage = (count / len(actions)) * 100
            logging.info(f"'{value}': {count} occurrences ({percentage:.1f}%)")


def read_from_hdf5(input_file):
    """
    Reads action data and video features from an HDF5 file.
    Converts video features back to torch tensors.
    """
    with h5py.File(input_file, 'r') as f:
        actions = []
        for action_key in f.keys():
            action_group = f[action_key]

            # Read attributes (non-clip data)
            action_data = {attr: action_group[attr][()] for attr in action_group if attr != 'clips'}

            # Read clips data
            clips = []
            if 'clips' in action_group:
                clips_group = action_group['clips']
                for clip_key in clips_group.keys():
                    clip_group = clips_group[clip_key]
                    clip_data = {key: clip_group[key][()] for key in clip_group if key != 'video_features'}

                    # Read video_features if available
                    video_features = None
                    if 'video_features' in clip_group:
                        features_group = clip_group['video_features']
                        if 'data' in features_group:
                            # Convert NumPy array back to a PyTorch tensor
                            video_features_np = features_group['data'][()]
                            video_features = torch.tensor(video_features_np)

                    clip_data['video_features'] = video_features
                    clips.append(clip_data)
            
            action_data['clips'] = clips
            actions.append(action_data)

        logging.info(f"Read {len(actions)} actions from {input_file}")
        return actions


def main():
    """
    Main function to load and inspect HDF5 data.
    """
    input_file = 'data/dataset/train/train_features.h5'

    if os.path.exists(input_file):
        # Read data
        actions = read_from_hdf5(input_file)

        # Print details of each action and its fields
        for idx, action in enumerate(actions):
            logging.info(f"\nAction {idx + 1}:")
            for key, value in action.items():
                if key == 'clips':
                    logging.info(f"  {key} (List of {len(value)} clips):")
                    for clip_idx, clip in enumerate(value):
                        logging.info(f"    Clip {clip_idx + 1}:")
                        for clip_key, clip_value in clip.items():
                            # Print type and size for video features
                            if clip_key == 'video_features' and clip_value is not None:
                                logging.info(f"      {clip_key}: {type(clip_value)}, shape: {clip_value.shape}")
                            else:
                                logging.info(f"      {clip_key}: {type(clip_value)}, value: {clip_value}")
                else:
                    logging.info(f"  {key}: {type(value)}, value: {value}")

        # Print field occurrences
        print_field_occurrences(actions)

    else:
        logging.error(f"File not found: {input_file}")


if __name__ == "__main__":
    main()
