import os
import h5py
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read data from HDF5 file
def read_from_hdf5(input_file):
    """
    Reads action data and video features from an HDF5 file.
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
                    video_features = {}
                    if 'video_features' in clip_group:
                        features_group = clip_group['video_features']
                        video_features = {key: features_group[key][()] for key in features_group}
                    
                    clip_data['video_features'] = video_features
                    clips.append(clip_data)
            
            action_data['clips'] = clips
            actions.append(action_data)

        logging.info(f"Read {len(actions)} actions from {input_file}")
        return actions

# Example usage
if __name__ == "__main__":
    input_file = 'data/video_features.h5'
    if os.path.exists(input_file):
        actions = read_from_hdf5(input_file)
        for idx, action in enumerate(actions):  # Print details of the first 5 actions
            logging.info(f"Action {idx+1}: {action}")
    else:
        logging.error(f"File not found: {input_file}")
