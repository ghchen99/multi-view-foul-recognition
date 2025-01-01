import os

import logging
from src.HDF5Reader import read_from_hdf5

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
    else:
        logging.error(f"File not found: {input_file}")

if __name__ == "__main__":
    main()