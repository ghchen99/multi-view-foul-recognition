import os
import logging
from feature_engineering.HDF5Reader import read_from_hdf5

def resahape_action():
    input_file = 'data/dataset/train/train_features.h5'

    if os.path.exists(input_file):
        actions = read_from_hdf5(input_file)
        
        for action in actions:
            actionclass = action['actionclass'] # type bytes
            bodypart = action['bodypart'] # type bytes
            offence = action['offence'] # type bytes
            severity = action['severity'] # type bytes
            touchball = action['touchball'] # type bytes
            trytoplay = action['trytoplay'] # type bytes
            
            for clip in action['clips']:
                camera_angle = clip['Camera type'] # type bytes
                replay_speed = clip['Replay speed'] # numpy.float64
                video_features = clip['video_features'] # type torch.Tensor (1, 512)
                continue # test
    else:
        logging.error(f"File not found: {input_file}")

def main():
    resahape_action()    

if __name__ == "__main__":
    main()