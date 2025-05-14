import os
import logging
import torch
from utils.HDF5Reader import read_from_hdf5

class FoulDataPreprocessor:
    def __init__(self):
        self.action_class_map = {
            b'Standing tackling': 0, b'Tackling': 1, b'Holding': 2,
            b'Challenge': 3, b'Elbowing': 4, b'High leg': 5,
            b'Pushing': 6, b'Dive': 7
        }
        
        self.bodypart_map = {
            b'Under body': 0, b'Use of arms': 1,
            b'Use of shoulder': 2, b'Upper body': 3
        }
        
        self.offence_map = {
            b'Offence': 2, b'No offence': 0, b'Between': 1
        }
        
        self.touchball_map = {
            b'No': 0, b'Yes': 2, b'Maybe': 1
        }
        
        self.trytoplay_map = {
            b'No': 0, b'Yes': 1
        }
        
        self.severity_map = {
            b'1.0': 0,
            b'2.0': 1,
            b'3.0': 2,
            b'4.0': 3,
            b'5.0': 4
        }
        
        self.target_camera = b'Close-up player or field referee'

    def get_class_weights(self, labels, num_classes):
        """Calculate class weights for imbalanced classes."""
        counts = torch.bincount(labels, minlength=num_classes)
        total = len(labels)
        weights = total / (counts * num_classes)
        # Clamp weights to prevent extreme values
        weights = torch.clamp(weights, min=0.1, max=10.0)
        return weights / weights.sum()

    def is_valid_features(self, video_features):
        """Check if video features are valid (non-empty and non-zero)."""
        return (video_features is not None and 
                isinstance(video_features, torch.Tensor) and 
                video_features.numel() > 0 and 
                not torch.all(video_features == 0))

    def encode_labels(self, action):
        encoded = {
            'actionclass': self.action_class_map[action['actionclass']],
            'bodypart': self.bodypart_map[action['bodypart']],
            'offence': self.offence_map[action['offence']],
            'severity': self.severity_map[action['severity']],
            'touchball': self.touchball_map[action['touchball']],
            'trytoplay': self.trytoplay_map[action['trytoplay']]
        }
        return encoded
    
    def process_data(self, input_file):
        """Process and reshape the data for deep learning."""
        if not os.path.exists(input_file):
            logging.error(f"File not found: {input_file}")
            return None
            
        actions = read_from_hdf5(input_file)
        
        features = []
        labels = []
        
        # Track statistics
        total_actions = len(actions)
        processed_actions = 0
        skipped_no_target_camera = 0
        skipped_empty_features = 0
        
        for action_idx, action in enumerate(actions):
            # Track target camera clips
            target_camera_count = 0
            valid_features = []

            # Process clips
            for clip in action['clips']:
                if clip['Camera type'] == self.target_camera:
                    target_camera_count += 1
                    
                    # Skip empty features
                    if not self.is_valid_features(clip['video_features']):
                        skipped_empty_features += 1
                        continue
                    
                    # Extract and combine features
                    video_features = clip['video_features'].squeeze()
                    replay_speed = torch.tensor([float(clip['Replay speed'])])
                    combined_features = torch.cat([video_features, replay_speed])
                    valid_features.append(combined_features)
            
            # Log target camera count
            # logging.info(f"Action {action_idx + 1}/{total_actions}: {target_camera_count} target camera clips.")

            # Check if action is valid
            if not valid_features:
                skipped_no_target_camera += 1
                continue

            processed_actions += 1
            encoded_labels = self.encode_labels(action)
            features.extend(valid_features)
            labels.extend([encoded_labels] * len(valid_features))
        
        # Log statistics
        logging.info(f"\nProcessing Summary:")
        logging.info(f"Total actions: {total_actions}")
        logging.info(f"Processed actions: {processed_actions}")
        logging.info(f"Skipped actions (no target camera): {skipped_no_target_camera}")
        logging.info(f"Skipped clips (empty features): {skipped_empty_features}")
        
        if not features:
            logging.error("No valid features found in the dataset")
            return None
            
        # Stack features and convert labels
        X = torch.stack(features)
        y = {
            'actionclass': torch.tensor([label['actionclass'] for label in labels]),
            'bodypart': torch.tensor([label['bodypart'] for label in labels]),
            'offence': torch.tensor([label['offence'] for label in labels]),
            'severity': torch.tensor([label['severity'] for label in labels]),
            'touchball': torch.tensor([label['touchball'] for label in labels]),
            'trytoplay': torch.tensor([label['trytoplay'] for label in labels])
        }
        
        logging.info(f"Final dataset shape: {X.shape}")
        logging.info(f"Features per action: {len(features) / processed_actions:.2f}")
        return X, y

def main():
    logging.basicConfig(level=logging.INFO)
    preprocessor = FoulDataPreprocessor()
    input_file = 'data/dataset/train/train_features.h5'
    
    X, y = preprocessor.process_data(input_file)
    
    if X is not None:
        # Calculate class weights for each task
        class_weights = {
            'actionclass': preprocessor.get_class_weights(y['actionclass'], len(preprocessor.action_class_map)),
            'bodypart': preprocessor.get_class_weights(y['bodypart'], len(preprocessor.bodypart_map)),
            'offence': preprocessor.get_class_weights(y['offence'], len(preprocessor.offence_map)),
            'touchball': preprocessor.get_class_weights(y['touchball'], len(preprocessor.touchball_map)),
            'trytoplay': preprocessor.get_class_weights(y['trytoplay'], len(preprocessor.trytoplay_map)),
            'severity': preprocessor.get_class_weights(y['severity'], len(preprocessor.severity_map))
        }
        return X, y, class_weights

if __name__ == "__main__":
    main()
