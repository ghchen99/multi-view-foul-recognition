# pipeline.py

import os
import logging
from pathlib import Path
import torch
from typing import Optional, Dict

from feature_engineering.FeatureExtractor import FeatureExtractor
from feature_engineering.ActionData import ActionData
from feature_engineering.HDF5Reader import save_to_hdf5
from training.FoulDataPreprocessor import FoulDataPreprocessor
from training.Decoder import Decoder
from train import MultiTaskModel, train_model, save_model

class FoulDetectionPipeline:
    """Main pipeline for training and inference in the foul detection system."""
    
    def __init__(self, base_dir: str = 'data/dataset/'):
        self.base_dir = Path(base_dir)
        self.preprocessor = FoulDataPreprocessor()
        self.decoder = Decoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def extract_features(self, split: str, max_actions: Optional[int] = None) -> str:
        """Extract features for a specific dataset split."""
        input_file = self.base_dir / split / 'annotations.json'
        output_file = self.base_dir / split / f'{split}_features.h5'
        
        if not input_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {input_file}")
            
        # Load annotations
        with open(input_file, 'r') as f:
            import json
            annotations = json.load(f)
            
        logging.info(f"Dataset Set: {annotations['Set']}")
        logging.info(f"Total Actions: {annotations['Number of actions']}")
        
        # Process annotations
        actions = []
        action_count = 0
        
        for action_id, action_data in annotations['Actions'].items():
            if max_actions and action_count >= max_actions:
                break
                
            logging.info(f"Processing Action ID: {action_id}")
            action = ActionData(action_data)
            
            if action.valid:
                # Extract features for each clip
                for clip in action.clips:
                    video_path = os.path.join('data', clip['Url'].lower() + '.mp4')
                    if os.path.exists(video_path):
                        try:
                            extractor = FeatureExtractor(model_type='r3d_18', device=self.device)
                            clip['video_features'] = extractor.extract_features(video_path)
                        except Exception as e:
                            logging.error(f"Error extracting features for {video_path}: {str(e)}")
                            clip['video_features'] = None
                    else:
                        logging.error(f"Video file not found: {video_path}")
                        clip['video_features'] = None
                
                actions.append(action)
            else:
                logging.info(f"Skipped Action ID: {action_id}")
            
            action_count += 1
        
        # Save extracted features
        save_to_hdf5(actions, str(output_file))
        logging.info(f"Saved features to {output_file}")
        
        return str(output_file)

    def process_video_for_inference(self, video_path: str, replay_speed: float = 1.0) -> str:
        """Process a single video for inference."""
        output_dir = self.base_dir / 'inference'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'inference_features.h5'
        
        # Create a dummy action with default values for preprocessing
        dummy_action = ActionData({
            'Offence': 'Offence',
            'Bodypart': 'Under body',  
            'Action class': 'Tackling',
            'Touch ball': 'No',
            'Try to play': 'Yes',
            'Severity': '1.0',
            'Clips': []
        })
        
        # Extract features
        extractor = FeatureExtractor(model_type='r3d_18', device=self.device)
        features = extractor.extract_features(video_path)
        
        # Add clip with features
        dummy_action.clips = [{
            'video_features': features,
            'Replay speed': replay_speed,
            'Camera type': 'Close-up player or field referee'
        }]
        
        # Save features
        save_to_hdf5([dummy_action], str(output_file))
        return str(output_file)

    def train(self, train_file: str, valid_file: str, epochs: int = 100, 
             batch_size: int = 64, learning_rate: float = 0.0005) -> MultiTaskModel:
        """Train the model using the specified training and validation data."""
        logging.info("Starting model training...")
        
        # Process training data
        X_train, y_train = self.preprocessor.process_data(train_file)
        # X_valid, y_valid = self.preprocessor.process_data(valid_file)
        
        # if X_train is None or X_valid is None:
        if X_train is None:
            raise ValueError("Failed to process training or validation data")
            
        # Calculate class weights
        class_weights = self._calculate_class_weights(y_train)
        
        # Train model
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            # X_valid=X_valid, not implemented yet
            # y_valid=y_valid,
            class_weights=class_weights,
            severity_classes=len(self.preprocessor.severity_map),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save model with metadata
        metadata = {
            'input_size': X_train.shape[1],
            'action_classes': len(class_weights['actionclass']),
            'bodypart_classes': len(class_weights['bodypart']),
            'offence_classes': len(class_weights['offence']),
            'touchball_classes': len(class_weights['touchball']),
            'trytoplay_classes': len(class_weights['trytoplay']),
            'severity_classes': len(class_weights['severity'])
        }
        
        save_model(model, "foul_detection_model.pth", metadata)
        return model

    def _calculate_class_weights(self, y_train: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate class weights for all tasks."""
        return {
            'actionclass': self.preprocessor.get_class_weights(
                y_train['actionclass'], len(self.preprocessor.action_class_map)),
            'bodypart': self.preprocessor.get_class_weights(
                y_train['bodypart'], len(self.preprocessor.bodypart_map)),
            'offence': self.preprocessor.get_class_weights(
                y_train['offence'], len(self.preprocessor.offence_map)),
            'touchball': self.preprocessor.get_class_weights(
                y_train['touchball'], len(self.preprocessor.touchball_map)),
            'trytoplay': self.preprocessor.get_class_weights(
                y_train['trytoplay'], len(self.preprocessor.trytoplay_map)),
            'severity': self.preprocessor.get_class_weights(
                y_train['severity'], len(self.preprocessor.severity_map))
        }

    def inference(self, model: MultiTaskModel, inference_file: str) -> Dict:
        """Run inference on processed video data."""
        logging.info("Starting inference...")
        
        X_test, _ = self.preprocessor.process_data(inference_file)
        if X_test is None:
            raise ValueError("Failed to process inference data")
            
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            
        # Decode and format predictions
        return self.decoder.decode_predictions(*predictions)

def main():
    """Run the complete pipeline: feature extraction, training, and inference."""
    pipeline = FoulDetectionPipeline()
    
    try:
        # 1. Extract features for all splits
        logging.info("Extracting features...")
        # train_features = pipeline.extract_features('train')
        # valid_features = pipeline.extract_features('valid')
        # test_features = pipeline.extract_features('test')
        train_features = 'data/dataset/train/train_features.h5'
        valid_features = 'data/dataset/valid/valid_features.h5'
        test_features = 'data/dataset/test/test_features.h5'
        
        # 2. Train model
        logging.info("Training model...")
        model = pipeline.train(
            train_file=train_features,
            valid_file=valid_features,
            epochs=100,
            batch_size=64,
            learning_rate=0.0005
        )
        
        # 3. Run inference on test set
        logging.info("Running inference on test set...")
        pipeline.inference(model, test_features)
        
        # 4. Optional: Run inference on a single video
        video_path = 'data/dataset/inference/testaction5_clip1.mp4'
        if os.path.exists(video_path):
            logging.info("Running inference on single video...")
            inference_features = pipeline.process_video_for_inference(
                video_path,
                replay_speed=1.4
            )
            pipeline.inference(model, inference_features)
            
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()