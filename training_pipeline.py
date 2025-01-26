import os
import logging
from pathlib import Path
import torch
from typing import Optional, Dict

from utils.FeatureExtractor import FeatureExtractor
from models.ActionData import ActionData
from utils.HDF5Reader import save_to_hdf5
from utils.FoulDataPreprocessor import FoulDataPreprocessor
from utils.training import MultiTaskModel, train_model, save_model

class FoulTrainingPipeline:
    """Pipeline for training the foul detection model."""
    
    def __init__(self, base_dir: str = 'data/dataset/', model_type: str = 'r3d_18'):
        self.base_dir = Path(base_dir)
        self.preprocessor = FoulDataPreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor(base_dir=base_dir, model_type=model_type)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def extract_features(self, split: str, max_actions: Optional[int] = None) -> str:
        """Extract features for a specific dataset split."""
        return self.feature_extractor.extract_features(split, max_actions)

    def train(self, train_file: str, valid_file: str, epochs: int = 100, 
             batch_size: int = 64, learning_rate: float = 0.0005) -> MultiTaskModel:
        """Train the model using the specified training and validation data."""
        logging.info("Starting model training...")
        
        # Process training data
        X_train, y_train = self.preprocessor.process_data(train_file)
        
        if X_train is None:
            raise ValueError("Failed to process training data")
            
        # Calculate class weights
        class_weights = self._calculate_class_weights(y_train)
        
        # Train model
        model = train_model(
            X_train=X_train,
            y_train=y_train,
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

def main():
    """Run the training pipeline."""
    pipeline = FoulTrainingPipeline()
    
    try:
        # 1. Extract features for training and validation
        logging.info("Extracting features...")
        train_features = pipeline.extract_features('train', max_actions=100)
        valid_features = pipeline.extract_features('valid', max_actions=20)
        
        # 2. Train model
        logging.info("Training model...")
        model = pipeline.train(
            train_file=train_features,
            valid_file=valid_features,
            epochs=100,
            batch_size=64,
            learning_rate=0.0005
        )
        
        logging.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()