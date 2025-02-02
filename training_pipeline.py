import os
import logging
from pathlib import Path
import torch
from typing import Optional, Dict
from datetime import datetime

from utils.FeatureExtractor import FeatureExtractor
from utils.FoulDataPreprocessor import FoulDataPreprocessor
from utils.MultiTaskModel import ImprovedMultiTaskModel, train_model, save_model

class TrainingPipeline:
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
              batch_size: int = 64, learning_rate: float = 0.0005) -> ImprovedMultiTaskModel:
        """Train the model using the specified training and validation data."""
        logging.info("Starting model training...")
        
        # Process training and validation data
        X_train, y_train = self.preprocessor.process_data(train_file)
        X_val, y_val = self.preprocessor.process_data(valid_file)
        
        if X_train is None or X_val is None:
            raise ValueError("Failed to process training or validation data")
        
        # Train model with simplified approach
        model, history = train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            severity_classes=len(self.preprocessor.severity_map),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Create model save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("pretrained_models") / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model with simplified components
        model_path = save_dir / "foul_detection_model.pth"
        save_model(
            model=model,
            file_path=str(model_path),
            training_history=history
        )
        
        # Save additional metadata for reference
        self._save_mapping_info(save_dir)
        
        logging.info(f"Model and metadata saved to {save_dir}")
        return model
    
    def _save_mapping_info(self, save_dir: Path) -> None:
        """Save label mapping information."""
        import json

        def convert_bytes_to_str(mapping):
            """Convert bytes keys/values to strings in a mapping."""
            converted = {}
            for k, v in mapping.items():
                key = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                value = v.decode('utf-8') if isinstance(v, bytes) else str(v)
                converted[key] = value
            return converted
        
        mapping_info = {
            'action_class_map': convert_bytes_to_str(self.preprocessor.action_class_map),
            'bodypart_map': convert_bytes_to_str(self.preprocessor.bodypart_map),
            'offence_map': convert_bytes_to_str(self.preprocessor.offence_map),
            'touchball_map': convert_bytes_to_str(self.preprocessor.touchball_map),
            'trytoplay_map': convert_bytes_to_str(self.preprocessor.trytoplay_map),
            'severity_map': convert_bytes_to_str(self.preprocessor.severity_map)
        }
        
        with open(save_dir / 'label_mappings.json', 'w') as f:
            json.dump(mapping_info, f, indent=2)

def main():
    """Run the training pipeline."""
    pipeline = TrainingPipeline()
    
    try:
        # Use existing feature files
        # train_features = pipeline.extract_features('train', max_actions=None)
        # valid_features = pipeline.extract_features('valid', max_actions=None)
        
        train_features = 'data/dataset/train/train_features.h5'
        valid_features = 'data/dataset/valid/valid_features.h5'
        
        # Train model
        logging.info("Training model...")
        model = pipeline.train(
            train_file=train_features,
            valid_file=valid_features,
            epochs=150,  
            batch_size=128,  
            learning_rate=0.0002
        )
        
        logging.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
