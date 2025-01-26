import os
import logging
from pathlib import Path
import torch
from typing import Dict

from utils.FeatureExtractor import FeatureExtractor
from models.ActionData import ActionData
from utils.HDF5Reader import save_to_hdf5
from utils.FoulDataPreprocessor import FoulDataPreprocessor
from models.Decoder import Decoder
from utils.training import MultiTaskModel, load_model

class FoulInferencePipeline:
    """Pipeline for running inference with the foul detection model."""
    
    def __init__(self, model_path: str, base_dir: str = 'data/dataset/'):
        self.base_dir = Path(base_dir)
        self.preprocessor = FoulDataPreprocessor()
        self.decoder = Decoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.extractor = FeatureExtractor(base_dir=base_dir, model_type='r3d_18')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_model(self, model_path: str) -> MultiTaskModel:
        """Load the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = load_model(model_path)
        model.to(self.device)
        model.eval()
        return model

    def process_video_for_inference(self, video_path: str, replay_speed: float = 1.0) -> str:
        """Process a single video for inference."""
        output_dir = self.base_dir / 'inference'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'inference_features.h5'
        
        logging.info(f"Processing video: {video_path}")
        
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
        features = self.extractor.extract_video_features(video_path)
        if features is None:
            raise ValueError("Failed to extract video features")
            
        # Add clip with features
        dummy_action.clips = [{
            'video_features': features,
            'Replay speed': replay_speed,
            'Camera type': 'Close-up player or field referee'
        }]
        
        logging.info("Features extracted successfully")
        
        # Save features
        save_to_hdf5([dummy_action], str(output_file))
        logging.info(f"Saved features to: {output_file}")
        
        return str(output_file)

    def inference(self, inference_file: str) -> Dict:
        """Run inference on processed video data."""
        logging.info("Starting inference...")
        
        # Add more detailed logging
        if not os.path.exists(inference_file):
            raise FileNotFoundError(f"Inference file not found: {inference_file}")
            
        try:
            processed_data = self.preprocessor.process_data(inference_file)
            if processed_data is None:
                raise ValueError("Preprocessor returned None")
                
            X_test, y_test = processed_data  # Proper unpacking
            if X_test is None:
                raise ValueError("X_test is None after preprocessing")
                
            logging.info(f"Input shape for inference: {X_test.shape}")
            
            with torch.no_grad():
                predictions = self.model(X_test)
                
            # Decode and format predictions
            return self.decoder.decode_predictions(*predictions)
            
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            raise

def main():
    """Run the inference pipeline on a test video."""
    model_path = "foul_detection_model.pth"
    pipeline = FoulInferencePipeline(model_path)
    
    try:
        # Process and run inference on a test video
        video_path = 'data/dataset/inference/testaction5_clip1.mp4'
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Test video not found: {video_path}")
            
        logging.info("Processing video for inference...")
        inference_features = pipeline.process_video_for_inference(
            video_path,
            replay_speed=1.4
        )
        
        logging.info("Running inference...")
        predictions = pipeline.inference(inference_features)

        # Print predictions in a nice format
        print("Decoded Predictions with Probabilities:\n")
        for category in predictions:
            print(f"{category['category']}:")
            for pred, prob in zip(category['predictions'], category['probabilities']):
                print(f"  • {pred:<20} {prob:.2%}")
            print()
            
        logging.info("Inference pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Inference pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()