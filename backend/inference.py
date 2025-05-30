import os
import logging
from pathlib import Path
import torch
from typing import Dict, Tuple
import torch.nn.functional as F

from utils.FeatureExtractor import FeatureExtractor
from models.ActionData import ActionData
from utils.HDF5Reader import save_to_hdf5
from utils.FoulDataPreprocessor import FoulDataPreprocessor
from models.Decoder import Decoder
from utils.MultiTaskModel import ImprovedMultiTaskModel, load_model

class InferencePipeline:
    """Pipeline for running inference with the improved foul detection model."""
    
    def __init__(self, model_path: str, base_dir: str = 'backend/data/dataset/'):
        self.base_dir = Path(base_dir)
        self.preprocessor = FoulDataPreprocessor()
        self.decoder = Decoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.metadata, self.history = self._load_model(model_path)
        self.extractor = FeatureExtractor(base_dir=base_dir, model_type='r3d_18')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logging.info(f"Model loaded successfully. Device: {self.device}")

    def _load_model(self, model_path: str) -> Tuple[ImprovedMultiTaskModel, dict, dict]:
        """Load the trained model and its components."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model, metadata, history = load_model(model_path, self.device)
        model.eval()
        
        logging.info("Model architecture:")
        for task, head in model.primary_heads.items():
            out_features = head.net[-1].out_features
            logging.info(f"- {task}: {out_features} classes")
        logging.info(f"- severity: {model.severity_head[-1].out_features} classes")
        
        return model, metadata, history

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
        
        if not os.path.exists(inference_file):
            raise FileNotFoundError(f"Inference file not found: {inference_file}")
            
        try:
            # Process data
            processed_data = self.preprocessor.process_data(inference_file)
            if processed_data is None:
                raise ValueError("Preprocessor returned None")
                
            X_test, _ = processed_data
            if X_test is None:
                raise ValueError("X_test is None after preprocessing")
            
            # Convert to tensor and move to device
            X_test = torch.FloatTensor(X_test).to(self.device)
            logging.info(f"Input shape for inference: {X_test.shape}")
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(X_test)
                
                predictions = {
                    task: F.softmax(output, dim=1).cpu().numpy()
                    for task, output in outputs.items()
                }
            
            # Decode predictions
            decoded_predictions = []
            
            # Map task names to decoder maps
            task_to_map = {
                'actionclass': self.decoder.action_class_map,
                'bodypart': self.decoder.bodypart_map,
                'offence': self.decoder.offence_map,
                'touchball': self.decoder.touchball_map,
                'trytoplay': self.decoder.trytoplay_map,
                'severity': self.decoder.severity_map
            }
            
            # Nice display names for tasks
            display_names = {
                'actionclass': 'Action Class',
                'bodypart': 'Body Part',
                'offence': 'Offence',
                'touchball': 'Touch Ball',
                'trytoplay': 'Try to Play',
                'severity': 'Severity'
            }
            
            for task, probs in predictions.items():
                # Get only the top probability
                top_prob, top_idx = torch.max(torch.from_numpy(probs[0]), dim=0)
                
                # Get the mapping for this task
                task_map = task_to_map[task]
                
                # Decode prediction using the appropriate map
                decoded_class = task_map[top_idx.item()].decode('utf-8')
                
                category_pred = {
                    'category': display_names[task],
                    'prediction': decoded_class,
                    'probability': top_prob.item()
                }
                decoded_predictions.append(category_pred)
            
            return decoded_predictions
            
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            raise

def main():
    """Run the inference pipeline on a test video."""
    model_path = "./backend/pretrained_models/20250129_195031/foul_detection_model.pth"
    pipeline = InferencePipeline(model_path)
    
    try:
        # Process and run inference on a test video
        video_path = 'backend/data/dataset/inference/testaction5_clip1.mp4'
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
        print("\nDecoded Predictions with Probabilities:")
        print("=" * 50)
        for category in predictions:
            confidence = "▓" * int(category['probability'] * 20) + "░" * (20 - int(category['probability'] * 20))
            print(f"\n{category['category'].upper()}:")
            print(f"  • {category['prediction']:<20} [{confidence}] {category['probability']:.1%}")
        
        logging.info("Inference pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Inference pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()