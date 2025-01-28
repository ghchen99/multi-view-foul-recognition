import os
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.FoulDataPreprocessor import FoulDataPreprocessor
from utils.training import load_model
from models.Decoder import Decoder

class TestSetInference:
    """Class for running inference on the full test set."""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = FoulDataPreprocessor()
        self.decoder = Decoder()
        self.model = self._load_model(model_path)
        
        # Create reverse mappings for converting string labels to numbers
        self.reverse_mappings = {
            'Action Class': {v.decode() if isinstance(v, bytes) else v: k 
                           for k, v in self.preprocessor.action_class_map.items()},
            'Body Part': {v.decode() if isinstance(v, bytes) else v: k 
                         for k, v in self.preprocessor.bodypart_map.items()},
            'Offence': {v.decode() if isinstance(v, bytes) else v: k 
                       for k, v in self.preprocessor.offence_map.items()},
            'Touchball': {v.decode() if isinstance(v, bytes) else v: k 
                         for k, v in self.preprocessor.touchball_map.items()},
            'Try to Play': {v.decode() if isinstance(v, bytes) else v: k 
                           for k, v in self.preprocessor.trytoplay_map.items()},
            'Severity': {v.decode() if isinstance(v, bytes) else v: k 
                        for k, v in self.preprocessor.severity_map.items()}
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = load_model(model_path)
        model.to(self.device)
        model.eval()
        return model

    def _convert_predictions_to_numeric(self, predictions: List[str], category: str) -> np.ndarray:
        """Convert string predictions to numeric values using mappings."""
        mapping_dict = {
            'Action Class': self.preprocessor.action_class_map,
            'Body Part': self.preprocessor.bodypart_map,
            'Offence': self.preprocessor.offence_map,
            'Touchball': self.preprocessor.touchball_map,
            'Try to Play': self.preprocessor.trytoplay_map,
            'Severity': self.preprocessor.severity_map
        }
        category_map = mapping_dict[category]
        
        numeric_predictions = []
        for pred in predictions:
            # Handle both string and bytes predictions
            if isinstance(pred, bytes):
                pred = pred.decode()
            
            # Special handling for severity labels
            if category == 'Severity':
                # Extract just the numeric part (e.g., "1.0" from "1.0 No card")
                severity_value = pred.split()[0] if isinstance(pred, str) else pred.split()[0].decode()
                severity_key = severity_value.encode() if isinstance(severity_value, str) else severity_value
            else:
                severity_key = pred.encode() if isinstance(pred, str) else pred
            
            numeric_predictions.append(category_map[severity_key])
        
        return np.array(numeric_predictions)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, category: str) -> Dict:
        """Calculate accuracy metrics for a given category."""
        # Convert string predictions to numeric if needed
        if isinstance(y_pred[0], (str, bytes)):
            y_pred = self._convert_predictions_to_numeric(y_pred, category)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred)
        
        metrics = {
            'category': category,
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return metrics

    def process_test_set(self, test_file: str) -> Tuple[Dict, Dict]:
        """Process the entire test set and return predictions and metrics."""
        logging.info(f"Processing test set from: {test_file}")
        
        try:
            # Process the test data
            processed_data = self.preprocessor.process_data(test_file)
            if processed_data is None:
                raise ValueError("Failed to process test data")
                
            X_test, y_test = processed_data
            if X_test is None:
                raise ValueError("X_test is None after preprocessing")
                
            logging.info(f"Test set shape: {X_test.shape}")
            
            # Move data to device
            X_test = X_test.to(self.device)
            
            # Initialize dictionaries for predictions and metrics
            predictions_dict = {
                'Action Class': [],
                'Body Part': [],
                'Offence': [],
                'Touchball': [],
                'Try to Play': [],
                'Severity': [],
                'Probabilities': []
            }
            
            # Process in batches
            batch_size = 64
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    batch = X_test[i:i + batch_size]
                    batch_outputs = self.model(batch)
                    batch_decoded = self.decoder.decode_predictions(*batch_outputs)
                    
                    # Add batch predictions to appropriate lists
                    for category in batch_decoded:
                        predictions_dict[category['category']].extend(category['predictions'])
                        if category['category'] == 'Action Class':
                            predictions_dict['Probabilities'].extend(
                                category['probabilities'].cpu().numpy().tolist()
                            )
                    
                    if (i + batch_size) % 320 == 0:
                        logging.info(f"Processed {i + batch_size}/{len(X_test)} samples")
            
            # Calculate metrics for each category
            metrics_dict = {}
            category_mapping = {
                'Action Class': 'actionclass',
                'Body Part': 'bodypart',
                'Offence': 'offence',
                'Touchball': 'touchball',
                'Try to Play': 'trytoplay',
                'Severity': 'severity'
            }
            
            for pred_key, true_key in category_mapping.items():
                y_true = y_test[true_key].cpu().numpy()
                y_pred = predictions_dict[pred_key]
                metrics_dict[pred_key] = self.calculate_metrics(y_true, y_pred, pred_key)
                
                # Log results
                logging.info(f"\nMetrics for {pred_key}:")
                logging.info(f"Accuracy: {metrics_dict[pred_key]['accuracy']:.3f}")
                logging.info(f"Weighted Precision: {metrics_dict[pred_key]['weighted_precision']:.3f}")
                logging.info(f"Weighted Recall: {metrics_dict[pred_key]['weighted_recall']:.3f}")
                logging.info(f"Weighted F1: {metrics_dict[pred_key]['weighted_f1']:.3f}")
                
                # Log per-class metrics
                mapping_dict = {
                    'actionclass': self.preprocessor.action_class_map,
                    'bodypart': self.preprocessor.bodypart_map,
                    'offence': self.preprocessor.offence_map,
                    'touchball': self.preprocessor.touchball_map,
                    'trytoplay': self.preprocessor.trytoplay_map,
                    'severity': self.preprocessor.severity_map
                }
                category_map = mapping_dict[true_key]
                reverse_map = {v: k.decode() if isinstance(k, bytes) else k 
                             for k, v in category_map.items()}
                
                for class_idx in range(len(metrics_dict[pred_key]['per_class_precision'])):
                    class_name = reverse_map.get(class_idx, f"Class {class_idx}")
                    if isinstance(class_name, bytes):
                        class_name = class_name.decode()
                    logging.info(f"\nClass: {class_name}")
                    logging.info(f"Precision: {metrics_dict[pred_key]['per_class_precision'][class_idx]:.3f}")
                    logging.info(f"Recall: {metrics_dict[pred_key]['per_class_recall'][class_idx]:.3f}")
                    logging.info(f"F1: {metrics_dict[pred_key]['per_class_f1'][class_idx]:.3f}")
            
            return predictions_dict, metrics_dict
            
        except Exception as e:
            logging.error(f"Error during test set inference: {str(e)}")
            raise

    def save_predictions(self, predictions: Dict, metrics: Dict, output_path: str):
        """Save predictions and metrics to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Save predictions
        df_predictions = pd.DataFrame({
            'Action Class': predictions['Action Class'],
            'Body Part': predictions['Body Part'],
            'Offence': predictions['Offence'],
            'Touchball': predictions['Touchball'],
            'Try to Play': predictions['Try to Play'],
            'Severity': predictions['Severity'],
            'Confidence': predictions['Probabilities']
        })
        
        predictions_file = output_dir / 'test_predictions.csv'
        df_predictions.to_csv(predictions_file, index=False)
        logging.info(f"Saved predictions to: {predictions_file}")
        
        # Save metrics
        metrics_file = output_dir / 'test_metrics.json'
        pd.DataFrame(metrics).to_json(metrics_file, orient='index', indent=2)
        logging.info(f"Saved metrics to: {metrics_file}")
        
        # Print distribution statistics
        logging.info("\nPrediction Distribution:")
        for column in df_predictions.columns[:-1]:  # Exclude confidence column
            value_counts = df_predictions[column].value_counts()
            logging.info(f"\n{column} distribution:")
            for value, count in value_counts.items():
                percentage = (count / len(df_predictions)) * 100
                logging.info(f"  {value}: {count} ({percentage:.1f}%)")

def main():
    """Run inference on the test set."""
    model_path = "foul_detection_model.pth"
    test_file = "data/dataset/test/test_features.h5"
    output_path = "results"
    
    try:
        # Initialize inference pipeline
        pipeline = TestSetInference(model_path)
        
        # Process test set
        logging.info("Starting test set inference...")
        predictions, metrics = pipeline.process_test_set(test_file)
        
        # Save predictions and metrics
        pipeline.save_predictions(predictions, metrics, output_path)
        
        logging.info("Test set inference completed successfully!")
        
    except Exception as e:
        logging.error(f"Test set inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()