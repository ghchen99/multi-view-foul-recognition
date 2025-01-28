import os
import logging
from pathlib import Path
import torch
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

from utils.FeatureExtractor import FeatureExtractor
from utils.FoulDataPreprocessor import FoulDataPreprocessor
from utils.training import ImprovedMultiTaskModel, load_model

class FoulTestPipeline:
    """Pipeline for testing the foul detection model on a test dataset."""
    
    def __init__(self, model_path: str, base_dir: str = 'data/dataset/'):
        self.base_dir = Path(base_dir)
        self.preprocessor = FoulDataPreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.metadata, self.class_weights, _, self.scaler = self._load_model(model_path)
        self.feature_extractor = FeatureExtractor(base_dir=base_dir, model_type='r3d_18')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logging.info(f"Model loaded successfully. Device: {self.device}")
        if self.scaler:
            logging.info("Input scaler loaded")

    def _load_model(self, model_path: str) -> Tuple[ImprovedMultiTaskModel, dict, dict, dict, object]:
        """Load the trained model and its components."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model, metadata, class_weights, history, scaler = load_model(model_path, self.device)
        model.eval()
        
        logging.info("Model architecture:")
        for task, head in model.primary_heads.items():
            out_features = head.net[-1].out_features
            logging.info(f"- {task}: {out_features} classes")
        logging.info(f"- severity: {model.severity_head[-1].out_features} classes")
        
        return model, metadata, class_weights, history, scaler

    def extract_features(self, split: str, max_actions: Optional[int] = None) -> str:
        """Extract features for the test dataset split."""
        return self.feature_extractor.extract_features(split, max_actions)

    def evaluate(self, test_file: str) -> Dict:
        """Run evaluation on the test dataset and compute metrics."""
        logging.info("Starting test evaluation...")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
            
        try:
            # Process test data
            X_test, y_test = self.preprocessor.process_data(test_file)
            if X_test is None or y_test is None:
                raise ValueError("Preprocessor returned None")
                
            # Apply scaler if available
            if self.scaler:
                X_test = self.scaler.transform(X_test)
            
            # Convert to tensor and move to device
            X_test = torch.FloatTensor(X_test).to(self.device)
            logging.info(f"Input shape for testing: {X_test.shape}")
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(X_test)
                predictions = {
                    task: F.softmax(output, dim=1).cpu().numpy()
                    for task, output in outputs.items()
                }
            
            # Compute metrics for each task
            metrics = {}
            task_labels = {
                'actionclass': 'Action Class',
                'bodypart': 'Body Part',
                'offence': 'Offence',
                'touchball': 'Touch Ball',
                'trytoplay': 'Try to Play',
                'severity': 'Severity'
            }
            
            for task, probs in predictions.items():
                # Get predicted classes
                pred_classes = np.argmax(probs, axis=1)
                true_classes = y_test[task].cpu().numpy()
                
                # Generate classification report
                task_report = classification_report(
                    true_classes, 
                    pred_classes,
                    output_dict=True
                )
                
                # Generate confusion matrix
                conf_matrix = confusion_matrix(true_classes, pred_classes)
                
                metrics[task_labels[task]] = {
                    'classification_report': task_report,
                    'confusion_matrix': conf_matrix,
                    'accuracy': task_report['accuracy']
                }
            
            # Save detailed results
            self._save_test_results(metrics)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error during test evaluation: {str(e)}")
            raise

    def _save_test_results(self, metrics: Dict) -> None:
        """Save detailed test results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("test_results") / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'test_metrics.txt', 'w') as f:
            f.write("Test Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            for task, task_metrics in metrics.items():
                f.write(f"\n{task.upper()}\n")
                f.write("-" * len(task) + "\n\n")
                
                # Write accuracy
                f.write(f"Accuracy: {task_metrics['accuracy']:.4f}\n\n")
                
                # Write classification report
                f.write("Classification Report:\n")
                report = task_metrics['classification_report']
                for label, values in report.items():
                    if isinstance(values, dict):
                        f.write(f"\nClass {label}:\n")
                        for metric, value in values.items():
                            f.write(f"  {metric}: {value:.4f}\n")
                
                # Write confusion matrix
                f.write("\nConfusion Matrix:\n")
                np.savetxt(f, task_metrics['confusion_matrix'], fmt='%d')
                f.write("\n" + "=" * 50 + "\n")
        
        logging.info(f"Test results saved to {results_dir}")

def main():
    """Run the test pipeline."""
    model_path = "pretrained_models/20250128_223835/foul_detection_model.pth"
    pipeline = FoulTestPipeline(model_path)
    
    try:
        # Extract features for test set
        logging.info("Extracting test features...")
        # test_features = pipeline.extract_features('test', max_actions=None)
        
        test_features = 'data/dataset/test/test_features.h5'
        
        # Run evaluation
        logging.info("Running test evaluation...")
        metrics = pipeline.evaluate(test_features)
        
        # Print summary metrics
        print("\nTest Results Summary:")
        print("=" * 50)
        for task, task_metrics in metrics.items():
            print(f"\n{task}:")
            print(f"Accuracy: {task_metrics['accuracy']:.2%}")
        
        logging.info("Test pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Test pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()