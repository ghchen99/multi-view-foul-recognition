from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from inference_pipeline import FoulInferencePipeline
import logging
from typing import Tuple, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

class APIError(Exception):
    """Base exception for API errors"""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code

app = Flask(__name__)
CORS(app)

# Configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_video_file(file) -> None:
    """Validate the uploaded video file"""
    if not file:
        raise APIError('No file provided', 400)
    
    if file.filename == '':
        raise APIError('No selected file', 400)
        
    if not allowed_file(file.filename):
        raise APIError(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 400)

def format_predictions(raw_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format predictions from the new model structure into the existing API response format.
    The new model returns a list of dictionaries with category, prediction, and probability.
    """
    try:
        formatted_predictions = []
        for pred in raw_predictions:
            category_result = {
                'category': pred['category'],
                'details': [{
                    'prediction': str(pred['prediction']),
                    'probability': float(pred['probability'])
                }]
            }
            formatted_predictions.append(category_result)
            
            # Log in a nice format
            logger.info(f"\n{pred['category']}:")
            logger.info(f"  • {pred['prediction']:<20} {pred['probability']:.2%}")
                
        return {'predictions': formatted_predictions}
        
    except Exception as e:
        logger.error(f"Error formatting predictions structure: {str(e)}")
        logger.debug(f"Raw predictions structure: {raw_predictions}")
        raise

def process_video(video_file) -> Tuple[str, Dict[str, Any]]:
    """Process the video file and return predictions"""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            logger.info("Processing video for inference...")
            features_path = pipeline.process_video_for_inference(temp_path, replay_speed=1.4)
            
            logger.info("Running inference...")
            raw_predictions = pipeline.inference(features_path)
            
            # Format predictions using the enhanced formatter
            formatted_predictions = format_predictions(raw_predictions)
            
            logger.info("Inference completed successfully!")
            return features_path, formatted_predictions
            
        except Exception as e:
            raise VideoProcessingError(f"Error processing video: {str(e)}")
            
    except Exception as e:
        raise APIError(f"Error saving video file: {str(e)}", 500)
    finally:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

def cleanup_files(*files: str) -> None:
    """Clean up temporary files"""
    for file_path in files:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")

# Initialize pipeline with error handling
try:
    pipeline = FoulInferencePipeline(model_path="pretrained_models/20250128_223835/foul_detection_model.pth")
except Exception as e:
    logger.critical(f"Failed to initialize inference pipeline: {str(e)}")
    raise

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': f'File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024 * 1024)}MB'
    }), 413

@app.route('/api/inference', methods=['POST'])
def process_video_endpoint():
    temp_path = None
    features_path = None
    
    try:
        # Validate request
        if not request.files:
            raise APIError('No file part in the request', 400)
            
        video_file = request.files.get('video')
        validate_video_file(video_file)
        
        # Process video
        features_path, predictions = process_video(video_file)
        
        return jsonify({
            'status': 'success',
            **predictions  # Unpack formatted predictions
        })
        
    except APIError as e:
        logger.warning(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), e.status_code
        
    except VideoProcessingError as e:
        logger.error(f"Video Processing Error: {str(e)}")
        return jsonify({'error': 'Error processing video'}), 500
        
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500
        
    finally:
        # Clean up temporary files in all cases
        cleanup_files(temp_path, features_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)