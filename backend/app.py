from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from inference import InferencePipeline
import logging
from typing import Dict, Any, List, Optional
import uvicorn
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exception models
class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

# Response models
class PredictionDetail(BaseModel):
    prediction: str
    probability: float

class CategoryResult(BaseModel):
    category: str
    details: List[PredictionDetail]

class PredictionResponse(BaseModel):
    status: str
    predictions: List[CategoryResult]
    ai_decision: Optional[str] = None
    explanation: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str

# Initialize FastAPI
app = FastAPI(
    title="Video Analysis API",
    description="API for analyzing videos and making predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_video_file(file: UploadFile) -> None:
    """Validate the uploaded video file"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")
        
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # FastAPI doesn't check file size during upload, we'll check it after saving

def format_predictions(raw_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format predictions from the model into the API response format."""
    try:
        formatted_predictions = []
        for pred in raw_predictions:
            category_result = CategoryResult(
                category=pred['category'],
                details=[
                    PredictionDetail(
                        prediction=str(pred['prediction']),
                        probability=float(pred['probability'])
                    )
                ]
            )
            formatted_predictions.append(category_result)
            
            # Log predictions in a readable format
            confidence = "▓" * int(pred['probability'] * 20) + "░" * (20 - int(pred['probability'] * 20))
            logger.info(f"\n{pred['category']}:")
            logger.info(f"  • {pred['prediction']:<20} [{confidence}] {pred['probability']:.1%}")
                
        return {"predictions": formatted_predictions}
        
    except Exception as e:
        logger.error(f"Error formatting predictions: {str(e)}")
        logger.debug(f"Raw predictions: {raw_predictions}")
        raise

async def process_video(video_file: UploadFile) -> tuple[str, Dict[str, Any]]:
    """Process the video file and return predictions"""
    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video_file.read()
            
            # Check file size
            if len(content) > MAX_CONTENT_LENGTH:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024 * 1024)}MB"
                )
                
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            logger.info("Processing video for inference...")
            features_path = pipeline.process_video_for_inference(temp_path, replay_speed=1.0)
            
            logger.info("Running inference...")
            raw_predictions = pipeline.inference(features_path)
            
            formatted_predictions = format_predictions(raw_predictions)
            logger.info("Inference completed successfully!")
            
            return features_path, formatted_predictions
            
        except Exception as e:
            raise VideoProcessingError(f"Error processing video: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving video file: {str(e)}")
    finally:
        # In FastAPI, we don't delete the file here, we'll use BackgroundTasks instead
        pass

def cleanup_files(*files: str) -> None:
    """Clean up temporary files"""
    for file_path in files:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")

# Initialize pipeline with error handling
try:
    pipeline = InferencePipeline(
        model_path="./backend/pretrained_models/20250129_195031/foul_detection_model.pth",
        base_dir='./backend/data/dataset/'
    )
except Exception as e:
    logger.critical(f"Failed to initialize inference pipeline: {str(e)}")
    raise

@app.post(
    "/api/inference", 
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse},
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Inference"]
)
async def process_video_endpoint(
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Process a video file and return predictions with referee decision explanation."""
    temp_path = None
    features_path = None
    
    try:
        # Validate request
        validate_video_file(video)
        
        # Process video
        features_path, predictions = await process_video(video)
        
        # Generate explanation from predictions
        explanation_result = {"decision": "No Card", "explanation": "No explanation available"}
        try:
            from explanation import generate_explanation
            explanation_result = generate_explanation(predictions["predictions"])
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_files, temp_path, features_path)
        
        return {
            "status": "success",
            "predictions": predictions["predictions"],
            "ai_decision": explanation_result.get("decision", "Unknown"),
            "explanation": explanation_result.get("explanation", "No explanation available")
        }
        
    except HTTPException as e:
        logger.warning(f"API Error: {e.detail}")
        raise
        
    except VideoProcessingError as e:
        logger.error(f"Video Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing video")
        
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
        
    finally:
        # If background_tasks wasn't available, clean up manually
        if not background_tasks and (temp_path or features_path):
            cleanup_files(temp_path, features_path)

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)