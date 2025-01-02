import logging
import torch
from train import load_model
from training.Decoder import Decoder
from training.FoulDataPreprocessor import FoulDataPreprocessor
from feature_engineering.FeatureExtractor import FeatureExtractor
from feature_engineering.ActionData import ActionData
from feature_engineering.HDF5Reader import save_to_hdf5

def save_extracted_features(actions: list, output_file: str) -> None:
    """
    Saves the extracted actions into an HDF5 file.
    """
    save_to_hdf5(actions, output_file)
    logging.info(f"Done: Extracted features for {len(actions)} actions.")
    
def process_inference_video(video_path: str, replay_speed: float, output_file: str) -> None:
    """
    Processes a single video for inference and saves the features to an HDF5 file.
    """
    logging.info(f"Processing inference video: {video_path}")
    
    extractor = FeatureExtractor(model_type='r3d_18', device='cpu')
    features = extractor.extract_features(video_path)
    
    save_extracted_features([ActionData({'Offence': 'Offence', 
                                         'Bodypart': 'Under body',
                                         'Action class': 'Tackling',
                                         'Touch ball': 'No',
                                         'Try to play': 'Yes',
                                         'Severity': '1.0',
                                         'Clips': [
                                                {
                                                    'video_features': features,
                                                    'Replay speed': replay_speed,
                                                    'Camera type': 'Close-up player or field referee'
                                                }
                                         ]})], output_file)
    
    logging.info(f"Saved inference features to: {output_file}")

def main() -> None:

    inference_file = 'data/dataset/inference/inference_features.h5'
    video_path = 'data/dataset/inference/test_action_5_1.4_replay_speed.mp4'
    replay_speed = 1.4
    process_inference_video(video_path, replay_speed, inference_file)
    
    model= load_model("foul_detection_model.pth")
    
    input_file = 'data/dataset/inference/inference_features.h5'
    preprocessor = FoulDataPreprocessor()
    X_test, _ = preprocessor.process_data(input_file)

    model.eval()

    with torch.no_grad():  # No need to compute gradients during inference
        actionclass_pred, bodypart_pred, offence_pred, touchball_pred, trytoplay_pred, severity_pred = model(X_test)

    # Get predicted classes and probabilities
    def get_predictions_and_probs(predictions):
        probs = torch.softmax(predictions, dim=1)
        labels = torch.argmax(probs, dim=1)
        max_probs = probs.max(dim=1).values
        return labels, max_probs

    actionclass_pred_labels, actionclass_probs = get_predictions_and_probs(actionclass_pred)
    bodypart_pred_labels, bodypart_probs = get_predictions_and_probs(bodypart_pred)
    offence_pred_labels, offence_probs = get_predictions_and_probs(offence_pred)
    touchball_pred_labels, touchball_probs = get_predictions_and_probs(touchball_pred)
    trytoplay_pred_labels, trytoplay_probs = get_predictions_and_probs(trytoplay_pred)
    severity_pred_labels, severity_probs = get_predictions_and_probs(severity_pred)

    # Example usage
    decoder = Decoder()

    # Decode and print results
    decoder.decode_predictions(
        actionclass_pred_labels,
        bodypart_pred_labels,
        offence_pred_labels,
        touchball_pred_labels,
        trytoplay_pred_labels,
        severity_pred_labels
    )

    # Print probabilities for each prediction
    print("Probabilities:")
    print(f"Action class: {actionclass_probs}")
    print(f"Body part: {bodypart_probs}")
    print(f"Offence: {offence_probs}")
    print(f"Touch ball: {touchball_probs}")
    print(f"Try to play: {trytoplay_probs}")
    print(f"Severity: {severity_probs}")


if __name__ == "__main__":
    main()
