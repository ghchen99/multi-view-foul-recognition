import logging
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
    
    save_extracted_features([ActionData({'video_features': features, 'replay_speed': replay_speed})], output_file)
    
    logging.info(f"Saved inference features to: {output_file}")

def main() -> None:
    # TODO: Inference script!
    inference_file = 'data/dataset/inference/inference_features.h5'
    video_path = 'data/dataset/inference/test_action_5_1.4_replay_speed.mp4'
    replay_speed = 1.4
    process_inference_video(video_path, replay_speed, inference_file)


if __name__ == "__main__":
    main()
