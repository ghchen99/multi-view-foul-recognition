import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualise_features(file_path):
    """
    Visualises the video features saved in the specified .npy file with enhanced aesthetics,
    accounting for empty features.

    Args:
    file_path (str): Path to the .npy file containing the saved features.
    """
    # Load the features
    features = np.load(file_path, allow_pickle=True)

    # Check if features list is empty
    if len(features) == 0:
        print("No features available for visualization.")
        return

    # Filter out empty features
    valid_features = [f for f in features if f]  # Exclude empty dictionaries
    
    if len(valid_features) == 0:  # If no valid features remain
        print("No valid features available for visualization.")
        return

    # Extract features into separate arrays for visualization
    mean_motion = [f.get('mean_motion', 0) for f in valid_features]
    std_motion = [f.get('std_motion', 0) for f in valid_features]
    max_motion = [f.get('max_motion', 0) for f in valid_features]

    # Use Seaborn style for better aesthetics
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 6))

    # Plot Mean Motion
    plt.subplot(1, 3, 1)
    sns.lineplot(data=mean_motion, marker='o', color='blue', linewidth=2.5)
    plt.title('Mean Motion Intensity', fontsize=14, fontweight='bold')
    plt.xlabel('Clip Index', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Standard Deviation of Motion
    plt.subplot(1, 3, 2)
    sns.lineplot(data=std_motion, marker='o', color='green', linewidth=2.5)
    plt.title('Standard Deviation of Motion', fontsize=14, fontweight='bold')
    plt.xlabel('Clip Index', fontsize=12)
    plt.ylabel('Std Intensity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Max Motion
    plt.subplot(1, 3, 3)
    sns.lineplot(data=max_motion, marker='o', color='red', linewidth=2.5)
    plt.title('Max Motion Intensity', fontsize=14, fontweight='bold')
    plt.xlabel('Clip Index', fontsize=12)
    plt.ylabel('Max Intensity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle('Video Motion Features Visualization', fontsize=16, fontweight='bold', y=1.02)
    plt.show()
    
visualise_features('data/video_features.npy')
