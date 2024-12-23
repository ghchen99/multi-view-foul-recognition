import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# Load annotation file
with open('data/dataset/test/annotations.json', 'r') as f:
    annotations = json.load(f)

# Extract general dataset information
print("Dataset Set:", annotations['Set'])
print("Total Actions:", annotations['Number of actions'])

actions = annotations['Actions']

for action_id, action_data in actions.items():
    print(f"Action ID: {action_id}")
    
    # Extract action annotations
    offence = action_data['Offence']
    contact = action_data['Contact']
    bodypart = action_data['Bodypart']
    upperbodypart = action_data['Upper body part']
    actionclass = action_data['Action class']
    severity = action_data['Severity']
    multiplefouls = action_data['Multiple fouls']
    trytoplay = action_data['Try to play']
    touchball = action_data['Touch ball']
    handball = action_data['Handball']
    handballoffence = action_data['Handball offence']
    
    # Extract action clips
    for clip in action_data['Clips']:
        url = clip['Url']
        cameratype = clip['Camera type']
        timestamp = clip['Timestamp']
        replayspeed = clip['Replay speed']

# Explore video data and prepare features
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path}")
    print(f"Frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

    # Example: Calculate frame differences as motion intensity
    frame_diffs = []
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        frame_diffs.append(np.sum(diff))
        prev_gray = gray

    cap.release()

    # Normalize differences
    frame_diffs = np.array(frame_diffs) / np.max(frame_diffs)

    plt.plot(frame_diffs)
    plt.title('Motion Intensity over Frames')
    plt.show()

    # Return statistics as potential features
    return {
        'mean_motion': np.mean(frame_diffs),
        'std_motion': np.std(frame_diffs),
        'max_motion': np.max(frame_diffs)
    }


# Iterate through clips and test feature extraction
features = []
for action_id, action_data in actions.items():
    for clip in action_data['Clips']:
        video_path = os.path.join('data/' + clip['Url'].lower() + '.mp4')
        if os.path.exists(video_path):
            features.append(extract_features(video_path))

# Example output of features
print(features[:5])

# Save features for model input
# np.save('video_features.npy', features)

