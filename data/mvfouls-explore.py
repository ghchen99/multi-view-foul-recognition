import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load annotation file
with open('data/mvfouls/test/annotations.json', 'r') as f:
    annotations = json.load(f)

# Extract general dataset information
print("Dataset Set:", annotations['Set'])
print("Total Actions:", annotations['Number of actions'])

# Analyze action distribution
actions = annotations['Actions']
action_classes = defaultdict(int)
severities = []
clip_counts = []

for action_id, action_data in actions.items():
    action_classes[action_data['Action class']] += 1
    severity = action_data['Severity']
    if severity:  # Check if severity is not empty
        severities.append(float(severity))
    clip_counts.append(len(action_data['Clips']))

# Plot action class distribution
plt.figure(figsize=(12, 6))
plt.bar(action_classes.keys(), action_classes.values())
plt.title('Distribution of Action Classes')
plt.xlabel('Action Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot severity distribution
plt.figure(figsize=(8, 6))
plt.hist(severities, bins=10, alpha=0.7, color='blue')
plt.title('Severity Distribution')
plt.xlabel('Severity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot clip count distribution
plt.figure(figsize=(8, 6))
plt.hist(clip_counts, bins=10, alpha=0.7, color='green')
plt.title('Number of Clips per Action')
plt.xlabel('Clip Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

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

'''
# Iterate through clips and test feature extraction
features = []
for action_id, action_data in actions.items():
    for clip in action_data['Clips']:
        video_path = os.path.join(clip['Url'] + '.mp4')
        if os.path.exists(video_path):
            features.append(extract_features(video_path))

# Example output of features
print(features[:5])

# Save features for model input
# np.save('video_features.npy', features)

'''