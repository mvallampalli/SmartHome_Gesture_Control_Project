import cv2
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from handshape_feature_extractor import HandShapeFeatureExtractor

# video_path = "/Users/manasa/Documents/Visual Studio Codes/Python/SmartHomeGestureAppProjectPart2/traindata/LightOn_PRACTICE_1_Vallampalli.mp4"
training_folder = "traindata"
test_folder = "test"

gesture_labels = {
    "Num0": 0,
    "Num1": 1,
    "Num2": 2,
    "Num3": 3,
    "Num4": 4,
    "Num5": 5,
    "Num6": 6,
    "Num7": 7,
    "Num8": 8,
    "Num9": 9,
    "FanDown": 10,
    "FanOff": 11,
    "FanOn": 12,
    "FanUp": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16,
}


def get_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Couldn't open video: " + video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    success, frame = cap.read()
    cap.release()
    return frame


def normalize_name(name):
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def get_label_from_filename(filename):
    filename_without_extension = os.path.splitext(filename)[0]
    normalized_filename = normalize_name(filename_without_extension)
    for gesture_name, label in gesture_labels.items():
        normalized_gesture = normalize_name(gesture_name)
        expected_prefix = normalized_gesture + "practice"
        if normalized_filename.startswith(expected_prefix):
            return label
    return None


def natural_sort_key(filename):
    parts = re.split(r"(\d+)", filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


feature_extractor = HandShapeFeatureExtractor.get_instance()
print("Model loaded successfully")

training_features = []
training_labels_list = []

training_files = sorted(os.listdir(training_folder), key=natural_sort_key)

print("\nProcessing training videos...")

for filename in training_files:
    if not filename.lower().endswith(".mp4"):
        continue
    video_path = os.path.join(training_folder, filename)
    frame = get_middle_frame(video_path)
    feature_vector = feature_extractor.extract_feature(frame)
    feature_vector = np.asarray(feature_vector).reshape(1, -1)
    label = get_label_from_filename(filename)
    if label is None:
        raise RuntimeError("Couldn't determine label for training files: " + filename)
    training_features.append(feature_vector)
    training_labels_list.append(label)
    print(
        "Training:",
        filename,
        "-> Label:",
        label,
        "Feature shape:",
        feature_vector.shape,
    )


print("\n Total training videos processed:", len(training_features))

predictions = []

test_files = [
    filename
    for filename in os.listdir(test_folder)
    if filename.lower().endswith(".mp4")
]

test_files = sorted(test_files, key=natural_sort_key)
print("\nProcessing test videos...")

for filename in test_files:
    video_path = os.path.join(test_folder, filename)
    frame = get_middle_frame(video_path)
    test_feature = feature_extractor.extract_feature(frame)
    test_feature = np.asarray(test_feature).reshape(1, -1)
    best_similarity = -1.0
    predicted_label = None

    for i in range(len(training_features)):
        similarity = cosine_similarity(test_feature, training_features[i])[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            predicted_label = training_labels_list[i]

    predictions.append(int(predicted_label))
    print(
        "Test:",
        filename,
        "-> predicted Label:",
        predicted_label,
        "Similarity:",
        best_similarity,
    )

result = pd.DataFrame(predictions)
result.to_csv("Results.csv", index=False, header=False)
print("\nClassification complete.")
print("Total test videos processed:", len(predictions))
print("Results.csv generated successfully.")

if len(predictions) != 51:
    print("WARNING: Expected 51 predictions, but generated", len(predictions))
