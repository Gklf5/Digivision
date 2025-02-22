import time
import cv2
import numpy as np
from face_recognizer import FaceHandler
from face_detection import FaceDetector
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Initialize components
face_detector = FaceDetector()
face_handler = FaceHandler()
test_images = []  # List of test face images (load from folder)
true_labels = []  # List of correct names (or "Unknown" for negative cases)

# Load test images (Modify this path)
TEST_FOLDER = "./test_faces/"  

for filename in os.listdir(TEST_FOLDER):
    img_path = os.path.join(TEST_FOLDER, filename)
    img = cv2.imread(img_path)
    
    if img is not None:
        test_images.append(img)
        true_labels.append(filename.split("_")[0])  # Extract name from filename format (e.g., "John_1.jpg")

# Accuracy Evaluation
def evaluate_accuracy():
    predicted_labels = []

    for img in test_images:
        name, _ = face_handler.is_known(img, threshold=0.6)
        predicted_labels.append(name if name else "Unknown")

    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# False Positive & False Negative Test
def test_false_recognition():
    false_positives = 0
    false_negatives = 0
    total_tests = len(test_images)

    for i, img in enumerate(test_images):
        name, _ = face_handler.is_known(img, threshold=0.6)
        if name and name != true_labels[i]:  # Incorrect match
            false_positives += 1
        elif not name and true_labels[i] != "Unknown":  # Missed match
            false_negatives += 1

    print(f"False Positives: {false_positives}/{total_tests} ({(false_positives/total_tests)*100:.2f}%)")
    print(f"False Negatives: {false_negatives}/{total_tests} ({(false_negatives/total_tests)*100:.2f}%)")

# Speed Performance Test
def test_speed():
    total_time = 0
    for img in test_images:
        start_time = time.time()
        face_handler.is_known(img, threshold=0.6)
        total_time += (time.time() - start_time)

    avg_time = total_time / len(test_images)
    print(f"Average Face Recognition Time: {avg_time:.3f} seconds per face")

# Run Tests
print("\nEvaluating System Performance...\n")
evaluate_accuracy()
test_false_recognition()
test_speed()



# test_faces/
# │── John_1.jpg    # Known face (John)
# │── John_2.jpg    # Same person, different angle
# │── Alice_1.jpg   # Known face (Alice)
# │── Bob_1.jpg     # Known face (Bob)
# │── Unknown_1.jpg # Unknown person (should NOT be recognized)
# │── Unknown_2.jpg # Another unknown face