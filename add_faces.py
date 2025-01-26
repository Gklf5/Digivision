import cv2
import os
from face_detection import FaceDetector
from face_recognizer import FaceHandler
# Create the faces directory if it doesn't exist
face_detector = FaceDetector()
faces_dir = "./faces"
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Run for n times
n = 10
for _ in range(n):
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Save the captured frame to the faces directory
        faces = face_detector.detect_faces(frame)
        face_detector.save_faces(frame, faces, faces_dir)
# Release the webcam
cap.release()
cv2.destroyAllWindows()
face_handler = FaceHandler()
face_handler.run()