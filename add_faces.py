import cv2
import os
from face_detection import FaceDetector
from face_recognizer import FaceHandler

face_detector = FaceDetector()
face_handler = FaceHandler()

def add_face_to_supabase(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not load image.")
        return

    faces = face_detector.get_face_images(image)
    if not faces:
        print("❌ No face detected.")
        return

    embedding = face_handler.get_embedding(faces[0])
    if embedding is None:
        print("❌ Failed to extract embedding.")
        return

    # Save to Supabase
    face_handler.save_embedding_to_supabase(name, embedding)

# Example usage
image_path = "unknown_faces/Adithya_3.jpg"
person_name = "adithya"
add_face_to_supabase(image_path, person_name)
