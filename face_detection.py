from mtcnn import MTCNN
import cv2
import numpy as np
import os
import time

class FaceDetector:
    def __init__(self, min_confidence=0.90, resize_width=640):
        """ Initialize MTCNN face detector with confidence threshold and image resizing. """
        self.detector = MTCNN()
        self.min_confidence = min_confidence
        self.resize_width = resize_width

    def detect_faces(self, image):
        """ Detect faces in an image with confidence filtering. """
        if image is None or image.size == 0:
            print("❌ Invalid or empty image.")
            return []

        original_width = image.shape[1]
        scale = self.resize_width / original_width if original_width > self.resize_width else 1
        resized_image = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image

        faces = self.detector.detect_faces(resized_image)
        filtered_faces = [face for face in faces if face['confidence'] >= self.min_confidence]

        for face in filtered_faces:
            x, y, w, h = face['box']
            face['box'] = [int(x / scale), int(y / scale), int(w / scale), int(h / scale)]

        return filtered_faces

    def draw_faces(self, image, faces):
        """ Draw bounding boxes around detected faces. """
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image

    def crop_faces(self, image, faces, margin=20):
        cropped_faces = []
        for face in faces:
            x, y, w, h = face['box']
            x, y, w, h = max(0, x - margin), max(0, y - margin), w + 2 * margin, h + 2 * margin  # ✅ Add margin
            cropped_face = image[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)
        return cropped_faces


    def save_faces(self, image, faces, output_dir="./faces"):
        """ Save cropped faces with unique filenames. """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for face in faces:
            x, y, w, h = face['box']
            cropped_face = image[y:y+h, x:x+w]
            filename = os.path.join(output_dir, f"face_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(filename, cropped_face)
            print(f"✅ Face saved: {filename}")

    def process_image(self, image, output_dir="./faces"):
        """ Detect, draw, crop, and save faces from an image. """
        faces = self.detect_faces(image)
        image_with_boxes = self.draw_faces(image, faces)
        self.save_faces(image, faces, output_dir)
        return image_with_boxes, self.crop_faces(image, faces)

    def get_face_images(self, image):
        """ Detect and return cropped face images. """
        faces = self.detect_faces(image)
        return self.crop_faces(image, faces), faces
    

    
    
