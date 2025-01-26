from mtcnn import MTCNN
import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image, resize_width=640):
        original_height, original_width = image.shape[:2]
        scale = resize_width / original_width if original_width > resize_width else 1
        resized_image = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image

        faces = self.detector.detect_faces(resized_image)

        for face in faces:
            if face['box']:
                x, y, w, h = face['box']
                face['box'] = [int(x / scale), int(y / scale), int(w / scale), int(h / scale)]
        return faces

    

    def draw_faces(self, image, faces):
        for face in faces:
            if 'box' in face and isinstance(face['box'], (list, tuple)):
                x, y, w, h = map(int, face['box'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                print(f"Invalid face box format: {face.get('box', 'No box found')}")
        return image

    def crop_faces(self, image, faces):
        cropped_faces = []
        for face in faces:
            if 'box' in face and isinstance(face['box'], (list, tuple)):
                x, y, w, h = map(int, face['box'])
                cropped_face = image[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                cropped_faces.append(cropped_face)
            else:
                print(f"Invalid face box format: {face.get('box', 'No box found')}")
        return cropped_faces

    def save_faces(self, image, faces, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        existing_files = len(os.listdir(output_dir))

        for i, face in enumerate(faces):
            try:
                if not isinstance(face, dict):
                    print(f"Skipping invalid face entry: {type(face)}")
                    continue

                if 'box' in face and isinstance(face['box'], (list, tuple)) and len(face['box']) == 4:
                    x, y, w, h = map(int, face['box'])
                    cropped_face = image[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                    cv2.imwrite(f'{output_dir}/face_{existing_files + i + 1}.jpg', cropped_face)
                else:
                    print(f"Skipping invalid face box: {face}")
            except Exception as e:
                print(f"Error processing face {i}: {e}")

    def process_image(self, image, output_dir):
        faces = self.detect_faces(image)
        image = self.draw_faces(image, faces)
        cropped_faces = self.crop_faces(image, faces)
        self.save_faces(image, faces, output_dir)
        return image, cropped_faces
    
    def get_face_images(self, image):
        faces = self.detect_faces(image)
        face_images = []
        for face in faces:
            if 'box' in face and isinstance(face['box'], (list, tuple)):
                x, y, w, h = map(int, face['box'])
                cropped_face = image[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                face_images.append(cropped_face)
            else:
                print(f"Invalid face box format: {face.get('box', 'No box found')}")
        return face_images
    

