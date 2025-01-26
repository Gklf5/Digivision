import os
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
import pickle
from PIL import Image


class FaceHandler:
    def __init__(self, faces_folder='faces', embeddings_file='face_embeddings.pkl', device=None):
        """
        Initialize the FaceHandler with default paths and device.
        :param faces_folder: Path to the folder containing face images.
        :param embeddings_file: Path to save the face embeddings.
        :param device: Specify 'cpu' or 'cuda'. Defaults to GPU if available.
        """
        self.faces_folder = faces_folder
        self.embeddings_file = embeddings_file
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embeddings = self.load_embeddings()
        print(f"Initialized FaceHandler on {self.device}.")

    def __del__(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("GPU memory cleared.")

    def extract_face(self, image_path, target_size=(160, 160)):
        """
        Extract and resize a face from an image.
        :param image_path: Path to the image file.
        :param target_size: Size to resize the image to.
        :return: A NumPy array of the resized face or None if there's an error.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size)  # Resize image to 160x160
            return np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def preprocess_face(self, face_pixels):
        """Preprocess the cropped face for FaceNet."""
        required_size = (160, 160)  # Required size for FaceNet
        if face_pixels.shape[:2] != required_size:
            face_pixels = cv2.resize(face_pixels, required_size)
        face_pixels = face_pixels.astype('float32') / 255.0  # Normalize pixel values
        face_pixels = np.transpose(face_pixels, (2, 0, 1))  # Convert to (channels, height, width)
        face_tensor = torch.tensor(face_pixels).unsqueeze(0)  # Add batch dimension
        return face_tensor

    def get_embedding(self, face_pixels):
        """
        Generate an embedding for a single face.
        :param face_pixels: Raw face image as a NumPy array.
        :return: Face embedding as a NumPy array.
        """
        face_tensor = self.preprocess_face(face_pixels)
        with torch.no_grad():
            embedding = self.facenet_model(face_tensor).cpu().numpy().flatten()
        return embedding

    def process_faces(self):
        """
        Process all face images in the faces folder to generate embeddings.
        :return: A dictionary of filenames and their corresponding embeddings.
        """
        embeddings = {}
        for filename in os.listdir(self.faces_folder):
            if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(self.faces_folder, filename)
                print(f"Processing {filename}...")
                face = self.extract_face(image_path)
                if face is not None:
                    try:
                        embedding = self.get_embedding(face)
                        embeddings[filename] = embedding
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                else:
                    print(f"Face not detected or error in {filename}")
        return embeddings

    def save_embeddings(self, embeddings):
        """
        Save the embeddings to a file.
        :param embeddings: A dictionary of filenames and embeddings.
        """
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {self.embeddings_file}")

    def run(self):
        """
        Main function to process faces and save embeddings.
        """
        print("Starting face processing...")
        embeddings = self.process_faces()
        if embeddings:
            self.save_embeddings(embeddings)
        else:
            print("No embeddings to save.")


    def is_match(self, embedding1, embedding2, threshold=0.6):
        """
        Compare two face embeddings to determine if they match.
        :param embedding1: First face embedding as a NumPy array.
        :param embedding2: Second face embedding as a NumPy array.
        :param threshold: Distance threshold for considering a match.
        :return: True if the embeddings match, False otherwise.
        """
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance < threshold
    
    def load_embeddings(self):
        """
        Load face embeddings from a file.
        :return: A dictionary of filenames and their corresponding embeddings.
        """
        if not os.path.exists(self.embeddings_file):
            print(f"{self.embeddings_file} not found. Creating a new one.")
            return {}
        with open(self.embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} embeddings from {self.embeddings_file}")
        return embeddings
    
    def is_known(self, target_face, threshold=0.6):
        """
        Search all embeddings to see if any match the target frame.
        :param target_frame: The frame containing the face to match against.
        :param threshold: Distance threshold for considering a match.
        :return: True if any embedding matches the target embedding, False otherwise.
        """
        if target_face is None:
            print("No face detected in the target frame.")
            return False
        
        target_embedding = self.get_embedding(target_face)
        if target_embedding.shape[0] == 0:
            print("Empty embedding. Skipping...")
            return False
        for filename, embedding in self.embeddings.items():
            if self.is_match(target_embedding, embedding, threshold):
                print(f"Match found: {filename}")
                return True
        print("No match found.")
        return False


# # Example Usage
# if __name__ == "__main__":
#     face_handler = FaceHandler(faces_folder="faces", embeddings_file="face_embeddings.pkl")
#     face_handler.run()
