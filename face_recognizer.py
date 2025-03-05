import os
import cv2
import numpy as np
import torch
import pickle
from facenet_pytorch import InceptionResnetV1
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
import time
import threading
from dotenv import load_dotenv

load_dotenv()

# Supabase Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class FaceHandler:
    def __init__(self, device=None, threshold=0.5):
        """ Initialize FaceNet model, load embeddings, and set recognition threshold. """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold  # Cosine similarity threshold for matching
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embeddings_file = "face_embeddings.pkl"
        threading.Thread(target=self.refresh_embeddings, args=(300,), daemon=True).start()
        
        # Try to load from Supabase, else fallback to local storage
        try:
            self.embeddings = self.load_embeddings_from_supabase()
        except:
            self.embeddings = self.load_embeddings_from_local()

        print(f"✅ FaceHandler initialized with {len(self.embeddings)} faces.")

    def preprocess_face(self, face_pixels):
        """ Preprocess face for FaceNet: Resize, normalize, and convert to tensor. """
        face_pixels = cv2.resize(face_pixels, (160, 160))
        face_pixels = (face_pixels.astype('float32') - 127.5) / 127.5  # Normalize
        face_tensor = torch.tensor(np.transpose(face_pixels, (2, 0, 1))).unsqueeze(0).to(self.device)
        return face_tensor

    def get_embedding(self, face_pixels):
        """ Generate an embedding for a face. """
        face_tensor = self.preprocess_face(face_pixels)
        with torch.no_grad():
            return self.facenet_model(face_tensor).cpu().numpy().flatten()

    def load_embeddings_from_local(self):
        """ Load face embeddings from local storage. """
        if not os.path.exists(self.embeddings_file):
            return {}
        with open(self.embeddings_file, 'rb') as f:
            return pickle.load(f)

    def load_embeddings_from_supabase(self):
        """Fetch embeddings from Supabase"""
        response = supabase.table("criminal_faces").select("*").execute()
        embeddings = {}
        if response.data:
            for record in response.data:
                name, embedding = record["name"], np.array(record["embedding"])
                if name in embeddings:
                    embeddings[name].append(embedding)
                else:
                    embeddings[name] = [embedding]
        print(f"✅ Loaded {len(embeddings)} faces from Supabase.")
        return embeddings

    def save_embedding_to_supabase(self, name, embedding):
        """Store a new face embedding in Supabase"""
        data = {"name": name, "embedding": embedding.tolist()}
        supabase.table("criminal_faces").insert(data).execute()
        print(f"✅ Face embedding for {name} saved to Supabase.")

    def save_embeddings(self, embeddings):
        """ Save embeddings locally. """
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)


    def is_known(self, target_face):
        target_embedding = self.get_embedding(target_face)
        if target_embedding is None:
            return None, None

        best_match, best_score = None, 0

        for name, stored_embeddings in self.embeddings.items():
            for stored_embedding in stored_embeddings:
                similarity = cosine_similarity([target_embedding], [stored_embedding])[0][0]
                print(f"🔍 Similarity with {name}: {similarity:.3f}")  # Debugging

                if similarity > self.threshold and similarity > best_score:
                    best_match, best_score = name, similarity

        return (best_match, best_score) if best_match else (None, None)
    
    def refresh_embeddings(self, interval=300):
        """Refresh embeddings every `interval` seconds"""
        while True:
            self.embeddings = self.load_embeddings_from_supabase()
            print("🔄 Face embeddings updated from Supabase.")
            time.sleep(interval)



