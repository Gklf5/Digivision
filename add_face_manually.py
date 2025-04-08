from dotenv import load_dotenv
import os
import requests
import cv2
from flask import request, jsonify
from face_detection import FaceDetector
from face_recognizer import FaceHandler
load_dotenv()
from supabase import create_client, Client
# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
# Initialize components
face_detector = FaceDetector()
face_handler = FaceHandler()
    
    

def add_face(id, name):
    """Download video from URL using ID"""
    try:
        video_url = supabase.table("criminal_db").select("video_url").eq("id", id).execute().data[0]["video_url"]
    except Exception as e:
        return f"Failed to fetch video URL: {str(e)}", 400
    print("Downloading video...")
    temp_folder = "./temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    try:
        response = requests.get(video_url)
        with open(f"{temp_folder}/video_{id}.mp4", "wb") as f:
            f.write(response.content)
    except Exception as e:
        return f"Failed to download video: {str(e)}", 400
    
    print("Reading video...")
    video = cv2.VideoCapture("temp/video.mp4")
    embeddings = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        faces, _ = face_detector.get_face_images(frame)
        print(f"Found {len(faces)} faces")

        if not faces:
            continue
        for face in faces:
            embedding = face_handler.get_embedding(face)
            if face_handler.is_similar(embedding):
                continue
            if embedding is not None:
                embeddings.append(embedding)
        print(f"Found {len(embeddings)} embeddings")
        if len(embeddings) > 20:
            break

    video.release()
    os.remove("temp/video.mp4")

    if not embeddings:
        print("No embeddings found")
        return "Failed to generate face embeddings", 400

    try:
        print("Storing embeddings in Supabase...")
        data = [{"id": id, "name": name, "embedding": emb.tolist()} for emb in embeddings]
        response = supabase.table("criminal_faces").insert(data).execute()
        print(f"response: {response}")
        return f"Added {len(embeddings)} faces successfully", 200
    except Exception as e:
        print(str(e))
        return f"Failed to store in database: {str(e)}", 500



id = input("ENTER id:")
name = input("Name:")
add_face(id, name)
