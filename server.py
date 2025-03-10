from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import base64
import requests
import time
import threading
from supabase import create_client, Client
from face_detection import FaceDetector
from face_recognizer import FaceHandler
from cam_handler import CamHandler
from dotenv import load_dotenv
import asyncio
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")



DETECTION_INTERVAL = 10  # Minimum time before re-notifying the same person
UNKNOWN_FACE_DIR = "./unknown_faces"
CAMERA_SOURCES = [0]  # Add more camera indices or RTSP URLs

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)




# Initialize components
face_detector = FaceDetector()
face_handler = FaceHandler()
cameras = [CamHandler(cv2.VideoCapture(src)) for src in CAMERA_SOURCES]
last_notified = {}

app = Flask(__name__)

# Ensure unknown faces directory exists
if not os.path.exists(UNKNOWN_FACE_DIR):
    os.makedirs(UNKNOWN_FACE_DIR)

def log_event(event):
    """Logs recognition events to Supabase"""
    data = {"log_message": event, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        supabase.table("system_logs").insert(data).execute()
    except:
        print("⚠️ Log not sent to Supabase!")

    print(event)  # Debugging

def send_notification(name, location, confidence, timestamp):
    """ Sends a recognition alert via API """
    data = {
        "name": name,
        "location": location,
        "confidence": float(confidence),
        "timestamp": float(timestamp)
    }
    try:
        #post to supabase table notification_test
        supabase.table("notification_test").insert(data).execute()
        print(f"✅ Notification sent for {name} at {location}")
        log_event(f"Notification sent for {name} at {location} with confidence {confidence}")
    except Exception as e:
        print(f"❌ Notification failed for {name} at {location}: {e}")
        log_event(f"Notification failed for {name} at {location}: {e}")

def save_unknown_face(face_image):
    """Saves unrecognized faces for later review"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(UNKNOWN_FACE_DIR, f"unknown_{timestamp}.jpg")
    # cv2.imwrite(filename, face_image)
    print(f"❌ Unknown face saved: {filename}")

def process_camera(cam, cam_index):
    """Process camera feed for face recognition"""
    global last_notified
    while True:
        time.sleep(60)
        start_time = time.time()

        frame = cam.get_frame()
        faces = face_detector.get_face_images(frame)
        print(f"📸 Detected {len(faces)} faces in Camera {cam_index}")

        for face in faces:
            recog_start = time.time()
            name, confidence = face_handler.is_known(face)
            recog_time = time.time() - recog_start

            timestamp = time.time()
            location = f"Camera {cam_index}"

            if name:
                if name not in last_notified or (timestamp - last_notified[name] > DETECTION_INTERVAL):
                    send_notification(name, location, confidence, timestamp)
                    last_notified[name] = timestamp
                    log_event(f"✅ Recognition Time: {recog_time:.3f}s | {name} detected")
            else:
                save_unknown_face(face)
                log_event(f"❌ Unknown face detected at {location}")

        total_time = time.time() - start_time
        log_event(f"⏳ Frame Processing Time: {total_time:.3f}s")

def process_new_face(image_data, name):
    """Convert image to embedding & store in Supabase"""
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Get face embedding
    faces = face_detector.get_face_images(image)
    if not faces:
        return False

    embedding = face_handler.get_embedding(faces[0])
    if embedding is None:
        return False

    # Store in Supabase
    data = {"name": name, "embedding": embedding.tolist()}
    response = supabase.table("criminal_faces").insert(data).execute()

    return response

@app.route("/add_face", methods=["POST"])
def add_face():
    """API to add new face to criminal database from web app"""
    data = request.get_json()
    if not data or "video_url" not in data or "name" not in data:
        return jsonify({"error": "Missing video_url or name"}), 400

    video_url = data["video_url"]
    name = data["name"]

    # Download video from URL
    try:
        response = requests.get(video_url)
        video_bytes = response.content
        video_array = np.frombuffer(video_bytes, np.uint8)
        video = cv2.imdecode(video_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Failed to download video: {str(e)}"}), 400

    # Get face embeddings
    faces = face_detector.get_face_images(video)
    if not faces:
        return jsonify({"error": "No faces detected in video"}), 400

    # Process all detected faces
    embeddings = []
    for face in faces:
        embedding = face_handler.get_embedding(face)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return jsonify({"error": "Failed to generate face embeddings"}), 400

    # Store all embeddings in Supabase
    try:
        data = [{"name": name, "embedding": emb.tolist()} for emb in embeddings]
        response = supabase.table("criminal_faces").insert(data).execute()
        return jsonify({
            "success": True, 
            "message": f"Added {len(embeddings)} faces successfully"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to store in database: {str(e)}"}), 500

def main():
    """ Starts multiple threads to process camera feeds in parallel """
    threads = []
    for i, cam in enumerate(cameras):
        t = threading.Thread(target=process_camera, args=(cam, i))
        t.daemon = True
        threads.append(t)
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for cam in cameras:
            cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=main, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
