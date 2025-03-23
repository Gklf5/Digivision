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
from skimage.metrics import structural_similarity as ssim
import asyncio
from flask_cors import CORS
load_dotenv()


SHOW_PREVIEW = False

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

rtsp_url_1 = "http://192.168.250.8:4747/video"
rtsp_url_2 = "http://172.16.12.143:4747/video"
rtsp_url_3 = "http://172.16.13.151:4747/video"
DETECTION_INTERVAL = 10  # Minimum time before re-notifying the same person
UNKNOWN_FACE_DIR = "./unknown_faces"
CAMERA_SOURCES = [rtsp_url_1]  # Add more camera indices or RTSP URLs
FPS = 10

LOCATION_NAME = {1:""}


# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)




# Initialize components
face_detector = FaceDetector()
face_handler = FaceHandler()
cameras = [CamHandler(cv2.VideoCapture(src)) for src in CAMERA_SOURCES]
last_notified = {}

app = Flask(__name__)
CORS(app)

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

def send_notification(name, location, confidence, photo_url):
    """ Sends a recognition alert via API """
    #get id from supabase table criminal_faces where name = name
    id = supabase.table("criminal_faces").select("id").eq("name", name).execute().data[0]["id"]
    # photo_url = supabase.table("criminal_db").select("photo_url").eq("id", id).execute().data[0]["photo_url"]
    if not photo_url:
        photo_url  = supabase.table("criminal_db").select("photo_url").eq("id", id).execute().data[0]["photo_url"]
    data = {
        "criminal_id": id,
        "location": location,
        "found_snap":photo_url
    }
    try:
        #post to supabase table notification_test
        supabase.table("criminal_notification").insert(data).execute()
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

def is_significant_change(frame1, frame2, threshold=0.80):
    """Check if two frames are significantly different using SSIM."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    similarity = ssim(gray1, gray2)
    return similarity < threshold  # Process only if similarity is below threshold

def process_camera(cam, cam_index):
    f_c = 0
    """Process camera feed for face recognition"""
    global last_notified
    last_frame = cam.get_frame()
    while True:
        # time.sleep(1/FPS)
        start_time = time.time()

        frame = cam.get_frame()
        f_c = f_c + 1
        if f_c != 25:
            continue
        f_c = 0
        if not is_significant_change(frame, last_frame):
            continue
        last_frame = frame
        faces, face_coords = face_detector.get_face_images(frame)
        frame_with_face = face_detector.draw_faces(frame, face_coords)
       
        print(f"📸 Detected {len(faces)} faces in Camera {cam_index}")



        for face in faces:
            recog_start = time.time()
            
            name, confidence = face_handler.is_known(face)
            recog_time = time.time() - recog_start

            timestamp = time.time()
            location = f"Camera {cam_index}"

            if name:
                if name not in last_notified or (timestamp - last_notified[name] > DETECTION_INTERVAL):
                    if SHOW_PREVIEW:
                        cv2.putText(frame_with_face, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow(f"Camera {cam_index}", frame_with_face)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        
                    photo_url = upload_photo(face)
                    send_notification(name, location, confidence, photo_url)
                    # print(f"✅ Notification sent for {name} at {location}")
                    last_notified[name] = timestamp
                    log_event(f"✅ Recognition Time: {recog_time:.3f}s | {name} detected, confidence: {confidence:.3f}")
            else:
                # save_unknown_face(face)
                log_event(f"❌ Unknown face detected at {location}")

        total_time = time.time() - start_time
        log_event(f"⏳ Frame Processing Time: {total_time:.3f}s")

def process_new_face(image_data, name):
    """Convert image to embedding & store in Supabase"""
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Get face embedding
    faces, _ = face_detector.get_face_images(image)
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
    print("Adding face...Got data")
    data = request.get_json()
    if not data or "video_url" not in data or "name" not in data:
        return jsonify({"error": "Missing video_url or name"}), 400

    video_url = data["video_url"]
    name = data["name"]
    print("Downloading video...")
    #download from supabase storage save in temp folder
    temp_folder = "./temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    try:
        response = requests.get(video_url)
        with open("temp/video.mp4", "wb") as f:
            f.write(response.content)
        #delete video temp/video.mp4

    except Exception as e:
        return jsonify({"error": f"Failed to download video: {str(e)}"}), 400   
    
    #read video
    print("Reading video...")
    video = cv2.VideoCapture("temp/video.mp4")
    #get frame from video
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
            #if embeding is similar to any of the embeddings in the database, skip
            if face_handler.is_similar(embedding):
                continue
            if embedding is not None:
                embeddings.append(embedding)
        print(f"Found {len(embeddings)} embeddings")
        if len(embeddings) > 20:
            break

    if not embeddings:
        print("No embeddings found")
        return jsonify({"error": "Failed to generate face embeddings"}), 400

    # Store all embeddings in Supabase
    try:
        print("Storing embeddings in Supabase...")
        #fetch id from table criminal_db where name = name
        id = supabase.table("criminal_db").select("id").eq("name", name).execute().data[0]["id"]
        data = [{"id": id, "name": name, "embedding": emb.tolist()} for emb in embeddings]
        response = supabase.table("criminal_faces").insert(data).execute()
        os.remove("temp/video.mp4")
        face_handler.refresh_embeddings()
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

def upload_photo(image):
    """
    Upload an image (numpy array) to Supabase Storage.
    
    :param image: OpenCV image (numpy array)
    :return: Public URL of the uploaded image or None if failed.
    """

    if image is None:
        print("Error: No image provided.")
        return None

    # Encode image as JPEG
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        print("Error: Could not encode image.")
        return None

    # Generate unique filename
    timestamp = int(time.time())
    file_name = f"{timestamp}_photo.jpg"

    try:
        # Upload image to Supabase Storage
        response = supabase.storage.from_('criminal_photos').upload(
            file_name,
            buffer.tobytes(),  # Convert to bytes for upload
            file_options={"content-type": "image/jpeg"}  # Set MIME type
        )

        # Check if response has an "error" attribute
        if hasattr(response, "error") and response.error:
            print("Upload failed:", response.error)
            return None

        # Get the public URL of the uploaded file
        public_url = supabase.storage.from_('criminal_photos').get_public_url(file_name)
        print("Uploaded successfully:", public_url)

        return public_url

    except Exception as e:
        print("Exception occurred while uploading:", str(e))
        return None

if __name__ == "__main__":
    threading.Thread(target=main, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
