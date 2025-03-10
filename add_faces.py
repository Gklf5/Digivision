import cv2
import os
import time
from face_detection import FaceDetector
from face_recognizer import FaceHandler

face_detector = FaceDetector()
face_handler = FaceHandler()

def add_face_to_supabase(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not load image.")
        return

    faces = face_detector.detect_faces(image)
    if not faces:
        print("❌ No face detected.")
        return

    cropped_faces = face_detector.crop_faces(image, faces)
    if not cropped_faces:
        print("❌ Failed to crop faces.")
        return

    embedding = face_handler.get_embedding(cropped_faces[0])
    if embedding is None:
        print("❌ Failed to extract embedding.")
        return

    # Save to Supabase
    face_handler.save_embedding_to_supabase(name, embedding)


def download_video_from_supabase(video_url):
    try:
        # Get video data from Supabase storage URL
        response = face_handler.supabase.storage.from_('videos').get_public_url(video_url)
        
        # Save video locally
        filename = video_url.split('/')[-1]
        output_path = f'downloaded_videos/{filename}'
        os.makedirs('downloaded_videos', exist_ok=True)
        
        # Download from public URL
        import requests
        video_response = requests.get(response)
        
        with open(output_path, 'wb') as f:
            f.write(video_response.content)
            
        print(f"✅ Video downloaded successfully to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error downloading video: {str(e)}")
        return None


def process_video(video_url):
    video_path = download_video_from_supabase(video_url)
    if video_path:
        # Process the video here
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video.")
            return

        # Process frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every 30th frame to reduce computation
            if frame_count % 30 == 0:
                # Detect faces in frame
                faces = face_detector.detect_faces(frame)
                cropped_faces = face_detector.crop_faces(frame, faces)
                
                # Process each detected face
                for face in cropped_faces:
                    # Generate embedding
                    embedding = face_handler.get_embedding(face)
                    if embedding is not None:
                        # Generate unique name using timestamp
                        timestamp = int(time.time() * 1000)
                        name = f"person_{timestamp}"
                        
                        # Save to Supabase
                        face_handler.save_embedding_to_supabase(name, embedding)
                        print(f"✅ Face embedding saved from frame {frame_count}")

            frame_count += 1

        # Clean up
        cap.release()
        os.remove(video_path)  # Remove downloaded video after processing
        print(f"✅ Video processed successfully: {video_path}")
    else:
        print("❌ Failed to download video.")