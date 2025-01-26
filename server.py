import tkinter as tk
from PIL import Image, ImageTk
import cv2
from face_detection import FaceDetector
from face_recognizer import FaceHandler
from cam_handler import CamHandler

# Initialize components
camera_handler = CamHandler(cv2.VideoCapture(0))
face_detector = FaceDetector()
face_handler = FaceHandler()

root = tk.Tk()
root.title("Face Recognition System")

# GUI Widgets
fps_label = tk.Label(root, text="Set FPS:")
fps_label.pack()

fps_entry = tk.Entry(root)
fps_entry.pack()

def on_fps_change():
    try:
        fps = int(fps_entry.get())
        camera_handler.set_fps(fps)
    except ValueError:
        print("Invalid FPS value")
fps_entry.bind("<Return>", lambda event: on_fps_change())

camera_label = tk.Label(root)
camera_label.pack()

faces_frame = tk.Frame(root)
faces_frame.pack()

def show_detected_faces(faces):
    for widget in faces_frame.winfo_children():
        widget.destroy()

    for face in faces:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(face))
        label = tk.Label(faces_frame, image=img)
        label.imgtk = img
        label.pack(side="left")

def update_frame():
    frame = camera_handler.get_frame()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        camera_label.imgtk = img
        camera_label.configure(image=img)
    root.after(10, update_frame)

def handle_frame():
    frame = camera_handler.get_frame()
    faces = face_detector.get_face_images(frame)
    if faces:
        show_detected_faces(faces)
        face_detector.save_faces(frame, faces, "./detected_faces")
        for face in faces:
            if face_handler.is_known(face, get_threshold()):
                print("Known face detected!")
    root.after(1000, handle_frame)

threshold_label = tk.Label(root, text="Set Recognition Threshold:")
threshold_label.pack()

threshold_entry = tk.Entry(root)
threshold_entry.pack()
threshold_entry.insert(0, "0.6")  # Default value

def get_threshold():
    try:
        return float(threshold_entry.get())
    except ValueError:
        print("Invalid threshold value, using default 0.6")
        return 0.6

def on_close():
    camera_handler.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

update_frame()
handle_frame()

root.mainloop()


#
#
#
## Replace with your Supabase URL and API Key
# SUPABASE_URL = "https://zjecjfsufhuxygmzosuh.supabase.co"
# SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpqZWNqZnN1Zmh1eHlnbXpvc3VoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjkxNzQyODMsImV4cCI6MjA0NDc1MDI4M30.87rEdxoi6elx5qq6-W8wxrclotHpbjaYbX7Wtz4z3ZA"

# supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
#
#
#
#
#