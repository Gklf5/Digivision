import requests
import base64

API_URL = "http://127.0.0.1:5000/add_face"
image_path = "faces/Adithya_5.jpg"

# Convert image to Base64
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

data = {
    "name": "abin",
    "image": base64_image
}

response = requests.post(API_URL, json=data)
print(response.json())
