import requests
import cv2
import base64
import json

# Flask API endpoint
URL = "http://127.0.0.1:5000/infer"  # Adjust if your server is running elsewhere

def encode_image_to_base64(image_path):
    # Read the image from file using OpenCV
    image = cv2.imread(image_path)
    # Encode image as PNG in memory
    _, buffer = cv2.imencode('.png', image)
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64

def send_inference_request(image_path):
    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)
    # Prepare JSON payload
    payload = {
        "image": image_base64
    }
    headers = {
        "Content-Type": "application/json"
    }
    # Send POST request to Flask API
    response = requests.post(URL, headers=headers, json=payload)
    # Check for successful response
    if response.status_code == 200:
        result = response.json()
        print("Inference Result:", result["result"])
        # print("Base64 Encoded Final Image:", result["final_output_image"])
    else:
        print("Error:", response.json())

# Specify the path to the image file you want to test with
image_path = "/home/dhvani-pranav/projects/pixIQ-dev/flaskapi/sampleimgs/000.png"  # Replace with the actual image path
send_inference_request(image_path)
