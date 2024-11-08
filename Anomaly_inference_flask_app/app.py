from flask import Flask, request, jsonify
import base64
import json
import numpy as np
import cv2
import sys
import os
import time 

# Add inference module path
sys.path.append("./Inference")
from inference import Patchcore

app = Flask(__name__)

# Load configuration from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

# Extract configuration values
WEIGHTS_PATH = config["weights_path"]
SCORE_THRESHOLD = config["score_threshold"]
BOX_COLOR = tuple(config["bounding_box_color"])  # Convert color to tuple for cv2
BOX_THICKNESS = config["bounding_box_thickness"]

# Load the model once when the app starts
model = Patchcore()
model.load_model(WEIGHTS_PATH)

def decode_base64_image_to_np(base64_str):
    """Decode base64 string to OpenCV numpy array."""
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode("utf-8")

@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Step 1: Parse input JSON and decode image
        base64_image = request.json.get('image')
        if not base64_image:
            return jsonify({"error": "No image provided"}), 400

        input_image_np = decode_base64_image_to_np(base64_image)
        if input_image_np is None:
            return jsonify({"error": "Failed to decode base64 image"}), 400

        # Step 2: Perform inference
        start_time = time.time()  # Start time before inference
        score_map, masked_output, score = model.infer(input_image_np, SCORE_THRESHOLD)
        inference_time = time.time() - start_time  # End time after inference
        
        # Step 3: Post-processing (Draw bounding boxes)
        overlay_img = input_image_np.copy()
        score_img = cv2.resize(score_map, (input_image_np.shape[1], input_image_np.shape[0]))
        _, binary_image = cv2.threshold(score_img, SCORE_THRESHOLD, 255, cv2.THRESH_BINARY)
        binary_image = binary_image.astype(np.uint8)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay_img, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

        # Step 4: Encode the overlay image
        final_image_b64 = encode_image_to_base64(overlay_img)

        # Step 5: Determine result based on score threshold
        result = "good" if score < SCORE_THRESHOLD else "bad"
        print(result,score,inference_time)
        # Step 6: Return JSON response
        return jsonify({
            "final_output_image": final_image_b64,
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
