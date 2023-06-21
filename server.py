import base64
import json
import cv2
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def template_match(template_data, query_data):
    try:
        # Decode base64 encoded images
        template_image = cv2.imdecode(np.frombuffer(base64.b64decode(template_data), np.uint8), cv2.IMREAD_COLOR)
        query_image = cv2.imdecode(np.frombuffer(base64.b64decode(query_data), np.uint8), cv2.IMREAD_COLOR)

        # Perform template matching
        result = cv2.matchTemplate(query_image, template_image, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  
        loc = np.where(result >= threshold)

        # Draw bounding boxes around the matches
        for x,y in zip(*loc[::-1]):
            cv2.rectangle(query_image, (x,y), (x + template_image.shape[1], y + template_image.shape[0]), (0, 255, 0), 2)

        # Encode the image as base64
        _, encoded_image = cv2.imencode('.png', query_image)
        encoded_image_data = base64.b64encode(encoded_image).decode('utf-8')

        return encoded_image_data
    except Exception as e:
        print(f"Error during template matching: {str(e)}")
        return None

@app.route('/template-match', methods=['POST'])
def handle_template_match():
    try:
        data = json.loads(request.data)
        template_data = data['template_image']
        query_data = data['query_image']

        result = template_match(template_data, query_data)

        if result is not None:
            # Return the result as base64 encoded JSON
            response_data = {
                'result_image': result
            }
            return json.dumps(response_data)
        else:
            return jsonify({'error': 'Template matching failed.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
