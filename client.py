import base64
import json
import requests
import matplotlib.pyplot as plt
import cv2
import numpy as np

def display_images(template_image, query_image, result_image):
    # Decode base64 encoded images
    template_image = cv2.imdecode(np.frombuffer(base64.b64decode(template_image), np.uint8), cv2.IMREAD_COLOR)
    query_image = cv2.imdecode(np.frombuffer(base64.b64decode(query_image), np.uint8), cv2.IMREAD_COLOR)
    result_image = cv2.imdecode(np.frombuffer(base64.b64decode(result_image), np.uint8), cv2.IMREAD_COLOR)

    # Display images using matplotlib
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Template Image')
    axs[0].axis('off')
    axs[1].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Query Image')
    axs[1].axis('off')
    axs[2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Result Image')
    axs[2].axis('off')
    plt.show()

def send_request(template_image_path, query_image_path):
    try:
        # Read template image and query image
        with open(template_image_path, 'rb') as f:
            template_image_data = f.read()
        with open(query_image_path, 'rb') as f:
            query_image_data = f.read()

        # Encode images as base64
        template_image_encoded = base64.b64encode(template_image_data).decode('utf-8')
        query_image_encoded = base64.b64encode(query_image_data).decode('utf-8')

        # Prepare data as JSON
        data = {
            'template_image': template_image_encoded,
            'query_image': query_image_encoded
        }

        # Send POST request to the server
        response = requests.post('http://localhost:5000/template-match', json=data)

        if response.status_code == 200:
            response_data = json.loads(response.text)
            result_image = response_data['result_image']

            # Display the images
            display_images(template_image_encoded, query_image_encoded, result_image)
        elif response.status_code == 400:
            error_message = json.loads(response.text)['error']
            print(f"Error: {error_message}")
        else:
            print("Template matching failed.")
    except Exception as e:
        print(f"Error during request: {str(e)}")

if __name__ == '__main__':
    template_image_path = 'ball.png' 
    query_image_path = 'soccer_practice.jpg' 

    send_request(template_image_path, query_image_path)
