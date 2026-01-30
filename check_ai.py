import requests
import base64
import os
import numpy as np
import cv2

# Create a blank image 
img = np.zeros((300, 300, 3), dtype=np.uint8)
_, buffer = cv2.imencode('.jpg', img)
encoded = base64.b64encode(buffer).decode('utf-8')
image_data = f'data:image/jpeg;base64,{encoded}'

try:
    print("Sending test request to persistent AI engine...")
    resp = requests.post('http://localhost:5000/api/predict', json={'image': image_data})
    print("Response Received:")
    print(resp.json())
except Exception as e:
    print(f"Error: {e}")
