import torch
import os

model_path = r'd:\face_project\face_antispoofing_model\data.pkl'

try:
    print("Loading with weights_only=False...")
    # This is often needed for files that contain more than just tensors
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"Success! Type: {type(model)}")
    if isinstance(model, dict):
        print(f"Keys: {model.keys()}")
except Exception as e:
    print(f"Error: {e}")
