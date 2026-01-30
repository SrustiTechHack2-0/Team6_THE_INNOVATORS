import torch
import os

model_path = r'd:\face_project\face_antispoofing_model'

try:
    print(f"Attempting to load as ScriptModule...")
    # ScriptModules can be loaded without the original python class
    model = torch.jit.load(model_path, map_location='cpu')
    print("Success! Loaded as ScriptModule.")
    print(model)
except Exception as e:
    print(f"Not a ScriptModule: {e}")
    try:
        print("Checking if it is a standard torch save...")
        # If it was saved with torch.save(model), it might work if the environment has the classes
        # But if it's a directory, it's often a saved model format
        model = torch.load(model_path, map_location='cpu')
        print(f"Loaded successfully as {type(model)}")
        print(model)
    except Exception as e2:
        print(f"Failed standard load: {e2}")
