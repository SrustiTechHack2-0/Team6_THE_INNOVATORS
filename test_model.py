import torch
import os

model_path = r'd:\face_project\face_antispoofing_model'

try:
    # Typical way to load a saved model directory or file
    # If it's a directory, torch.load might work if it's the zip format
    # But usually torch.load(file_path)
    
    # Let's try loading it directly
    print(f"Loading model from {model_path}...")
    
    # Check if there's a pkl or pth file
    # Actually, the directory itself is often what's passed if it was a saved model
    
    # Trying torch.hub or similar is also a possibility, but let's try torch.load
    # Sometimes models are saved as .pth but the folder is just a container
    
    # Wait, I saw data.pkl in the folder.
    data_pkl = os.path.join(model_path, 'data.pkl')
    
    # In newer torch versions, torch.load(folder) works if it's a package
    # Let's try both
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print("Model loaded successfully from folder!")
    except:
        try:
            model = torch.load(data_pkl, map_location=torch.device('cpu'))
            print("Model loaded successfully from data.pkl!")
        except Exception as e:
            print(f"Failed to load: {e}")

    if 'model' in locals():
        print(f"Type of loaded object: {type(model)}")
        if hasattr(model, 'eval'):
            print("Model has eval() method.")
        else:
            print("Model does not seem to be a torch module (maybe a state dict or just data).")
            if isinstance(model, dict):
                print(f"Keys in dict: {model.keys()}")

except Exception as e:
    print(f"Error: {e}")
