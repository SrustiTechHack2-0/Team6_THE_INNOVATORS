import pickle
import os

path = r'd:\face_project\face_antispoofing_model\data.pkl'
with open(path, 'rb') as f:
    try:
        data = pickle.load(f)
        print("Pickle loaded!")
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {data.keys()}")
        else:
            print(f"Data: {data}")
    except Exception as e:
        print(f"Error: {e}")
