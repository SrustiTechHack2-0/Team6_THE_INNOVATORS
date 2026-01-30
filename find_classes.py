import pickle
import io

class ProxyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f"Loading class: {module}.{name}")
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return pid

import os

path = os.path.join(os.path.dirname(__file__), 'face_antispoofing_model', 'data.pkl')
with open(path, 'rb') as f:
    try:
        ProxyUnpickler(f).load()
        print("Successfully loaded pickle!")
    except Exception as e:
        print(f"Error: {e}")
