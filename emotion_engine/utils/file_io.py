"""
File I/O Utility
"""
import json

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_txt(text, path):
    with open(path, 'w') as f:
        f.write(text)
