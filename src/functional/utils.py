import os 
from PIL import Image
import numpy as np


def gather_file_paths(dir_path):
        file_paths = []
        for root, dirs, files in os.walk(dir_path):
            wav_files = [os.path.join(root, f) for f in files if f.endswith('.png')]
            file_paths.extend(wav_files)
        return file_paths

def extract_label(file_path):
        # Extract the file name from the full path
        file_name = os.path.basename(file_path)
        # Split the filename at the first space and take the first part (before the space)
        label = file_name.split("_")[0]
        return label

def load_chroma_images(image_paths):
    chroma_images = []
    for image_path in image_paths:
        # Load image, convert to grayscale (if needed), and resize to a fixed size if required
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = np.array(img).astype("float32") / 255.0  # Normalize to [0, 1]
        # Add a channel dimension to match CNN input requirements
        img = np.expand_dims(img, axis=-1)
        chroma_images.append(img)
    return np.array(chroma_images)