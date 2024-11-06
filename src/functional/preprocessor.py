import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def save_chroma_as_image(file_path, output_dir, sample_rate=22050):
    """
    Extracts chroma features from a .wav file and saves them as an image without axes or labels.

    Parameters:
    - file_path (str): Path to the .wav file.
    - output_dir (str): Directory where the image will be saved.
    - sample_rate (int): The sample rate to load the audio file (default: 22050).
    
    Returns:
    - None (saves an image file in the specified output directory).
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sample_rate)
    
    # Compute the chroma feature matrix
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output image file path
    file_name = os.path.splitext(os.path.basename(file_path))[0] + '_chroma.png'
    output_path = os.path.join(output_dir, file_name)
    
    # Plot the chroma feature matrix without any axes, labels, or colorbars
    plt.figure(figsize=(4, 4))  # Adjust figure size as needed
    plt.axis('off')  # Turn off axes
    librosa.display.specshow(chroma, cmap='gray_r')  # Optional: change cmap if needed
    
    # Save the plot as an image
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory

    print(f"Chroma image saved to: {output_path}")

from PIL import Image
import os

def resize_image(file_path, output_dir, size=(40, 40)):
    """
    Resizes an image to the specified size and saves it in the output directory.

    Parameters:
    - file_path (str): Path to the original image file.
    - output_dir (str): Directory where the resized image will be saved.
    - size (tuple): Target size for resizing (width, height), default is (40, 40).
    
    Returns:
    - None (saves the resized image in the specified output directory).
    """
    # Open the image
    img = Image.open(file_path)
    
    # Resize the image
    img_resized = img.resize(size, Image.LANCZOS)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output image file path
    file_name = os.path.splitext(os.path.basename(file_path))[0] + '_resized.png'
    output_path = os.path.join(output_dir, file_name)
    
    # Save the resized image
    img_resized.save(output_path, format='PNG')
    
    print(f"Resized image saved to: {output_path}")

# Example usage
# resize_image('path_to_image.png', 'output_directory')

def change_underscore(file_path, output_dir):
    """
    Replaces each '-' with '_' in the file name and saves the renamed file in the output directory.

    Parameters:
    - file_path (str): Path to the original file.
    - output_dir (str): Directory where the renamed file will be saved.
    
    Returns:
    - None (saves the renamed file in the specified output directory).
    """
    # Extract the directory, base name, and extension of the file
    dir_name, file_name = os.path.split(file_path)
    new_file_name = file_name.replace("-", "_")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the new file path in the output directory
    output_path = os.path.join(output_dir, new_file_name)
    
    # Copy the file to the new path
    os.rename(file_path, output_path)
    
    print(f"Renamed file saved to: {output_path}")

directory = r'C:\Users\rapha\repositories\guitar_vision\data\raw\other'
file_paths = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True) #change between .wav and .png

output_dir = r'chromas\other'


for file_path in file_paths:

    save_chroma_as_image(file_path, output_dir)
