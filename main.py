import numpy as np
import os
import shutil
from PIL import Image

def classify_image(image_path):
    # Read image using PIL
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)
    
    # Compute color channel means
    mean_r = np.mean(np_image[:, :, 0])
    mean_g = np.mean(np_image[:, :, 1])
    mean_b = np.mean(np_image[:, :, 2])
    
    # Compute brightness as mean of all channels
    brightness = np.mean(np_image)
    
    # Compute contrast using standard deviation
    contrast = np.std(np_image)
    
    # Compute LAB A-channel approximation using vectorized operations
    r, g, b = np_image[:, :, 0] / 255.0, np_image[:, :, 1] / 255.0, np_image[:, :, 2] / 255.0
    A = 500 * ((r - g) * 0.5)  # Approximate A-channel in vectorized form
    mean_a = np.mean(A)
    
    # Classification criteria based on refined analysis
    if mean_a > 2 and brightness < 200 and contrast > 25:
        return "bricky"
    else:
        return "grassy"

def classify_and_copy_images(dataset_path):
    new_dataset_path = "/Users/avilasha/Desktop/col780/"
    bricky_folder = os.path.join(new_dataset_path, "bricky")
    grassy_folder = os.path.join(new_dataset_path, "grassy")
    os.makedirs(bricky_folder, exist_ok=True)
    os.makedirs(grassy_folder, exist_ok=True)
    
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path):
            category = classify_image(file_path)
            destination_folder = bricky_folder if category == "bricky" else grassy_folder
            destination = os.path.join(destination_folder, filename)
            shutil.copy(file_path, destination)
            print(f"Copied {filename} -> {category}")

dataset_path = "/Users/avilasha/Desktop/col780/lane_dataset"
classify_and_copy_images(dataset_path)