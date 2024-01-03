import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


def process_and_save_lbp(file_path, output_dir):
    # Load the thermal data from the given path
    thermal_data = np.load(file_path)

    # Apply LBP
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(thermal_data, n_points, radius, method='uniform')

    # Save the LBP image with consistent naming
    base_name = os.path.basename(file_path)
    lbp_file_name = f"lbp_{base_name.replace('.npy', '.png')}"
    lbp_file_path = os.path.join(output_dir, lbp_file_name)
    plt.imsave(lbp_file_path, lbp_image, cmap='gray')

    return lbp_file_path


# Directory where the numpy images are stored
input_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/"
output_dir = os.path.join(input_dir, "lbp_images_r1_p8")  # Creating a sub-directory for LBP images

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process all the .npy files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".npy"):
        file_path = os.path.join(input_dir, file_name)
        lbp_path = process_and_save_lbp(file_path, output_dir)
        print(f"Processed and saved LBP image: {lbp_path}")

print("Processing completed!")
