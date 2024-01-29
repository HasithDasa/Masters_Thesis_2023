import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys

def view_and_copy_npy_files(source_folder, destination_folder):
    # List all .npy files in the source folder
    npy_files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]

    for filename in npy_files:
        # Close any existing figure
        plt.close('all')

        file_path = os.path.join(source_folder, filename)
        data = np.load(file_path)

        # Display the image
        plt.imshow(data)
        plt.title(filename)
        plt.show(block=False)

        action = input(f"Displaying {filename}. Type 'k' to copy, 'a' to skip, 'q' to quit: ").strip().lower()

        if action == 'k':
            shutil.copy(file_path, os.path.join(destination_folder, filename))
            print(f"Copied {filename}.")
        elif action == 'a':
            print("Skipping.")
        elif action == 'q':
            print("Quitting.")
            break
        else:
            print("Invalid input. Skipping.")

# Usage
source_folder = r"D:\Academic\MSc\Thesis\Project files\Project Complete\data\new data\npy\copied_images\New folder\231002_170018"
destination_folder =r"D:\Academic\MSc\Thesis\Project files\Project Complete\data\new data\npy\copied_images\New folder\231002_170018_selected"
view_and_copy_npy_files(source_folder, destination_folder)
