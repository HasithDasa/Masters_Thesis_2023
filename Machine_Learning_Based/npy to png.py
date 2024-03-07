import numpy as np
import imageio
import os


def npy_to_png(npy_file_path, output_folder):
    # Load the numpy array from the .npy file
    array = np.load(npy_file_path)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate output file path
    base_name = os.path.basename(npy_file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(output_folder, f"{file_name_without_ext}.png")

    # Normalize the array to be in the range of 0 to 255
    array = (array - array.min()) / (array.max() - array.min()) * 255.0
    array = array.astype(np.uint8)

    # Save the array as a PNG image
    imageio.imwrite(output_file_path, array)
    print(f"Saved PNG image to '{output_file_path}'")


def process_folder(input_folder, output_folder):
    # List all files in the input folder
    for file_name in os.listdir(input_folder):
        # Check if the file is an .npy file
        if file_name.endswith('.npy'):
            npy_file_path = os.path.join(input_folder, file_name)
            npy_to_png(npy_file_path, output_folder)

    print(f"Processed all .npy files in '{input_folder}' and saved them to '{output_folder}'")


# Example usage
input_folder = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/masks"
output_folder = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/masks/New folder"
process_folder(input_folder, output_folder)
