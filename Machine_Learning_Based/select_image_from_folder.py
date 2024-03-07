import pandas as pd
import shutil
import os

# Adjust these paths according to your directory structure
excel_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/trans_details.xlsx"
source_folder = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/masks"
destination_folder = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/masks/New folder"

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read the first column of the Excel file
df = pd.read_excel(excel_path, usecols=[0])

# Loop through the names in the first column
for name in df.iloc[:, 0]:
    name = name.replace(".npy", "")
    name = name + "_lami.npy"
    # Construct the file path
    file_path = os.path.join(source_folder, name)
    # Check if the file exists
    if os.path.exists(file_path):
        # Copy the file to the new folder
        shutil.copy(file_path, destination_folder)
        print(f'Copied {name} to {destination_folder}')
    else:
        print(f'File {name} not found in {source_folder}')
