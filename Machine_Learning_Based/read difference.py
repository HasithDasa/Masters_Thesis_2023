import pandas as pd
import numpy as np
import os

# Replace 'your_excel_file.xlsx' with the path to your Excel file
excel_file_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/Validation old/trans_details.xlsx'
# Replace 'your_folder_path' with the path to the folder containing the .npy files
folder_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/Validation old/'

# Read the Excel file
df = pd.read_excel(excel_file_path)

# Assume the first column is named 'image name'
file_names = df['image name']
print(file_names)
# Initialize a list to store the difference values
differences = []

# Iterate over the file names
for file_name in file_names:
    # Construct the .npy file path
    file_path = folder_path+file_name
    print(file_path)

    # Check if the .npy file exists
    if os.path.exists(file_path):
        # Load the .npy file
        data = np.load(file_path)

        # Extract the first column
        first_column = data[:, 0]

        # Initialize the transition index to None
        transition_index = None
        # Look for the transition from zero to non-zero
        for i in range(1, len(first_column)):
            if first_column[i - 1] == 0 and first_column[i] != 0:
                transition_index = i
                break

        differences.append(transition_index)
    else:
        differences.append(None)  # Append None if file not found

# Add the differences as a new column to the DataFrame
df['difference'] = differences

# Save the updated DataFrame back to an Excel file
# Replace 'updated_excel_file.xlsx' with your desired output Excel file name

df.to_excel(excel_file_path, index=False)
