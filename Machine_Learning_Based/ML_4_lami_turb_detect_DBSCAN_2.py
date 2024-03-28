import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import exposure
import os
import re
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
from collections import Counter


# Paths to your new image file and scaler
new_image_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/Validation old/GLCM/"
joblib_directory = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 3/glcm/systematic_error/Unsupervised_DBSCAN/patch_10_200_290/selected"
excel_file = "trans_details DBSCAN Day3.xlsx"

df_trans_details = pd.read_excel(new_image_path + excel_file)

crop_starting_row = 25
crop_ending_row = 75

patch_size_rows = 3
patch_size_cols = 10

eps = 0.2
min_samples = 2

def extract_column_values(filename):
    """
    Extracts the crop starting and ending column values from the filename.
    """
    match = re.search(r"(\d+)_(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None, None

def load_image(path):
    return np.load(path)

def convert_to_uint8(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    uint8_image = cv2.equalizeHist(normalized_image)
    return uint8_image.astype(np.uint8)  # Convert to uint8

def calculate_glcm_features_on_patch(patch):
    patch_uint8 = convert_to_uint8(patch)
    glcm = graycomatrix(patch_uint8, distances=[0], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    energy = graycoprops(glcm, 'energy')
    return np.hstack([energy]).flatten()

def preprocess_image(image, patch_size_rows, patch_size_cols, crop_starting_column, crop_ending_column):
    image = exposure.equalize_hist(image)
    image = image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
    features = []
    height, width = image.shape
    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            patch = image[y:y + patch_size_rows, x:x + patch_size_cols]
            patch_features = calculate_glcm_features_on_patch(patch)
            # selected_features = np.array([patch_features[0], patch_features[1], patch_features[2], patch_features[3]])
            selected_features = np.array([patch_features[0], patch_features[1]])
            features.append(selected_features)
    return np.array(features)


def visualize_regions(image, labels, patch_size_rows, patch_size_cols, new_image_name, crop_starting_column,
                      crop_ending_column):
    """
    Visualizes the regions based on DBSCAN clustering results.

    Parameters:
    - image: The original grayscale image.
    - labels: Clustering labels for each patch, obtained from DBSCAN.
    - patch_size_rows, patch_size_cols: Dimensions of each patch.
    - new_image_name: The name of the processed image.
    - crop_starting_column, crop_ending_column: The cropping window for column.
    """
    image = image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
    height, width = image.shape
    label_image = np.zeros((height, width))
    idx = 0
    transitional_line_detected = 0
    difference_due_zero_pixel_pos = 0

    # Count the number of patches in each cluster, ignoring noise (-1).
    cluster_sizes = Counter(label for label in labels if label != -1)

    # Find the label of the largest cluster.
    if cluster_sizes:
        largest_cluster_label = cluster_sizes.most_common(1)[0][0]
    else:
        largest_cluster_label = None


    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            if labels[idx] == largest_cluster_label:  # Largest cluster
                label_image[y:y + patch_size_rows, x:x + patch_size_cols] = 0
            # elif labels[idx] != -1:  # Other clusters, not noise
            #     label_image[y:y + patch_size_rows, x:x + patch_size_cols] = 0
            else:  # Noise remains as background
                label_image[y:y + patch_size_rows, x:x + patch_size_cols] = 1
            idx += 1

    # Additional processing and transitional line detection logic remains unchanged

    df_ground_truth = pd.read_excel(new_image_path + excel_file)

    if new_image_name in df_ground_truth['image name'].values:
        # Find the corresponding ground truth transitional line
        difference_due_zero_pixel_pos = df_ground_truth.loc[df_ground_truth['image name'] == new_image_name, 'difference'].values[0]

    detected_transitional_line_image = np.zeros((height, width))
    for row in range(1, label_image.shape[0]):
        if not np.all(label_image[row] == label_image[row - 1]):
            print("transitional_line:", row + crop_starting_row - difference_due_zero_pixel_pos)
            transitional_line_detected = row
            detected_transitional_line_image[transitional_line_detected, ::20] = 50
            break

    # Check if the image name exists in the 'image name' column of ground truth data excel

    df_ground_truth = pd.read_excel(new_image_path + excel_file)
    if new_image_name in df_ground_truth['image name'].values:
        # Find the corresponding ground truth transitional line
        transitional_line = df_ground_truth.loc[df_ground_truth['image name'] == new_image_name, 'transitional_line'].values[0]
        print("actual trans", int(transitional_line) - (crop_starting_row - difference_due_zero_pixel_pos))

        # Find the index of the row with the specified image name to update the excel file with newly deteced transitional line
        row_index = df_ground_truth[df_ground_truth['image name'] == new_image_name].index
        # Update the value in the 'detected transitional line' column

        df_ground_truth.at[row_index[0], f'{crop_starting_column}_{crop_ending_column}'] = transitional_line_detected + crop_starting_row - difference_due_zero_pixel_pos

        # Save the updated DataFrame back to the Excel file

        df_ground_truth.to_excel(new_image_path + excel_file, index=False)


    # Visualization
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('on')

    # Clustered Image
    plt.subplot(1, 2, 2)
    plt.title("Clustered Regions")
    # Adjust the colormap and handling of noise (-1 labels)
    label_image[label_image == -1] = np.nan  # Set noise to NaN for visualization
    plt.imshow(label_image, cmap='jet', interpolation='nearest')  # Use 'jet' but with noise handled differently
    plt.colorbar(label='Cluster Label')  # Optional: Add a colorbar to denote cluster labels
    plt.axis('on')

    plt.tight_layout()
    # plt.show()

def main():
    joblib_files = [f for f in os.listdir(joblib_directory) if f.endswith('.joblib')]

    # Extract crop_starting_column and crop_ending_column from model_path
    for joblib_file in joblib_files:
        crop_starting_column, crop_ending_column = extract_column_values(joblib_file)
        all_files = os.listdir(new_image_path)
        numpy_files = [file for file in all_files if file.endswith('.npy')]

        # Limit to the first 400 files
        numpy_files = numpy_files[:]

        for file_name in numpy_files:
            new_image_name = file_name

            print("new_image_name", new_image_name)

            image = load_image(new_image_path + new_image_name)
            # features = preprocess_image(image, patch_size_rows, patch_size_cols, 0, image.shape[1])  # Example column range
            features = preprocess_image(image, patch_size_rows, patch_size_cols, crop_starting_column, crop_ending_column)

            # # Normalize features
            # scaler = StandardScaler()
            # features_normalized = scaler.fit_transform(features)

            # DBSCAN clustering
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
            labels = db.labels_

            # Visualize the clustering result
            visualize_regions(image, labels, patch_size_rows, patch_size_cols, new_image_name, crop_starting_column,crop_ending_column)

if __name__ == "__main__":
    main()
