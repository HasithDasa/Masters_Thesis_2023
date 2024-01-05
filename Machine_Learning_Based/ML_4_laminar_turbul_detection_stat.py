import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2
from skimage import exposure
from scipy.stats import skew, kurtosis

# Paths to your new image file, saved model, and scaler
new_image_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/normalized_stat/validation/"
# new_image_name = "irdata_0002_0201.npy"
model_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_18_stat.joblib"
excel_file = "trans_details.xlsx"

# scaler_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/fourier_feature_normalized_scaler_3.joblib"

patch_size_rows = 3
patch_size_cols = 150

# cropping details: as an example [75:180, 50:200], [crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]

crop_starting_row = 100
crop_ending_row = 175
crop_starting_column = 100
crop_ending_column = 250

required_vali_images_number = 400


def main():

    # List all files in the folder
    all_files = os.listdir(new_image_path)

    numpy_files = [file for file in all_files if file.endswith('.npy')]

    # Limit to the first 400 files
    numpy_files = numpy_files[:]

    for file_name in numpy_files:

        new_image_name = file_name

        print("new_image_name", new_image_name)

        # Load the image and preprocess it
        image = load_image(new_image_path + new_image_name)

        features = preprocess_image(image, patch_size_rows, patch_size_cols)

        # Load the trained model
        model = joblib.load(model_path)

        # Predict the labels
        predictions = model.predict(features)

        # Visualize the predictions
        visualize_regions(image, predictions, patch_size_rows, patch_size_cols, new_image_name)


def load_image(path):
    return np.load(path)


def calculate_statistical_moments(patch):
    mean = np.mean(patch)
    median = np.median(patch)
    # variance = np.var(patch)
    # skewness = skew(patch.flatten())
    # kurt = kurtosis(patch.flatten())

    return np.array([mean, median])


def preprocess_image(image, patch_size_rows, patch_size_cols):
    image = image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]

    # plt.imshow(image)
    # plt.show()

    features = []
    height, width = image.shape
    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            patch = image[y:y + patch_size_rows, x:x + patch_size_cols]
            patch_features = calculate_statistical_moments(patch)
            # Select only "Feature_1", "Feature_3", and "Feature_4"
            print("patch_features", patch_features)
            selected_features = np.array([patch_features[0], patch_features[1]])
            features.append(selected_features)
    return np.array(features)

# def normalize_features(features, scaler_path):
#     scaler = joblib.load(scaler_path)
#     return scaler.transform(features)

def visualize_regions(image, predictions, patch_size_rows, patch_size_cols, new_image_name):

    image = image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
    height, width = image.shape
    label_image = np.zeros((height, width))
    idx = 0
    transitional_line_detected = 0

    # label prediction using the model
    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            label_image[y:y + patch_size_rows, x:x + patch_size_cols] = predictions[idx]
            idx += 1


    # detection of transitional_line
    detected_transitional_line_image = np.zeros((height, width))
    for row in range(1, label_image.shape[0]):
        if not np.all(label_image[row] == label_image[row - 1]):
            print("transitional_line:", row+crop_starting_row)
            transitional_line_detected = row
            detected_transitional_line_image[transitional_line_detected, ::20] = 50


    # Check if the image name exists in the 'image name' column of ground truth data excel

    df_ground_truth = pd.read_excel(new_image_path+excel_file)
    if new_image_name in df_ground_truth['image name'].values:
        # Find the corresponding ground truth transitional line
        transitional_line = df_ground_truth.loc[df_ground_truth['image name'] == new_image_name, 'transitional_line'].values[0]

        # marking the ground truth trans line on image used to show labels
        label_image[int(transitional_line)-crop_starting_row, ::20] = 2

        # marking the ground truth trans line on image used to show the detected trans line
        detected_transitional_line_image[int(transitional_line) - crop_starting_row, ::1] = 255

        # Find the index of the row with the specified image name to update the excel file with newly deteced transitional line
        row_index = df_ground_truth[df_ground_truth['image name'] == new_image_name].index
        # Update the value in the 'detected transitional line' column

        df_ground_truth.at[row_index[0], 'detected transitional line new mod_100_250'] = transitional_line_detected+crop_starting_row
        # Save the updated DataFrame back to the Excel file

        df_ground_truth.to_excel(new_image_path+excel_file, index=False)


    # plt.imshow(label_image, cmap='jet')  # 'jet' colormap: red for turbulent (1), blue for laminar (0)
    # plt.show()
    #
    # plt.imshow(detected_transitional_line_image)
    # plt.show()



if __name__ == "__main__":
    main()