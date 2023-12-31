import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2
from skimage import exposure
import os
from scipy.stats import skew, kurtosis

# Paths to your new image file, saved model, and scaler
new_image_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/normal/validation/"
new_image_name = "irdata_0001_0049.npy"
model_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/svm_classifier_Feature_1_Feature_2.joblib"
# scaler_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/fourier_feature_normalized_scaler_3.joblib"
excel_file = "trans_details.xlsx"

patch_size_rows = 3
patch_size_cols = 120


crop_starting_row = 110
crop_ending_row = 230
crop_starting_column = 200
crop_ending_column = 320

required_vali_images_number = 200


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

# Function to convert image to uint8
def convert_to_uint8(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    uint8_image = cv2.equalizeHist(normalized_image)

    return uint8_image.astype(np.uint8)  # Convert to uint8


def calculate_glcm_features_on_patch(patch):

    patch_uint8 = convert_to_uint8(patch)
    glcm = graycomatrix(patch_uint8, distances=[2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    energy = graycoprops(glcm, 'energy')
    return np.hstack([energy]).flatten()



def preprocess_image(image, patch_size_rows, patch_size_cols):

    image = exposure.equalize_hist(image)
    image = image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]

    # plt.imshow(image)
    # plt.show()

    features = []
    height, width = image.shape
    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            patch = image[y:y + patch_size_rows, x:x + patch_size_cols]
            patch_features = calculate_glcm_features_on_patch(patch)
            # Select only "Feature_1", "Feature_3", and "Feature_4"
            print("patch_features", patch_features)
            selected_features = np.array([patch_features[0], patch_features[1], patch_features[3]])
            # selected_features = np.array([patch_features[1], patch_features[3]])
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
    difference_due_zero_pixel_pos = 0

    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            label_image[y:y + patch_size_rows, x:x + patch_size_cols] = predictions[idx]
            idx += 1
    # np.save("000abc.npy", label_image)

    df_ground_truth = pd.read_excel(new_image_path + excel_file)

    if new_image_name in df_ground_truth['image name'].values:
        # Find the corresponding ground truth transitional line
        difference_due_zero_pixel_pos = df_ground_truth.loc[df_ground_truth['image name'] == new_image_name, 'difference'].values[0]


    # detection of transitional_line
    detected_transitional_line_image = np.zeros((height, width))
    for row in range(1, label_image.shape[0]):
        if not np.all(label_image[row] == label_image[row - 1]):
            print("transitional_line:", row+crop_starting_row-difference_due_zero_pixel_pos)
            transitional_line_detected = row
            detected_transitional_line_image[transitional_line_detected, ::20] = 50
            break


    # Check if the image name exists in the 'image name' column of ground truth data excel

    if new_image_name in df_ground_truth['image name'].values:
        # Find the corresponding ground truth transitional line
        transitional_line = df_ground_truth.loc[df_ground_truth['image name'] == new_image_name, 'transitional_line'].values[0]

        # marking the ground truth trans line on image used to show labels
        label_image[int(transitional_line) - crop_starting_row, ::20] = 2

        # marking the ground truth trans line on image used to show the detected trans line
        detected_transitional_line_image[int(transitional_line) - crop_starting_row, ::1] = 255

        # Find the index of the row with the specified image name to update the excel file with newly deteced transitional line
        row_index = df_ground_truth[df_ground_truth['image name'] == new_image_name].index
        # Update the value in the 'detected transitional line' column

        df_ground_truth.at[row_index[0], 'detected transitional line mod old_200_320_SVM'] = transitional_line_detected + crop_starting_row - difference_due_zero_pixel_pos
        # Save the updated DataFrame back to the Excel file

        df_ground_truth.to_excel(new_image_path + excel_file, index=False)

    # plt.imshow(label_image, cmap='jet')  # 'jet' colormap: red for turbulent (1), blue for laminar (0)
    # plt.show()
    #
    # plt.imshow(detected_transitional_line_image)
    # plt.show()

if __name__ == "__main__":
    main()
