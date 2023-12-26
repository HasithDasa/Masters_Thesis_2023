import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2
from skimage import exposure
from scipy.stats import skew, kurtosis

# Paths to your new image file, saved model, and scaler
new_image_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/normalized/"
new_image_name = "irdata_0001_0005.npy"
model_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_14_stat.joblib"
# scaler_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/fourier_feature_normalized_scaler_3.joblib"


def load_image(path):
    return np.load(path)

# Function to convert image to uint8
# def convert_to_uint8(image):
#     normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     uint8_image = cv2.equalizeHist(normalized_image)
#
#     return uint8_image.astype(np.uint8)  # Convert to uint8


# def calculate_glcm_features_on_patch(patch):
#
#     patch_uint8 = convert_to_uint8(patch)
#     glcm = graycomatrix(patch_uint8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
#     energy = graycoprops(glcm, 'energy')
#     return np.hstack([energy]).flatten()

def calculate_statistical_moments(patch):
    mean = np.mean(patch)
    median = np.median(patch)
    # variance = np.var(patch)
    # skewness = skew(patch.flatten())
    # kurt = kurtosis(patch.flatten())

    return np.array([mean, median])


def preprocess_image(image, patch_size_rows, patch_size_cols):
    image = image[75:180, 50:200]

    plt.imshow(image)
    plt.show()

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

def visualize_regions(image, predictions, patch_size_rows, patch_size_cols):
    image = image[75:180, 50:200]
    height, width = image.shape
    label_image = np.zeros((height, width))
    idx = 0
    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            label_image[y:y + patch_size_rows, x:x + patch_size_cols] = predictions[idx]
            idx += 1
    # np.save("000abc.npy", label_image)

    # Check if the image name exists in the 'image name' column of ground truth data excel
    excel_file = "trans_details.xlsx"
    df_ground_truth = pd.read_excel(new_image_path+excel_file)
    if new_image_name in df_ground_truth['image name'].values:
        # Find the corresponding transitional line
        transitional_line = df_ground_truth.loc[df_ground_truth['image name'] == new_image_name, 'transitional_line'].values[0]
        label_image[int(transitional_line)-75, ::20] = 2

    plt.imshow(label_image, cmap='jet')  # 'jet' colormap: red for turbulent (1), blue for laminar (0)
    plt.show()



# Load the image and preprocess it
image = load_image(new_image_path+new_image_name)

# image = exposure.equalize_hist(image)

patch_size_rows = 5
patch_size_cols = 150

features = preprocess_image(image, patch_size_rows, patch_size_cols)

# # Normalize the features
# normalized_features = normalize_features(features, scaler_path)

normalized_features = features

# Load the trained model
model = joblib.load(model_path)

# Predict the labels
predictions = model.predict(normalized_features)

# Visualize the predictions
visualize_regions(image, predictions, patch_size_rows, patch_size_cols)
