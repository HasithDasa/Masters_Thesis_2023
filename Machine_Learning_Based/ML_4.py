import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
import os
import pandas as pd


def load_image(path):
    return np.load(path)

# Load and binarize masks
def load_and_binarize_mask(path):
    print("path", path)
    mask = np.load(path)
    mask[mask == 1] = 0
    mask[mask == 10] = 1
    return mask

# Function to convert image to uint8
def convert_to_uint8(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    uint8_image = cv2.equalizeHist(normalized_image)

    return uint8_image.astype(np.uint8)  # Convert to uint8

def calculate_lbp(image, lbp_radius=1, lbp_n_points=8):
    """
    Calculate the Local Binary Pattern (LBP) of an image.
    """
    lbp_image = local_binary_pattern(image, lbp_n_points, lbp_radius, method='uniform')
    return lbp_image


def calculate_glcm_features_on_patch(patch):

    lbp_patch = calculate_lbp(patch)
    lbp_patch = lbp_patch.astype(np.uint8) # convert to uint8 without changing anyvalue
    # print(np.unique(lbp_patch))
    patch_uint8 = convert_to_uint8(patch) # keep this for general glcm calculations
    glcm = graycomatrix(lbp_patch, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation]).flatten()

def process_image_for_masked_regions(image, mask, label, patch_size):
    height, width = image.shape
    features = []
    labels = []

    for y in range(0, height - patch_size + 1):
        for x in range(0, width - patch_size + 1):
            if mask[y, x] == 1:  # Check if the current pixel in the mask is white
                patch = image[y:y + patch_size, x:x + patch_size]
                patch_features = calculate_glcm_features_on_patch(patch)
                features.append(patch_features)
                labels.append(label)

    return features, labels

# def draw_rectangle_on_patch(image, color=(0, 255, 0), thickness=2):
#     # Find non-zero coordinates in the image
#     coords = np.column_stack(np.where(image > 0))
#
#     # Calculate bounding box of the non-zero regions
#     y_min, x_min = coords.min(axis=0)
#     y_max, x_max = coords.max(axis=0)
#
#     # Convert image to uint8 if it's not already
#     if image.dtype != np.uint8:
#         image = convert_to_uint8(image)
#
#     # Create a copy of the original image to draw the rectangle
#     image_with_rectangle = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
#     # Draw the rectangle
#     cv2.rectangle(image_with_rectangle, (x_min, y_min), (x_max, y_max), color, thickness)
#
#     return image_with_rectangle, (x_min, y_min, x_max, y_max)


def get_matching_mask_path(image_path, mask_dir, mask_name_end):
    base_name = os.path.basename(image_path)
    mask_name = base_name.replace('.npy', mask_name_end)  # Replace .npy with ._turb.npy or _lamina.npy
    return os.path.join(mask_dir, mask_name).replace('\\', '/')

# Directories containing the images and masks
image_dir = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/'
mask_dir = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/masks'
mask_name_end_turb = '_turbul.npy'
mask_name_end_lami = '_lami.npy'

patch_size = 5  # Define the patch size

all_features = []
all_labels = []

for image_name in os.listdir(image_dir):
    if image_name.endswith('.npy'):  # Ensure processing .npy files
        img_path = os.path.join(image_dir, image_name)
        print("img_path", img_path)
        image = load_image(img_path)

        turb_mask_path = get_matching_mask_path(img_path, mask_dir, mask_name_end_turb)
        lamina_mask_path = get_matching_mask_path(img_path, mask_dir, mask_name_end_lami)
        turb_mask = load_and_binarize_mask(turb_mask_path)
        lamina_mask = load_and_binarize_mask(lamina_mask_path)

        turb_features, turb_labels = process_image_for_masked_regions(image, turb_mask, 1, patch_size)
        lami_features, lami_labels = process_image_for_masked_regions(image, lamina_mask, 0, patch_size)

        all_features.extend(turb_features)
        all_features.extend(lami_features)
        all_labels.extend(turb_labels)
        all_labels.extend(lami_labels)

# Convert to numpy arrays and create a DataFrame
all_features = np.array(all_features)
all_labels = np.array(all_labels)
feature_columns = [f'Feature_{i+1}' for i in range(all_features.shape[1])]
df = pd.DataFrame(all_features, columns=feature_columns)
df['Label'] = all_labels

# Assuming 'df' is your DataFram
save_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_3.csv'

# Save the DataFrame as a CSV file
df.to_csv(save_path, index=False)


















# # Statistical comparison
# transitional_mean = np.mean(haralick_features_transitional, axis=0)
# other_mean = np.mean(haralick_features_other, axis=0)
# transitional_std = np.std(haralick_features_transitional, axis=0)
# other_std = np.std(haralick_features_other, axis=0)

# # Print the mean and standard deviation for comparison
# print("Transitional Region Mean:", transitional_mean)
# print("Transitional Region Standard Deviation:", transitional_std)
# print("Other Region Mean:", other_mean)
# print("Other Region Standard Deviation:", other_std)
#
# # A simple comparison
# difference_in_means = np.abs(transitional_mean - other_mean)
# print("Absolute Difference in Means:", difference_in_means)