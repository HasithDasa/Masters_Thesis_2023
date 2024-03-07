import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure

# Initial crop values (will be updated in the loop)
crop_starting_row = 90
crop_ending_row = 150
# crop_starting_column and crop_ending_column will be set in the loop

patch_size_rows = 3
patch_size_cols = 10

# Define the complete range for columns
complete_start_column = 200
complete_end_column = 290

def load_image(path):
    return np.load(path)

def load_and_binarize_mask(path):
    print("path", path)
    mask = np.load(path)
    mask[mask == 1] = 0
    mask[mask == 10] = 1
    plt.imshow(mask)
    mask = mask[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
    return mask

def convert_to_uint8(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    uint8_image = cv2.equalizeHist(normalized_image)
    return uint8_image.astype(np.uint8)

def calculate_glcm_features_on_patch(patch):
    patch_uint8 = convert_to_uint8(patch)
    glcm = graycomatrix(patch_uint8, distances=[0], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    energy = graycoprops(glcm, 'energy')
    return np.hstack([energy]).flatten()

def process_image_for_masked_regions(image, mask, label, patch_size_rows, patch_size_cols):
    equalized_image = exposure.equalize_hist(image)
    plt.imshow(equalized_image)
    image = equalized_image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
    plt.imshow(image)
    height, width = image.shape
    features = []
    labels = []
    threshold = patch_size_rows * patch_size_cols / 2
    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            if np.sum(mask[y:y + patch_size_rows, x:x + patch_size_cols]) > threshold:
                patch = image[y:y + patch_size_rows, x:x + patch_size_cols]
                patch_features = calculate_glcm_features_on_patch(patch)
                features.append(patch_features)
                labels.append(label)
    return features, labels

def get_matching_mask_path(image_path, mask_dir, mask_name_end):
    base_name = os.path.basename(image_path)
    mask_name = base_name.replace('.npy', mask_name_end)
    return os.path.join(mask_dir, mask_name).replace('\\', '/')

image_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/231002_170018/glcm"
mask_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/231002_170018/glcm/masks"
mask_name_end_turb = '_turbul.npy'
mask_name_end_lami = '_lami.npy'

# Iterate over the range with step size equal to patch_size_cols
for start_col in range(complete_start_column, complete_end_column, patch_size_cols):
    crop_starting_column = start_col
    crop_ending_column = min(start_col + patch_size_cols, complete_end_column)

    all_features = []
    all_labels = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.npy'):
            img_path = os.path.join(image_dir, image_name)
            print("img_path", img_path)
            image = load_image(img_path)

            turb_mask_path = get_matching_mask_path(img_path, mask_dir, mask_name_end_turb)
            lamina_mask_path = get_matching_mask_path(img_path, mask_dir, mask_name_end_lami)

            turb_mask = load_and_binarize_mask(turb_mask_path)
            lamina_mask = load_and_binarize_mask(lamina_mask_path)

            turb_features, turb_labels = process_image_for_masked_regions(image, turb_mask, 1, patch_size_rows, patch_size_cols)
            lami_features, lami_labels = process_image_for_masked_regions(image, lamina_mask, 0, patch_size_rows, patch_size_cols)

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

    df = df[~((df['Label'] == 1) & (df.drop('Label', axis=1) > 0.9).all(axis=1))]

    save_path = f'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 3/glcm/systematic_error/patch_10_200_290/Sys_error_{crop_starting_column}_{crop_ending_column}.csv'
    df.to_csv(save_path, index=False)
