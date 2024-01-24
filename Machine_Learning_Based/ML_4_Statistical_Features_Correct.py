import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.feature import local_binary_pattern
import cv2

# [crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
crop_starting_row = 100
crop_ending_row = 175
crop_starting_column = 200
crop_ending_column = 320

patch_size_rows = 3
patch_size_cols = 120


def load_image(path):
    return np.load(path)

# Load and binarize masks
def load_and_binarize_mask(path):
    print("path", path)
    mask = np.load(path)
    mask[mask == 1] = 0
    mask[mask == 10] = 1
    mask = mask[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
    return mask

def calculate_statistical_moments(patch):
    mean = np.mean(patch)
    median = np.median(patch)
    # variance = np.var(patch)
    # skewness = skew(patch.flatten())
    # kurt = kurtosis(patch.flatten())

    return np.array([mean, median])

def process_image_for_masked_regions(image, mask, label, patch_size_rows, patch_size_cols):

    equalized_image = np.copy(image)
    equalized_cropped_image = equalized_image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]

    height, width = equalized_cropped_image.shape

    features = []
    labels = []
    threshold = patch_size_rows * patch_size_cols / 2

    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            # if np.any(mask[y:y + patch_size_rows, x:x + patch_size_cols] == 1):  # Check if any pixel in the patch in the mask is white
            if np.sum(mask[y:y + patch_size_rows,x:x + patch_size_cols]) > threshold:  # Check if the majority of pixels in the patch in the mask are white
                patch_equilized = equalized_cropped_image[y:y + patch_size_rows, x:x + patch_size_cols]

                patch_features = calculate_statistical_moments(patch_equilized)

                features.append(patch_features)
                labels.append(label)

    return features, labels


def get_matching_mask_path(image_path, mask_dir, mask_name_end):
    base_name = os.path.basename(image_path)
    mask_name = base_name.replace('.npy', mask_name_end)  # Replace .npy with ._turb.npy or _lamina.npy
    return os.path.join(mask_dir, mask_name).replace('\\', '/')

# Directories containing the images and masks
image_dir = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/normalized_stat'
mask_dir = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/normalized_stat/masks'
mask_name_end_turb = '_turbul.npy'
mask_name_end_lami = '_lami.npy'


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

        # plt.imshow(turb_mask)
        # plt.show()

        # plt.imshow(lamina_mask)
        # plt.show()

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
# Skipping the first row (header)
df_data = df.iloc[1:]
# # Filter rows
df_data = df_data[~((df_data['Label'] == 0) & (df_data.drop('Label', axis=1) < 0).any(axis=1))]
# df_data = df_data[~((df_data['Label'] == 1) & (df_data.drop('Label', axis=1) > 0).any(axis=1))]
final_df = df_data[~((df_data['Label'] == 1) & (df_data.iloc[:, 0:2] > 0).any(axis=1))]


# Assuming 'df' is your DataFram
save_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_18_stat.csv'

# Save the DataFrame as a CSV file
final_df.to_csv(save_path, index=False)
