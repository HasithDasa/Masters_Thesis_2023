import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure


# [crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
crop_starting_row = 140
crop_ending_row = 200
crop_starting_column = 240
crop_ending_column = 250

patch_size_rows = 3
patch_size_cols = 10


def load_image(path):
    return np.load(path)

# Load and binarize masks
def load_and_binarize_mask(path):
    print("path", path)
    mask = np.load(path)
    mask[mask == 1] = 0
    mask[mask == 10] = 1
    plt.imshow(mask)
    # plt.show()

    mask = mask[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
    # print("mask:", np.shape(mask))

    return mask

# Function to convert image to uint8
def convert_to_uint8(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    uint8_image = cv2.equalizeHist(normalized_image)

    return uint8_image.astype(np.uint8)  # Convert to uint8

def calculate_glcm_features_on_patch(patch):

    # lbp_patch = calculate_lbp(patch)
    # lbp_patch = lbp_patch.astype(np.uint8)
    # print(np.unique(lbp_patch))
    patch_uint8 = convert_to_uint8(patch)
    glcm = graycomatrix(patch_uint8, distances=[0], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    # contrast = graycoprops(glcm, 'contrast')
    # dissimilarity = graycoprops(glcm, 'dissimilarity')
    # homogeneity = graycoprops(glcm, 'homogeneity')
    # correlation = graycoprops(glcm, 'correlation')
    energy = graycoprops(glcm, 'energy')

    return np.hstack([energy]).flatten()

def process_image_for_masked_regions(image, mask, label, patch_size_rows, patch_size_cols):

    equalized_image = exposure.equalize_hist(image)

    plt.imshow(equalized_image)
    # plt.show()


    image = equalized_image[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]

    plt.imshow(image)
    # np.save("000abc.npy", image)


    height, width = image.shape
    # print("height:", height)
    # print("width:", width)
    features = []
    labels = []
    threshold = patch_size_rows * patch_size_cols / 2

    for y in range(0, height - patch_size_rows + 1, patch_size_rows):
        for x in range(0, width - patch_size_cols + 1, patch_size_cols):
            # if np.any(mask[y:y + patch_size_rows, x:x + patch_size_cols] == 1):  # Check if any pixel in the patch in the mask is white
            if np.sum(mask[y:y + patch_size_rows, x:x + patch_size_cols]) > threshold: # Check if the majority of pixels in the patch in the mask are white
                patch = image[y:y + patch_size_rows, x:x + patch_size_cols]
                patch_features = calculate_glcm_features_on_patch(patch)
                features.append(patch_features)
                labels.append(label)

    return features, labels

def get_matching_mask_path(image_path, mask_dir, mask_name_end):
    base_name = os.path.basename(image_path)
    mask_name = base_name.replace('.npy', mask_name_end)  # Replace .npy with ._turb.npy or _lamina.npy
    return os.path.join(mask_dir, mask_name).replace('\\', '/')

# Directories containing the images and masks
image_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/230920_164712/glcm"
mask_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/230920_164712/glcm/masks"
mask_name_end_turb = '_turbul.npy'
mask_name_end_lami = '_lami.npy'



all_features = []
all_labels = []

for image_name in os.listdir(image_dir):
    if image_name.endswith('.npy'):  # Ensure processing .npy files
        img_path = os.path.join(image_dir, image_name)
        print("img_path", img_path)
        image = load_image(img_path)

        # image = exposure.equalize_hist(image)

        # image[image < 296] = 296
        # image[image > 298] = 298

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


# # Skipping the first row (header)
# df_data = df.iloc[1:]

# # Identify rows to be removed by comparing numpy arrays directly
# temp_sav = (df_data.iloc[:-1, :-1].to_numpy() == df_data.iloc[1:, :-1].to_numpy()).all(axis=1) & (df_data.iloc[:-1, -1].to_numpy() != df_data.iloc[1:, -1].to_numpy())
#
# # Combine temp_sav for current and next rows
# temp_sav = temp_sav | np.roll(temp_sav, 1)
#
# # Adjust the length of temp_sav to match df_data
# temp_sav = np.append(temp_sav, False)
#
# # Apply temp_sav to df_data to filter out rows
# df_filtered = df_data[~temp_sav]


# Filter rows
df = df[~((df['Label'] == 1) & (df.drop('Label', axis=1) > 0.9).all(axis=1))]


# Assuming 'df' is your DataFram
save_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 4/glcm/5_240_250.csv'

# Save the DataFrame as a CSV file
df.to_csv(save_path, index=False)

