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
    lbp_image = local_binary_pattern(image, lbp_n_points, lbp_radius, method='uniform')
    return lbp_image

def custom_glcm(image, distances, angles, levels):
    max_val = np.max(image)
    image_scaled = np.floor((image / max_val) * (levels - 1)).astype(int)
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)

    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            dx = int(round(distance * np.cos(angle)))
            dy = int(round(distance * np.sin(angle)))

            for x in range(image.shape[1]):
                for y in range(image.shape[0]):
                    if (x + dx >= 0 and x + dx < image.shape[1] and y + dy >= 0 and y + dy < image.shape[0]):
                        row = image_scaled[y, x]
                        col = image_scaled[y + dy, x + dx]
                        glcm[row, col, d, a] += 1

    return glcm

def custom_graycoprops(glcm, prop='contrast'):
    result = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float64)

    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            # Contrast
            if prop == 'contrast':
                contrast = 0.0
                for i in range(glcm.shape[0]):
                    for j in range(glcm.shape[1]):
                        contrast += (i - j) ** 2 * glcm[i, j, d, a]
                result[d, a] = contrast
            # Dissimilarity
            elif prop == 'dissimilarity':
                dissimilarity = 0.0
                for i in range(glcm.shape[0]):
                    for j in range(glcm.shape[1]):
                        dissimilarity += abs(i - j) * glcm[i, j, d, a]
                result[d, a] = dissimilarity
            # Homogeneity
            elif prop == 'homogeneity':
                homogeneity = 0.0
                for i in range(glcm.shape[0]):
                    for j in range(glcm.shape[1]):
                        homogeneity += glcm[i, j, d, a] / (1.0 + (i - j) ** 2)
                result[d, a] = homogeneity
            # Energy
            elif prop == 'energy':
                energy = np.sum(glcm[:, :, d, a] ** 2)
                result[d, a] = energy
            # Correlation
            elif prop == 'correlation':
                # Calculate means and standard deviations of the row and column sums
                row_sum = np.sum(glcm[:, :, d, a], axis=1)
                col_sum = np.sum(glcm[:, :, d, a], axis=0)
                row_mean = np.dot(np.arange(glcm.shape[0]), row_sum)
                col_mean = np.dot(np.arange(glcm.shape[1]), col_sum)
                row_std = np.sqrt(np.dot((np.arange(glcm.shape[0]) - row_mean) ** 2, row_sum))
                col_std = np.sqrt(np.dot((np.arange(glcm.shape[1]) - col_mean) ** 2, col_sum))

                correlation = 0.0
                for i in range(glcm.shape[0]):
                    for j in range(glcm.shape[1]):
                        correlation += (i - row_mean) * (j - col_mean) * glcm[i, j, d, a]
                result[d, a] = correlation / (row_std * col_std)

    return result

def calculate_glcm_features_on_lbp_patch(patch, levels=256):
    lbp_patch = calculate_lbp(patch)
    glcm = custom_glcm(lbp_patch, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels)

    contrast = custom_graycoprops(glcm, 'contrast')
    dissimilarity = custom_graycoprops(glcm, 'dissimilarity')
    homogeneity = custom_graycoprops(glcm, 'homogeneity')
    energy = custom_graycoprops(glcm, 'energy')
    correlation = custom_graycoprops(glcm, 'correlation')

    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation]).flatten()

def process_image_for_masked_regions(image, mask, label, patch_size):
    height, width = image.shape
    features = []
    labels = []

    for y in range(0, height - patch_size + 1):
        for x in range(0, width - patch_size + 1):
            if mask[y, x] == 1:  # Check if the current pixel in the mask is white
                patch = image[y:y + patch_size, x:x + patch_size]
                patch_features = calculate_glcm_features_on_lbp_patch(patch)
                features.append(patch_features)
                labels.append(label)

    return features, labels


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
save_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_4.csv'

# Save the DataFrame as a CSV file
df.to_csv(save_path, index=False)

