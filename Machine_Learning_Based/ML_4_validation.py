import os
import joblib
import numpy as np
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def load_image(path):
    return np.load(path)

def convert_to_uint8(image):
    # Your conversion logic here, make sure it matches your training setup
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.equalizeHist(normalized_image)

def calculate_glcm_features_on_patch(patch):
    patch_uint8 = convert_to_uint8(patch)
    glcm = graycomatrix(patch_uint8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation]).flatten()

def process_image(image, model, patch_size=5, selected_feature_indices=None):
    if selected_feature_indices is None:
        selected_feature_indices = [16, 3, 0, 4, 1, 19, 2, 7, 17, 6, 5, 18, 8, 12]  # Example indices

    height, width = image.shape
    transitional_mask = np.zeros_like(image, dtype=np.uint8)
    feature_columns = [f'Feature_{i+1}' for i in selected_feature_indices]

    for y in range(0, height - patch_size + 1):
        for x in range(0, width - patch_size + 1):
            patch = image[y:y + patch_size, x:x + patch_size]
            patch_features = calculate_glcm_features_on_patch(patch)
            selected_features = patch_features[selected_feature_indices]

            # Create a DataFrame for the selected features
            patch_df = pd.DataFrame([selected_features], columns=feature_columns)
            prediction = model.predict(patch_df)

            if prediction == 1:  # Transitional region
                transitional_mask[y:y + patch_size, x:x + patch_size] = 255

    return transitional_mask

def mark_transitional_regions(image, mask):
    # Convert image to uint8 if it's not already
    if image.dtype != np.uint8:
        image_uint8 = convert_to_uint8(image)
    else:
        image_uint8 = image

    marked_image = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    marked_image[mask == 255] = [0, 255, 0]  # Mark with green color
    return marked_image


# Load the saved model
model_filename = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated lbp/glcm_rf_classifier.joblib'
rf_model = joblib.load(model_filename)

# Load a new thermal image (npy format)
new_image_path_1 = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/irdata_0001_0001.npy'  # Replace with the actual path
thermal_image_1 = load_image(new_image_path_1)

# Process the image and get transitional regions
transitional_mask_1 = process_image(thermal_image_1, rf_model, selected_feature_indices=[16, 3, 0, 4, 1, 19, 2, 7, 17, 6, 5, 18, 8, 12])

# Mark the transitional regions on the original image
marked_image_1 = mark_transitional_regions(thermal_image_1, transitional_mask_1)


##################testing

# Load a new thermal image (npy format)
new_image_path_2 = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/irdata_0001_0023.npy'  # Replace with the actual path
thermal_image_2 = load_image(new_image_path_2)

# Process the image and get transitional regions
transitional_mask_2 = process_image(thermal_image_2, rf_model, selected_feature_indices=[16, 3, 0, 4, 1, 19, 2, 7, 17, 6, 5, 18, 8, 12])

# Mark the transitional regions on the original image
marked_image_2 = mark_transitional_regions(thermal_image_2, transitional_mask_2)




# Display the marked image
cv2.imshow('Transitional Regions_1', marked_image_1)
cv2.imshow('Transitional Regions_2', marked_image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
