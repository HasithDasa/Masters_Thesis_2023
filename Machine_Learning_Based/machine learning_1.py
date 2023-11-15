import json
import os
import base64
import io
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the annotations
def load_annotations(path):
    with open(path, 'r') as file:
        return json.load(file)


# Decode RLE encoded mask
def rle_decode(mask_rle, shape):
    if isinstance(mask_rle, list):
        mask_rle = " ".join(map(str, mask_rle))
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1

    # Handle the case where starts and lengths have different lengths
    if len(starts) != len(lengths):
        min_len = min(len(starts), len(lengths))
        starts = starts[:min_len]
        lengths = lengths[:min_len]

    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape)


# Convert individual RLE masks to a single label matrix
def masks_to_labels(masks_rle, label_ids, shape):
    label_matrix = np.zeros(shape, dtype=int)
    for label_id, mask_rle in zip(label_ids, masks_rle):
        mask = rle_decode(mask_rle, shape)
        label_matrix[mask > 0] = label_id
    return label_matrix



# Extract features from LBP image
def extract_features(lbp_image, x, y):
    offsets = [-1, 0, 1]
    features = []
    for dx in offsets:
        for dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < lbp_image.shape[1] and 0 <= ny < lbp_image.shape[0]:
                # Ensure that the value being appended is a scalar
                value = lbp_image[ny, nx]
                if isinstance(value, np.ndarray):
                    value = value[0]  # Take the first value if it's an array
                features.append(value)
            else:
                features.append(0)
    return features


def display_masks(masks, shape):
    fig, axs = plt.subplots(1, len(masks), figsize=(20, 5))
    for i, mask in enumerate(masks):
        mask_decoded = rle_decode(mask, shape)
        axs[i].imshow(mask_decoded, cmap='gray')
        axs[i].set_title(f'Mask {i}')
        axs[i].axis('off')
    plt.show()

# Process data
def process_data(annotation_path, lbp_image_dir):
    annotations = load_annotations(annotation_path)

    all_features = []
    all_labels = []

    for entry in annotations:
        shape = (
        entry['annotations'][0]['result'][0]['original_height'], entry['annotations'][0]['result'][0]['original_width'])
        masks_rle = [entry['annotations'][0]['result'][i]['value']['rle'] for i in range(3)]
        display_masks(masks_rle, shape)

        labels = masks_to_labels(masks_rle, [0, 1, 2], shape)

        lbp_image_path = os.path.join(lbp_image_dir, entry['data']['image'].split("/")[-1])
        lbp_image = np.array(Image.open(lbp_image_path))

        for y in range(lbp_image.shape[0]):
            for x in range(lbp_image.shape[1]):
                features = extract_features(lbp_image, x, y)
                all_features.append(features)
                all_labels.append(labels[y, x])

    # Diagnostic code to identify inconsistent feature vectors
    # Diagnostic code to check the dimensions of the first few items
    print([len(feature) for feature in all_features[:10]])

    # Try converting the first few items to a numpy array
    try:
        subset_array = np.array(all_features[:10])
        print("Conversion successful for the subset.")
    except ValueError as e:
        print("Error encountered:", e)

    # Diagnostic code to examine the contents of the first few feature vectors
    print(all_features[:10])

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    X_train, X_val, y_train, y_val = train_test_split(all_features, all_labels, test_size=0.2)

    return X_train, X_val, y_train, y_val


# Usage
annotation_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated lbp json/annotated_lbp_r3_p28.json"
lbp_image_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/lbp_images_r3_p28/"
X_train, X_val, y_train, y_val = process_data(annotation_path, lbp_image_dir)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_clf.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = rf_clf.predict(X_val)

# Evaluate the model's performance
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy:", accuracy)

# unique_labels = np.unique(np.concatenate([y_val, y_val_pred]))
# print("Unique labels found:", unique_labels)
#
# # Adjust target names based on the unique labels
# target_names_map = {0: "Transitional", 1: "Other Regions", 2: "Background"}
# adjusted_target_names = [target_names_map[label] for label in unique_labels]
#
# class_report = classification_report(y_val, y_val_pred, target_names=adjusted_target_names)
# print(class_report)


# Load and process the image
image_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/lbp_images_r3_p28/d9fbc31b-lbp_irdata_0001_0013.png"
lbp_image = np.array(Image.open(image_path))

# Convert LBP image to grayscale (if it's not already)
if lbp_image.ndim == 3 and lbp_image.shape[2] == 4:
    lbp_image = lbp_image[:, :, 0]  # Assuming all channels are identical for LBP image

# Extract features for every pixel in the image
features = []
for y in range(lbp_image.shape[0]):
    for x in range(lbp_image.shape[1]):
        feature = extract_features(lbp_image, x, y)
        features.append(feature)

# Predict the label for each pixel
predictions = rf_clf.predict(features)
#
# # Reshape the prediction array to the shape of the original image
# predicted_image = predictions.reshape(lbp_image.shape)
#
# # Visualize the original and predicted images
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(lbp_image, cmap='gray')
# axes[0].set_title("Original LBP Image")
# axes[1].imshow(predicted_image, cmap='jet')
# axes[1].set_title("Predicted Regions")
# plt.show()

