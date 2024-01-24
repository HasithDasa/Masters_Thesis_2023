import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis


def load_image(path):
    return np.load(path)

# Load and binarize masks
def load_and_binarize_mask(path):
    print("path", path)
    mask = np.load(path)
    mask[mask == 1] = 0
    mask[mask == 10] = 1
    return mask

def calculate_fourier_features(patch):
    # Compute the Fourier Transform
    f_transform = np.fft.fft2(patch)
    f_shifted = np.fft.fftshift(f_transform)  # Shift the zero frequency component to the center

    # Amplitude Spectrum
    amplitude_spectrum = np.abs(f_shifted)

    # Power Spectrum
    power_spectrum = amplitude_spectrum ** 2

    # Average Power
    avg_power = np.mean(power_spectrum)

    # Low/High-Frequency Power
    # Assuming the patch is square for simplicity
    mid = int(patch.shape[0] / 2)
    low_freq_power = np.sum(power_spectrum[:mid, :mid])
    high_freq_power = np.sum(power_spectrum[mid:, mid:])

    # Phase Information
    phase_spectrum = np.angle(f_shifted)
    phase_mean = np.mean(phase_spectrum)
    phase_variance = np.var(phase_spectrum)

    # Statistical Measures
    amplitude_mean = np.mean(amplitude_spectrum)
    amplitude_variance = np.var(amplitude_spectrum)
    power_mean = np.mean(power_spectrum)
    power_variance = np.var(power_spectrum)

    return np.array([avg_power, low_freq_power, high_freq_power, amplitude_mean, amplitude_variance, power_mean, power_variance, phase_mean, phase_variance])

def process_image_for_masked_regions(image, mask, label, patch_size):
    height, width = image.shape
    features = []
    labels = []

    for y in range(0, height - patch_size + 1):
        for x in range(0, width - patch_size + 1):
            if mask[y, x] == 1:  # Check if the current pixel in the mask is white
                patch = image[y:y + patch_size, x:x + patch_size]
                patch_features = calculate_fourier_features(patch)
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
save_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_6_fourier.csv'

# Save the DataFrame as a CSV file
df.to_csv(save_path, index=False)
