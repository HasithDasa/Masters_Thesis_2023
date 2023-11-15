import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from scipy.ndimage import convolve
from skimage.filters import gabor_kernel
from numpy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt


# Load thermal image
thermal_image = np.load('D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/irdata_0001_0002.npy')

# Load and binarize masks
def load_and_binarize_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 255
    return mask

background_mask = load_and_binarize_mask('D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated lbp/ir_data_0001_0002_annotation-3-by-1-tag-Background-0.png')
transitional_mask = load_and_binarize_mask('D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated lbp/ir_data_0001_0002_annotation-3-by-1-tag-Transitional-0.png')
other_mask = load_and_binarize_mask('D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated lbp/ir_data_0001_0002_annotation-3-by-1-tag-Other Regions-0.png')


# Apply masks to LBP image
def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)



def calculate_glcm_features_on_patch(image):
    # Find non-zero coordinates in the image
    coords = np.column_stack(np.where(image > 0))

    # Calculate bounding box of the non-zero regions
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the non-zero region (patch)
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    # Convert cropped image to uint8
    cropped_image_uint8 = convert_to_uint8(cropped_image)

    #print("unique values:", np.unique(cropped_image_uint8))

    # Calculate GLCM on the cropped patch
    # glcm = graycomatrix(cropped_image_uint8, distances=[1, 2, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    #                     symmetric=True, normed=True)

    glcm = graycomatrix(cropped_image_uint8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        symmetric=True, normed=True)

    # Compute Haralick features on the GLCM
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    # Return the features
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])


def calculate_lbp_features_on_patch(image, radius=3, n_points=None, method='uniform', display_hist=False):
    if n_points is None:
        n_points = 8 * radius

    # Find non-zero coordinates in the image
    coords = np.column_stack(np.where(image > 0))

    # Calculate bounding box of the non-zero regions
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the non-zero region (patch)
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    # Calculate the LBP image on the cropped patch
    lbp_image = local_binary_pattern(cropped_image, n_points, radius, method)

    # Calculate the histogram of the LBP
    lbp_hist, bin_edges = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize the histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    if display_hist:
        # Display the histogram
        plt.figure(figsize=(8, 4))
        plt.bar(bin_edges[:-1], lbp_hist, width=0.5, color='blue')
        plt.title("LBP Histogram")
        plt.xlabel("LBP Value")
        plt.ylabel("Normalized Frequency")
        plt.show()

    return lbp_hist


# Function to apply Gabor filters, extract features, and visualize the filtered image
def apply_and_visualize_gabor(image, frequency, theta, sigma_x, sigma_y, gamma, psi):

    # Find non-zero coordinates in the image
    coords = np.column_stack(np.where(image > 0))

    # Calculate bounding box of the non-zero regions
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the non-zero region (patch)
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    # Create Gabor kernel with adjusted sigma_x and sigma_y based on gamma
    kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y / gamma)

    # Filter the image with the kernel
    filtered_real = convolve(cropped_image, np.real(kernel))
    filtered_imag = convolve(cropped_image, np.imag(kernel))
    filtered = filtered_real + 1j * filtered_imag

    # Compute magnitude
    magnitude = np.abs(filtered)

    # Visualize the filtered image
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(filtered_real, cmap='gray')
    plt.title('Real part of Gabor filter')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(filtered_imag, cmap='gray')
    plt.title('Imaginary part of Gabor filter')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return magnitude.ravel()


def calculate_fourier_transform_features_on_patch(image, display_fft=False):

    # Find non-zero coordinates in the image
    coords = np.column_stack(np.where(image > 0))

    # Calculate bounding box of the non-zero regions
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the non-zero region (patch)
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    # Apply Fourier Transform
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)  # Shift the zero frequency component to the center of the spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shifted)+1)  # Magnitude spectrum for visualization

    # Extract features from the magnitude
    # Here we use the magnitude as the feature directly,
    # but you can also compute statistics (mean, variance) from the magnitude
    fourier_features = np.abs(f_shifted).ravel()

    if display_fft:

        # plt.subplot(122)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.axis('off')
        plt.show()

        # Inverse Fourier Transform to reconstruct the image
        f_ishifted = fftshift(f_shifted)
        img_reconstructed = ifft2(f_ishifted)
        img_reconstructed = np.abs(img_reconstructed)

        # Visualize the reconstructed image
        plt.figure(figsize=(6, 6))
        plt.imshow(img_reconstructed, cmap='gray')
        plt.title('Reconstructed Image')
        plt.axis('off')
        plt.show()

    return fourier_features


def draw_rectangle_on_patch(image, color=(0, 255, 0), thickness=2):
    # Find non-zero coordinates in the image
    coords = np.column_stack(np.where(image > 0))

    # Calculate bounding box of the non-zero regions
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Convert image to uint8 if it's not already
    if image.dtype != np.uint8:
        image = convert_to_uint8(image)

    # Create a copy of the original image to draw the rectangle
    image_with_rectangle = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the rectangle
    cv2.rectangle(image_with_rectangle, (x_min, y_min), (x_max, y_max), color, thickness)

    return image_with_rectangle, (x_min, y_min, x_max, y_max)


# Function to convert image to uint8
def convert_to_uint8(image):
    # image = image - np.min(image)  # Ensure the minimum value is 0
    # image = (image / np.max(image)) * 255  # Normalize to the range 0-255
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    uint8_image = cv2.equalizeHist(normalized_image)

    return uint8_image.astype(np.uint8)  # Convert to uint8


# Apply masks to original thermal image
thermal_background = apply_mask(thermal_image, background_mask)
thermal_transitional = apply_mask(thermal_image, transitional_mask)
thermal_other = apply_mask(thermal_image, other_mask)


# Calculate and print Haralick textures for the transitional region of the original thermal image
haralick_features_transitional = calculate_glcm_features_on_patch(thermal_transitional)
print("Haralick features for the transitional region:", haralick_features_transitional)
print("Haralick features size:", np.size(haralick_features_transitional))

lbp_features_transitional = calculate_lbp_features_on_patch(thermal_transitional, display_hist=True)
print("lbp_features for the transitional region:", lbp_features_transitional)
print("lbp_features size:", np.size(lbp_features_transitional))

# Gabor filter parameters
frequency = 0.1  # Example frequency
theta = 0  # Orientation near horizontal
sigma_x = 3  # Larger value since texture is more spread out horizontally
sigma_y = 1  # Smaller value for the y-axis
gamma = 0.1  # Smaller than 1 for elliptical shape (more stretched along y-axis)
psi = 0  # Phase offset

# Apply Gabor filter, extract features, and visualize
gabor_features_transitional = apply_and_visualize_gabor(thermal_transitional, frequency, theta, sigma_x, sigma_y, gamma, psi)
print("gabor_features for the transitional region:", gabor_features_transitional)
print("gabor_features size:", np.size(gabor_features_transitional))


# Apply fourier, extract features, and visualize
fourier_features_transitional = calculate_fourier_transform_features_on_patch(thermal_transitional, display_fft=True)
print("fourier_features for the transitional region:", fourier_features_transitional)
print("fourier_features size:", np.size(fourier_features_transitional))



# Assuming 'thermal_transitional' is the image array you want to process
image_with_patch, patch_coords = draw_rectangle_on_patch(thermal_transitional)
# Display the image with the patch and relavent
cv2.imshow('Patch on Image', image_with_patch)
cv2.waitKey(0)
cv2.destroyAllWindows()

















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