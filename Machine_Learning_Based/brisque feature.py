from skimage import io, color
from skimage.feature import local_binary_pattern
import imquality.brisque as brisque
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import exposure
from skimage.filters import difference_of_gaussians
from skimage.segmentation import chan_vese


def patch_image(image, patch_size):
    """ Divide the image into non-overlapping patches """
    rows, cols = image.shape
    patch_rows, patch_cols = patch_size

    # Calculate the number of patches in each dimension
    num_patches_rows = rows // patch_rows
    num_patches_cols = cols // patch_cols

    # Slice the image into patches
    patches = []
    for i in range(num_patches_rows):
        for j in range(num_patches_cols):
            patch = image[i*patch_rows:(i+1)*patch_rows, j*patch_cols:(j+1)*patch_cols]
            patches.append(patch)

    return patches

def evaluate_lbp_brisque_patches(image_path, patch_size):
    image = np.load(image_path)
    image_copy = np.copy(image)

    image_copy[image_copy < 296] = 296
    image_copy[image_copy > 298] = 298


    plt.figure()
    plt.title('Original Image')
    plt.imshow(image_copy, cmap='gray')
    plt.colorbar()
    plt.show()

    # exp = exposure.equalize_hist(image)

    # log = exposure.adjust_log(image_copy, 1)
    # gama = exposure.adjust_gamma(image_copy, 2)

    cv = chan_vese(image, mu=0.35, lambda1=1, lambda2=1, tol=1e-3,
                   max_num_iter=200, dt=0.5, init_level_set="checkerboard",
                   extended_output=True)

    # gaussian = difference_of_gaussians(image_copy, 5.5)

    lbp_image = local_binary_pattern(cv[1], 24, 3, method='uniform')

    cropped_lbp_image = lbp_image[125:200, 150:300]

    plt.figure()
    plt.title('gaussian lbp_image')
    plt.imshow(cropped_lbp_image, cmap='gray')
    plt.colorbar()
    plt.show()

    # Divide the image into patches
    patches = patch_image(cropped_lbp_image, patch_size)

    quality_scores = []

    for patch in patches:
        quality_score = cv2.quality.QualityBRISQUE_compute(patch, "./brisque_model_live.yml", "./brisque_range_live.yml")
        quality_scores.append(quality_score[0]/100)
        # print("quality_score", quality_score)

    return quality_scores

def calculate_histo_equali_patches(image_path, patch_size):
    image = np.load(image_path)

    # Apply histogram equalization to the image
    equalized_image = exposure.equalize_hist(image)

    equalized_cropped_image = equalized_image[125:200, 150:300]

    plt.figure()
    plt.title('equalized_cropped_image')
    plt.imshow(equalized_cropped_image, cmap='gray')
    plt.colorbar()
    plt.show()

    patches = patch_image(equalized_cropped_image, patch_size)

    mean_values = []

    for patch in patches:
        mean_value = np.mean(patch)
        mean_values.append(mean_value)

        # print("mean_value", mean_value)

    return mean_values


patch_size = (5, 150)  # 5 rows, 150 columns
image_name_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/irdata_0001_0001.npy"
scores = evaluate_lbp_brisque_patches(image_name_path, patch_size)
print("BRISQUE Scores for each patch:", scores)

mean_values = calculate_histo_equali_patches(image_name_path, patch_size)
print("mean_values of each patch:", mean_values)