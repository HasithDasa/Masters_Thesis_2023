import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import matplotlib.colors as colors


image = np.load('D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/irdata_0001_0038.npy')


lbp_roi = local_binary_pattern(image, P=80, R=10, method="uniform")

# Plotting
plt.imshow(lbp_roi)
plt.colorbar()
plt.title("Logarithmic Scale Display of NPY File")
plt.show()


# image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
# # Display the original image
# plt.figure()
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.colorbar()
# plt.show()
#
# # Cropping the image (rows 50-200, columns 150-200)
# # cropped_image = image[125:200, 150:200]
#
# cropped_image = image
#
# # Extract HOG features
# fd, hog_image = hog(cropped_image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True)
#
# # # Rescale histogram for better display
# # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
#
# lbp_roi = local_binary_pattern(hog_image, P=24, R=3, method="uniform")
#
#
# # Display the cropped image
# plt.figure()
# plt.title('hog_image Image')
# plt.imshow(hog_image, cmap='gray')
# plt.colorbar()
# plt.show()
#
#
#
#
#
# # Display the cropped image
# plt.figure()
# plt.title('lbp_roi Image')
# plt.imshow(lbp_roi, cmap='gray')
# plt.colorbar()
# plt.show()
#







# # Display the cropped image
# plt.figure()
# plt.title('Cropped Image')
# plt.imshow(cropped_image, cmap='gray')
# plt.colorbar()
# plt.show()
#
# # Apply histogram equalization to the cropped image
# equalized_image = exposure.equalize_hist(image)
#
# # image_copy = np.copy(image)
# # image_copy[image_copy < 296] = 296
# # image_copy[image_copy > 298] = 298
# #
# # image_copy.astype(int)
# #
# # adapt_equalized_image = exposure.equalize_adapthist(image_copy)
#
# equalized_cropped_image = equalized_image[125:200, 150:200]
#
#
# np.save('equalized_cropped_image.npy', equalized_cropped_image)
#
# # Display the equalized image
# plt.figure()
# plt.title('equalized_cropped_image')
# plt.imshow(equalized_cropped_image, cmap='gray')
# plt.colorbar()
# plt.show()

# # Display the equalized image
# plt.figure()
# plt.title('adapt_equalized_cropped_image')
# plt.imshow(adapt_equalized_cropped_image, cmap='gray')
# plt.colorbar()
# plt.show()