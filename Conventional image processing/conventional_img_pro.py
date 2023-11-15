import cv2
import numpy as np

# Load the .npy image file
# img = np.load('D:/Academic/MSc/Thesis/data/documents-export-2023-06-29/1520.npy')
img = np.load('D:/Academic/MSc/Thesis/data/new data/npy/irdata_0001_0001.npy')

# Convert to uint8
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Apply median filter to remove "salt and pepper" noise
img = cv2.medianBlur(img, 5)

# Threshold the image to separate the blade from the background
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour, which should be the blade
blade_contour = max(contours, key=cv2.contourArea)

# Create a mask of the blade, but erode it to exclude the boundary
mask = np.zeros_like(img)
cv2.drawContours(mask, [blade_contour], -1, (255), thickness=cv2.FILLED)
kernel = np.ones((5, 5),np.uint8)
mask = cv2.erode(mask, kernel, iterations = 10)

# Apply the mask to the original image to get only the internal part of the blade
blade_internal = cv2.bitwise_and(img, img, mask=mask)

# Calculate the gradient using Sobel operator
# sobelx = cv2.Sobel(blade_internal, cv2.CV_64F, 1, 0, ksize=5)  # x
sobely = cv2.Sobel(blade_internal, cv2.CV_64F, 0, 1, ksize=5)  # y
# gradient_magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
gradient_magnitude = sobely


# Threshold on the gradient magnitude
_, gradient_mask = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

gradient_mask = cv2.medianBlur(gradient_mask.astype(np.uint8), 3)

# Threshold the image to binarize it
_, binary = cv2.threshold(gradient_mask, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours_1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming that the two largest contours are the arches
contours_1 = sorted(contours_1, key=cv2.contourArea, reverse=True)[:2]

# Create an empty mask to draw the contours on
mask_1 = np.zeros_like(gradient_mask)

# Draw the two arches (contours) on the mask
cv2.drawContours(mask_1, contours_1, -1, (255), thickness=cv2.FILLED)

# Invert the mask to get the area between the arches
mask_1 = cv2.bitwise_not(mask_1)

# Bitwise-AND the mask and the original image to get the dots
borders = cv2.bitwise_and(gradient_mask, gradient_mask, mask=mask_1)

dots = gradient_mask - borders

# Find the coordinates of the non-zero pixels
x_coords, _ = np.nonzero(dots)

# print("x_coords", x_coords)

# Determine the y-coordinate where the line will pass through by calculating the median
y_coord = np.median(x_coords)

print("trans line", y_coord+14)
print("median", y_coord)

# Draw the line on the image
output = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)
cv2.line(output, (0, int(y_coord)), (output.shape[1], int(y_coord)), (0, 255, 0), 2)

# Display the result
cv2.imshow("Perpendicular Line", output)

# Display the image with the filtered dots
cv2.imshow('Filtered Dots', dots)
cv2.imshow('Original img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()