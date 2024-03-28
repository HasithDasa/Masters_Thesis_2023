import numpy as np
import matplotlib.pyplot as plt
import os

file_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/Validation old/irdata_0019_0493.npy'
save_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/corotating_231207/glcm/validation/Validation old/edited_0019_0493.npy'

image = np.load(file_path)

# Initialize a list to hold the first non-zero transition row indices for each column
transition_indices = []

# Iterate through each column to find the transition from zero to non-zero
for col_idx in range(image.shape[1]):
    column = image[:, col_idx]
    zero_indices = np.where(column == 0)[0]
    non_zero_indices = np.where(column != 0)[0]

    # Find the first non-zero index that comes after at least one zero, if any
    valid_transitions = non_zero_indices[
        np.nonzero(non_zero_indices > zero_indices[-1])[0]] if zero_indices.size > 0 else []

    if valid_transitions.size > 0:
        transition_indices.append(valid_transitions[0])

# Determine the lowest (earliest) transition row index across all columns
lowest_transition_row_index = min(transition_indices) if transition_indices else -1

cropped_image = image[lowest_transition_row_index:, :]

plt.imshow(cropped_image, cmap='gray')  # Use 'cmap' as needed, e.g., for grayscale images
plt.axis('off')  # Hide axes
plt.show()

np.save(save_path, cropped_image)