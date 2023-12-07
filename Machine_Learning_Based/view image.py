import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
# Replace 'your_file_path.npy' with the path to your .npy file
thermal_image = np.load('D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/irdata_0001_0024.npy')
print(np.unique(thermal_image))
# Find max and min values
max_value = np.max(thermal_image)
min_value = np.min(thermal_image)

# Print max and min values
print(f"Max Value: {max_value}")
print(f"Min Value: {min_value}")

# Display the image as a heatmap
plt.imshow(thermal_image, cmap='viridis')
plt.colorbar()
plt.title("Thermal Image Heatmap")
plt.show()
