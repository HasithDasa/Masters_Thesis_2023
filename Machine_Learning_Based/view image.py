import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load the .npy file
# Replace 'your_file_path.npy' with the path to your .npy file
thermal_image = np.load('D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/npy/irdata_0001_0005.npy')

##'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/irdata_0001_0005.npy'

print(np.unique(thermal_image))
# Find max and min values
max_value = np.max(thermal_image)
min_value = np.min(thermal_image)



# # Display the image as a heatmap
# plt.imshow(thermal_image, cmap='viridis')
#
# plt.title("original image")
# plt.show()

# Round data to 14 decimal places
thermal_image[thermal_image < 296] = 296
thermal_image[thermal_image > 298] = 298
print(np.unique(thermal_image))

vmax = np.max(thermal_image)
vmin = np.min(thermal_image)

# Print max and min values
print(f"Max Value: {vmax}")
print(f"Min Value: {vmin}")

# Normalize the data to focus on 296 to 297 range
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Display the image
plt.imshow(thermal_image, cmap='viridis', norm=norm)

# Add a colorbar with precision
plt.colorbar(format='%.64f')

plt.show()