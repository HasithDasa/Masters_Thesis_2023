import numpy as np

# Load the numpy array from the file
file_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/irdata_0001_0005.npy'
data = np.load(file_path)

# Function to count decimal points
def count_decimal_points(number):
    """ Count the number of decimal places in a number. """
    number_str = str(number)
    if '.' in number_str:
        return len(number_str.split('.')[1])
    else:
        return 0

# Apply the function to each element in the array and find the maximum decimal count
decimal_counts = np.vectorize(count_decimal_points)(data)
max_decimal_count = np.max(decimal_counts)
max_decimal_count_indices = np.where(decimal_counts == max_decimal_count)

# Get the values with the highest number of decimal points
values_with_max_decimals = data[max_decimal_count_indices]

# Find the maximum and minimum values in the array
max_value = np.max(data)
min_value = np.min(data)

# Find the indices of the maximum and minimum values
max_value_index = np.where(data == max_value)
min_value_index = np.where(data == min_value)

# Output results
print("Maximum number of decimal points:", max_decimal_count)
# print("Values with maximum decimal points:", values_with_max_decimals.tolist())
print("Max value:", max_value)
# print("Max value indices:", max_value_index)
print("Min value:", min_value)
# print("Min value indices:", min_value_index)
