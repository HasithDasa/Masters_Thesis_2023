import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load your dataset
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 4/glcm/systematic_error/Unsupervised_DBSCAN/patch_5/Sys_error_265_270.csv' # Update this path
df = pd.read_csv(df_path)

# Assume you're working with two specific features for visualization, update these as necessary
feature_1 = 'Feature_1'
feature_2 = 'Feature_2'

# Scaling the features
X = df[[feature_1, feature_2]].values
X_scaled = StandardScaler().fit_transform(X)

# Load the best DBSCAN model for this dataset
model_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 4/glcm/systematic_error/Unsupervised_DBSCAN/patch_5/Sys_error_265_270_best_dbscan.joblib' # Update this path
dbscan = joblib.load(model_path)

# Predict the clusters using the original DBSCAN labels for reference
original_labels = dbscan.labels_

# Apply reclustering logic here if it's not already applied
# For the sake of example, let's assume you've applied the reclustering logic to obtain y_reclustered
# This code snippet does not show the reclustering logic itself; refer to the previous explanation for that

# Example reclustering logic (to be replaced with actual reclustering code)
unique, counts = np.unique(original_labels[original_labels != -1], return_counts=True)
largest_cluster = unique[np.argmax(counts)]
y_reclustered = np.where(original_labels == largest_cluster, 0, 1)

# Plotting
plt.figure(figsize=(10, 6))

# Define a color for each of the two new clusters (and optionally noise)
color_map = {0: 'red', 1: 'blue', -1: 'black'}  # Example: red for largest cluster, blue for others, black for noise

# Iterate over unique labels in the reclustering result
for label in np.unique(y_reclustered):
    idx = y_reclustered == label
    plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1], s=10, color=color_map.get(label, 'gray'), label=f'Cluster {label}')

plt.title('Reclustered DBSCAN Clustering')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.legend()
plt.show()
