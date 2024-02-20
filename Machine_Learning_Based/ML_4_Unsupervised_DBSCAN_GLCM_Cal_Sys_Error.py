import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import joblib


# Function to perform grid search for DBSCAN
def grid_search_dbscan(X, y_true, eps_values, min_samples_values):
    best_ari = -1  # Initialize best ARI score
    best_nmi = -1  # Initialize best NMI score
    best_params = {'eps': None, 'min_samples': None}
    best_model = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            y_pred = dbscan.labels_

            # Only evaluate if more than one cluster is found and not all points are noise
            if len(set(y_pred)) > 1 and np.any(y_pred != -1):
                ari = adjusted_rand_score(y_true, y_pred)
                nmi = normalized_mutual_info_score(y_true, y_pred)

                if ari > best_ari:  # Update the best model if current ARI is greater
                    best_ari = ari
                    best_nmi = nmi
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_model = dbscan

    return best_model, best_params, best_ari, best_nmi


# Directories for data and model saving
data_dir = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 4/glcm/systematic_error/Unsupervised_DBSCAN/patch_5'
model_save_dir = data_dir  # Update this path as needed for saving models
results_file_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 4/glcm/systematic_error/Unsupervised_DBSCAN/patch_5/dbscan_grid_search_results.xlsx'

# Define ranges for eps and min_samples for grid search
eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = range(2, 10)

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=['CSV File', 'Best Eps', 'Best Min_Samples', 'Best ARI', 'Best NMI'])

# Iterate through all CSV files in the directory
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df_path = os.path.join(data_dir, file)
        df = pd.read_csv(df_path)

        # Separate features from labels
        X = df.drop('Label', axis=1).values
        y_true = df['Label'].values

        # Scale the features
        X_scaled = StandardScaler().fit_transform(X)

        # Perform grid search
        best_model, best_params, best_ari, best_nmi = grid_search_dbscan(X_scaled, y_true, eps_values,
                                                                         min_samples_values)

        # Get cluster labels from the best model
        y_pred = best_model.labels_

        # Identify the largest cluster
        unique, counts = np.unique(y_pred[y_pred != -1], return_counts=True)  # Exclude noise points
        largest_cluster = unique[np.argmax(counts)]

        # Recluster: largest cluster gets label 0, all others (including noise) get label 1
        y_reclustered = np.where(y_pred == largest_cluster, 0, 1)


        # Append results to the DataFrame
        results_df = results_df.append({
            'CSV File': file,
            'Best Eps': best_params['eps'],
            'Best Min_Samples': best_params['min_samples'],
            'Best ARI': best_ari,
            'Best NMI': best_nmi
        }, ignore_index=True)

        # Save the best model for each file
        model_filename = f'{file[:-4]}_best_dbscan.joblib'
        joblib.dump(best_model, os.path.join(model_save_dir, model_filename))

# Save the results to an Excel file
results_df.to_excel(results_file_path, index=False)
