import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Path to your CSV file
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_6_fourier.csv'

# Load the CSV file
df = pd.read_csv(df_path)

# Select all columns except the last one for normalization
columns_to_normalize = df.columns[:-1]

# Initialize a MinMax Normalization
scaler = MinMaxScaler()

# Fit and transform the data
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Path for saving the normalized CSV file
normalized_csv_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_6_fourier_normalized_test.csv'

# Save the normalized DataFrame to the specified path
df.to_csv(normalized_csv_path, index=False)

# scaler_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_6_stat_normalized_scaler.joblib'
# joblib.dump(scaler, scaler_path)

print("Dataframe has been normalized and saved as 'normalized.csv' at the specified path.")


df_reduced = df[["Feature_4", "Feature_3", "Feature_1", "Feature_6"]]

# Initialize and fit the scaler
scaler_2 = MinMaxScaler()
scaler_2.fit(df_reduced)

# Save the scaler for later use
scaler_path_2 = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/fourier_feature_normalized_scaler_3.joblib'
joblib.dump(scaler_2, scaler_path_2)