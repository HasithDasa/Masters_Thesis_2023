import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Path to your CSV file
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_5_stat.csv'

# Load the CSV file
df = pd.read_csv(df_path)

# Select all columns except the last one for normalization
columns_to_normalize = df.columns[:-1]

# Initialize a MinMax Normalization
scaler = MinMaxScaler()

# Fit and transform the data
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Path for saving the normalized CSV file
normalized_csv_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_5_stat_normalized.csv'

# Save the normalized DataFrame to the specified path
df.to_csv(normalized_csv_path, index=False)

print("Dataframe has been normalized and saved as 'normalized.csv' at the specified path.")
