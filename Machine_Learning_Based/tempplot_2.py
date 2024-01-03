import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Load your dataset
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_5_stat.csv'
df = pd.read_csv(df_path).head(50000)  # Load only the first 50000 rows

# Define the features of interest
features = ['Feature_4']
label_column = df.columns[-1]  # Assuming the last column is the label

# Plot histograms for each feature, separated by the label
for feature in features:
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=feature, hue=label_column, element='step', stat='density', common_norm=False)
    plt.title(f'Histogram of {feature} by Label')
    plt.show()

