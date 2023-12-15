import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_6_fourier_normalized - Copy.csv'
df = pd.read_csv(df_path)

# Define the features of interest and the label column
features_of_interest = ['Feature_9', 'Feature_4']
label_column = df.columns[-1]  # The last column as the label

# Boxplots for the selected features
for feature in features_of_interest:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

# Scatter plot between the two features, colored by the label
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Feature_9', y='Feature_4', hue=label_column)
plt.title('Scatter Plot of Feature_9 vs Feature_4')
plt.show()

# Histograms for the selected features
for feature in features_of_interest:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()
