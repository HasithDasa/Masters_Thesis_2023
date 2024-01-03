import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your CSV file
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_6_fourier_normalized.csv'

# Load the CSV file
df = pd.read_csv(df_path).head(75100)

# # Scatter plot for two features
# sns.scatterplot(x='Feature_1', y='Feature_2', hue='Label', data=df)
# plt.title('Scatter Plot of Feature1 vs Feature2')
# plt.show()
#
# # Box plot for a single feature
# sns.boxplot(x='Label', y='Feature_1', data=df)
# plt.title('Box Plot of Feature1 by Label')
# plt.show()

# Pair plot for all features
sns.pairplot(df, hue='Label')
plt.show()

# Histogram for each feature
# for column in df.columns[:-1]:  # Exclude the label column
#     plt.figure()
#     sns.histplot(df, x=column, hue='Label', multiple='stack')
#     plt.title(f'Histogram of {column}')
#     plt.show()

# # Correlation heatmap
# correlation_matrix = df.iloc[:, :-1].corr()  # Exclude the label column
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True)
# plt.title('Correlation Heatmap of Features')
# plt.show()
