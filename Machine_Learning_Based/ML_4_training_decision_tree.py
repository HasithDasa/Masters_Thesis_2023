import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib

# Load your dataset
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_5_stat_normalized.csv'
df = pd.read_csv(df_path)

# Separate the dataset into two based on the label
df_zeros = df[df['Label'] == 0]
df_ones = df[df['Label'] == 1]

# Undersample the majority class
df_ones_undersampled = resample(df_ones,
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_zeros), # match number in minority class
                                 random_state=42)  # reproducible results

# Combine the minority class with the downsampled majority class
df = pd.concat([df_ones_undersampled, df_zeros])

# Count the number of zeros in the 'Label' column
count_zeros = (df['Label'] == 0).sum()

# Count the number of ones in the 'Label' column
count_ones = (df['Label'] == 1).sum()

# Print the counts
print(f"Count of Zeros: {count_zeros}")
print(f"Count of Ones: {count_ones}")



# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Select only the features of interest (Features 3 and 4)
# X = df[['Feature_3', 'Feature_1']]
X = df[['Feature_2']]
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
dt_clf.fit(X_train, y_train)

# Perform cross-validation on the training set
cv_scores = cross_val_score(dt_clf, X_train, y_train, cv=5)

# Predict on both the training and testing sets
y_train_pred = dt_clf.predict(X_train)
y_test_pred = dt_clf.predict(X_test)

# Calculate and compare accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Check for overfitting
if train_accuracy > test_accuracy:
    print("Model may be overfitting.")
else:
    print("Model seems fine, no significant overfitting detected.")

# Print out the accuracies and cross-validation scores
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.2f}")

# Save the trained model
model_filename = 'decision_tree_classifier.joblib'
joblib.dump(dt_clf, model_filename)
print(f"Decision Tree model saved as {model_filename}")
