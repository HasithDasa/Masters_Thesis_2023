import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.utils import resample
import joblib


# Load your dataset
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_14_stat.csv'
df = pd.read_csv(df_path)

# Count the number of zeros in the 'Label' column
count_zeros = (df['Label'] == 0).sum()

# Count the number of ones in the 'Label' column
count_ones = (df['Label'] == 1).sum()

# Print the counts
print(f"Count of Zeros b4: {count_zeros}")
print(f"Count of Ones b4: {count_ones}")


# # Count the number of ones in each row (excluding the label column)
# ones_count = (df.drop('Label', axis=1) == 1).sum(axis=1)
#
# # Filter out rows where the count of ones is greater than 4
# df = df[ones_count <= 4]
#
# print("after", len(df))
#
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

# Separate features and labels
X = df.drop('Label', axis=1)  # Features
y = df['Label']               # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=50, random_state=42)
rf_clf.fit(X_train, y_train)

# Get feature importances
importances = rf_clf.feature_importances_

# Convert the importances into a DataFrame
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Select a threshold for feature selection
threshold = 0.1  # Example threshold

# Select features whose importance is above the threshold
selected_features = feature_importances[feature_importances['importance'] > threshold]['feature']

print("feature imp", feature_importances['importance'])
print("selected features", selected_features)

# Filter the dataset to include only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train a new model using only the selected features
rf_clf_selected = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=50, random_state=42)
rf_clf_selected.fit(X_train_selected, y_train)

# Assuming rf_clf_selected is your trained Random Forest model
model_filename = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_14_stat.joblib'
joblib.dump(rf_clf_selected, model_filename)


# Cross-validation on the training set
cv_scores = cross_val_score(rf_clf_selected, X_train_selected, y_train, cv=5)

# Predict on both the training and testing sets
y_train_pred = rf_clf_selected.predict(X_train_selected)
y_test_pred = rf_clf_selected.predict(X_test_selected)

# Calculate and compare accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Check for overfitting
if train_accuracy > test_accuracy:
    print("Model may be overfitting.")
else:
    print("Model seems fine, no significant overfitting detected.")

# Display cross-validation scores
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.2f}")
print(f"New Testing Accuracy: {test_accuracy:.2f}")