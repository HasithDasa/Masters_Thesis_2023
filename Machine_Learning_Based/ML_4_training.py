import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.utils import resample


# Load your dataset
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated lbp/features.csv'
df = pd.read_csv(df_path)
print("before", len(df))

# Count the number of ones in each row (excluding the label column)
ones_count = (df.drop('Label', axis=1) == 1).sum(axis=1)

# Filter out rows where the count of ones is greater than 4
df = df[ones_count <= 4]

print("after", len(df))

# Count the number of zeros in the 'Label' column
count_zeros = (df['Label'] == 0).sum()

# Count the number of ones in the 'Label' column
count_ones = (df['Label'] == 1).sum()

# Print the counts
print(f"Count of Zeros: {count_zeros}")
print(f"Count of Ones: {count_ones}")


# Separate the dataset into two based on the label
df_zeros = df[df['Label'] == 0]
df_ones = df[df['Label'] == 1]

# Undersample the majority class
df_zeros_undersampled = resample(df_zeros,
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_ones), # match number in minority class
                                 random_state=42)  # reproducible results

# Combine the minority class with the downsampled majority class
df = pd.concat([df_zeros_undersampled, df_ones])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = df.drop('Label', axis=1)  # Features
y = df['Label']               # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize the Random Forest Classifier with more trees and limited depth
# rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# # Initialize the Random Forest Classifier
# rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)


# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 6]
}

# Initialize the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=3, scoring=make_scorer(accuracy_score), n_jobs=-1)

# Perform the Grid Search
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the best model
best_rf_clf = RandomForestClassifier(**best_params, random_state=42)
best_rf_clf.fit(X_train, y_train)

# Cross-validation on the training set
cv_scores = cross_val_score(best_rf_clf, X_train, y_train, cv=5)

# Predict on both the training and testing sets
y_train_pred = best_rf_clf.predict(X_train)
y_test_pred = best_rf_clf.predict(X_test)

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