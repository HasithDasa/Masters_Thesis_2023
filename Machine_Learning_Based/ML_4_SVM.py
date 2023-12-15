import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib

# Load your dataset
df_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/features_5_stat_normalized.csv'
df = pd.read_csv(df_path).head(75100)



# # Print the counts
# print(f"Count of Zeros: {count_zeros}")
# print(f"Count of Ones: {count_ones}")

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



# Define feature combinations
# feature_combinations = [('Feature_3', 'Feature_2'), ('Feature_3', 'Feature_1'), ('Feature_3', 'Feature_4')]
feature_combinations = [('Feature_3', 'Feature_4')]

for features in feature_combinations:
    # Separate features and labels for the specific combination
    X = df[list(features)]  # Select only the columns for the feature combination
    y = df['Label']         # Labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the SVM classifier
    svm_clf = SVC(random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {'C': [1], 'gamma': [100], 'kernel': ['rbf']} #60% accuracy
    # param_grid = {'C': [0.001, 0.01, 0.1, 1], 'gamma': [100, 10, 1, 0.1], 'kernel': ['linear', 'sigmoid']}

    # Perform grid search
    grid_search = GridSearchCV(svm_clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Retrieve the best estimator
    best_svm_clf = grid_search.best_estimator_

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(best_svm_clf, X_train, y_train, cv=5)

    # Predict on both the training and testing sets
    y_train_pred = best_svm_clf.predict(X_train)
    y_test_pred = best_svm_clf.predict(X_test)

    # Calculate and compare accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Training Accuracy with features {features}: {train_accuracy:.2f}")
    print(f"Testing Accuracy with features {features}: {test_accuracy:.2f}")

    # Check for overfitting
    if train_accuracy > test_accuracy:
        print("Model may be overfitting.")
    else:
        print("Model seems fine, no significant overfitting detected.")

    # Display cross-validation scores
    print(f"Cross-Validation Scores with features {features}: {cv_scores}")
    print(f"Average CV Score with features {features}: {cv_scores.mean():.2f}")

    # Save the best SVM model for each feature combination
    model_filename = f'svm_classifier_{features[0]}_{features[1]}.joblib'
    joblib.dump(best_svm_clf, model_filename)
    print(f"Best SVM model saved as {model_filename}")
    print(f"Best parameters found with features {features}: {grid_search.best_params_}")
