import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib

# Directory containing the CSV files
data_dir = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 3/statistics/systematic_error/patch_10_240_360'
model_save_dir = data_dir
excel_file_path = 'D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 3/statistics/systematic_error/patch_10_240_360/results.xlsx'

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=['CSV File', 'Train Accuracy', 'Test Accuracy', 'Best Parameters'])

# Define feature combinations
feature_combinations = [('Feature_1', 'Feature_2')]

# Iterate through all CSV files in the directory
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df_path = os.path.join(data_dir, file)
        df = pd.read_csv(df_path)

        # Data preprocessing steps
        count_zeros = (df['Label'] == 0).sum()
        count_ones = (df['Label'] == 1).sum()

        df_zeros = df[df['Label'] == 0]
        df_ones = df[df['Label'] == 1]

        if count_ones > count_zeros:
            df_ones_undersampled = resample(df_ones, replace=False, n_samples=len(df_zeros), random_state=42)
            df = pd.concat([df_ones_undersampled, df_zeros])
        elif count_ones < count_zeros:
            df_zeros_undersampled = resample(df_zeros, replace=False, n_samples=len(df_ones), random_state=42)
            df = pd.concat([df_zeros_undersampled, df_ones])

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Process each feature combination
        for features in feature_combinations:
            X = df[list(features)]  # Select only the columns for the feature combination
            y = df['Label']  # Labels

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Define the SVM classifier
            svm_clf = SVC(random_state=42)
            param_grid = {'C': [0.001, 0.01, 0.1, 1], 'gamma': [100, 10, 1, 0.1], 'kernel': ['linear', 'rbf']}

            # Perform grid search
            grid_search = GridSearchCV(svm_clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Retrieve the best estimator and best parameters
            best_svm_clf = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Predict on both the training and testing sets
            y_train_pred = best_svm_clf.predict(X_train)
            y_test_pred = best_svm_clf.predict(X_test)

            # Calculate and compare accuracies
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Save the best SVM model for each feature combination
            model_filename = f'{file[:-4]}_svm.joblib'
            path_and_name = os.path.join(model_save_dir, model_filename)
            joblib.dump(best_svm_clf, path_and_name)

            # Append results to the DataFrame
            results_df = results_df.append({
                'CSV File': file,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Best Parameters': str(best_params)
            }, ignore_index=True)

# Save the results to an Excel file
results_df.to_excel(excel_file_path, index=False)
