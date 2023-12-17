"""
    This module was designed to find the best hyperparameters for cross validation using 
    RandomizedSearchCV and the given dataset which is preprocessed
"""

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint


# Load dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Remove the unnamed: 133 column
data_cleaned = data.drop(columns=['Unnamed: 133'])

# Drop all duplicated data in dataset
data_cleaned = data_cleaned.drop_duplicates()

# Separate features and target variables

# The features
X = data_cleaned.drop('prognosis', axis=1)

# The Target variable
y = data_cleaned['prognosis']



# Using randomized search cross validation to search the best combination of hyperparameters

# Define the parameter distribution to sample from
param_dist = {
        'n_estimators': randint(100, 500),  
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 20)
}
rf_classifier_find = RandomForestClassifier()

random_search = RandomizedSearchCV(rf_classifier_find, param_dist, cv=5, n_iter=10)

# Fit the data to find the best hyperparameters
random_search.fit(X, y)

# Get the best hyperparameters
best_params = random_search.best_params_
