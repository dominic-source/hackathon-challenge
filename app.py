import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
file_path = 'hackathon-challenge/dataset.csv'
data = pd.read_csv(file_path)

# Remove the unnamed: 133 column
data_cleaned = data.drop(columns=['Unnamed: 133'])

data_cleaned = data_cleaned.drop_duplicates()

# Separate features and target variables

# The features
X = data_cleaned.drop('prognosis', axis=1)

# The Target variable
y = data_cleaned['prognosis']


def model_train_test(model, X, y):

    """
    Train and test a machine learning model
    
    Parameters:
    model: The machine learning model to be trained and tested.
    X_train: Training data features
    y_train: Training data labels
    X_test: Testing data features
    y_test: Testing data labels
    """
    
    #  Scores the store all a list of scores for each cross validation test   
    test = []
    prediction = []

    # The KFold cross validation to randomly split dataset into parts for training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Model training
        model.fit(X_train, y_train)
    
        # Predictions on the test set
        pred = model.predict(X_test)
    
        test.append(y_test)
        prediction.append(pred)

    # Convert the test sample and predictions to a flat list instead of a list of list
    flat_test = [item for sublist in test for item in sublist]
    flat_prediction = [item for sublist in prediction for item in sublist]
    
    # Accuracy
    accuracy = accuracy_score(flat_test, flat_prediction)

    print("Accuracy score: ", accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(flat_test, flat_prediction, zero_division=1))

    return model


# Instantiate the Random Forest Classifier
rf_classifier = RandomForestClassifier(max_depth=20, min_samples_split=4, n_estimators=472, random_state=42)

# model_train_test(rf_classifier, X_train, y_train, X_val, y_val)
model_train_test(rf_classifier, X.values, y.values)
