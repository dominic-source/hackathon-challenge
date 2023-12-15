import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load dataset
file_path = 'hackathon-challenge/dataset.csv'
data = pd.read_csv(file_path)

# Load and display the dataset
print(data.head())