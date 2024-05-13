import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
data = pd.read_csv('phishing_dataset.csv')

# Assuming your dataset has features extracted from URLs and a target column 'is_phishing'
X = data.drop('is_phishing', axis=1)  # Features
y = data['is_phishing']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'phishing_model.pkl')
