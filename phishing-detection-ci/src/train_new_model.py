# src/train_new_model.py
from sklearn.ensemble import RandomForestClassifier
from data_preprocess import preprocess_data
import joblib

X_train, y_train = preprocess_data('phishing-detection-ci/data/training.csv')

# Smaller model for testing rejection
model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'phishing-detection-cicid/models/new_model.pkl')
print("New model saved as new_model.pkl")
