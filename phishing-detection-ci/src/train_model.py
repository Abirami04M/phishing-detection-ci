# src/train_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocess import preprocess_data

# Load and preprocess training & test data
X_train, y_train = preprocess_data('phishing-detection-ci/data/training.csv')
X_test, y_test = preprocess_data('phishing-detection-ci/data/testing.csv')

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save base model
joblib.dump(model, 'phishing-detection-ci/models/base_model.pkl')
print("✅ Model saved as base_model.pkl")
