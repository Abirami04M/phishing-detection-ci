# src/model_performance.py
import joblib
from sklearn.metrics import accuracy_score, f1_score
from data_preprocess import preprocess_data

def load_model(path):
    return joblib.load(path)

def evaluate(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, f1

def compare_models():
    base_model = load_model('phishing-detection-ci/models/base_model.pkl')
    new_model = load_model('phishing-detection-ci/models/new_model.pkl')

    X_test, y_test = preprocess_data('phishing-detection-ci/data/testing.csv')

    base_acc, base_f1 = evaluate(base_model, X_test, y_test)
    new_acc, new_f1 = evaluate(new_model, X_test, y_test)

    print(f"Base Model - Acc: {base_acc}, F1: {base_f1}")
    print(f"New Model  - Acc: {new_acc}, F1: {new_f1}")

    if new_acc >= base_acc:
        print("✅ New model is accepted for deployment.")
        return True
    else:
        print("❌ New model rejected. Rolling back to base.")
        return False

if __name__ == "__main__":
    success = compare_models()
    exit(0 if success else 1)
