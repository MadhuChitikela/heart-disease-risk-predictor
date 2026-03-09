import joblib
from sklearn.metrics import accuracy_score
from feature_engineering import prepare_features

X_train, X_test, y_train, y_test = prepare_features("data/processed/cleaned_heart.csv")

model = joblib.load("models/heart_model.pkl")

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
