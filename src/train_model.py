from sklearn.ensemble import RandomForestClassifier
import joblib
from feature_engineering import prepare_features

X_train, X_test, y_train, y_test = prepare_features("data/processed/cleaned_heart.csv")

# Implementing RandomForest for Vercel/Serverless deployment
# RandomForest + Scikit-Learn is far lighter than XGBoost
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

model.fit(X_train, y_train)

joblib.dump(model, "models/heart_model.pkl")

print("RandomForest model trained and saved successfully!")
