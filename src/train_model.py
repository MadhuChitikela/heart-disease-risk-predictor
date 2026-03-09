import joblib
from xgboost import XGBClassifier
from feature_engineering import prepare_features

X_train, X_test, y_train, y_test = prepare_features("data/processed/cleaned_heart.csv")

# Implementing XGBoost with Early Stopping and GPU support
model = XGBClassifier(
    n_estimators=1000,          # Large number of trees, early stopping will halt it
    learning_rate=0.05,
    tree_method="hist",         # Newest approach for GPU (historically gpu_hist)
    device="cuda",              # Set to "cuda" to utilize your GPU
    early_stopping_rounds=50    # Stops if no improvement in 50 consecutive rounds
)

# Fit model with evaluation set for early stopping to monitor performance
model.fit(
    X_train, 
    y_train, 
    eval_set=[(X_test, y_test)],
    verbose=10
)

joblib.dump(model, "models/heart_model.pkl")

print("Model trained and saved successfully with Early Stopping!")
