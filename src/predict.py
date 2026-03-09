import joblib
import numpy as np

model = joblib.load("models/heart_model.pkl")

sample = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])

prediction = model.predict(sample)

print("Heart Disease Risk:", prediction)
