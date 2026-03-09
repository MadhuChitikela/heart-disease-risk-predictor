from fastapi import FastAPI
import joblib
import numpy as np

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "heart_model.pkl")
model = joblib.load(model_path)

frontend_dir = os.path.join(base_dir, "frontend_web")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

from pydantic import BaseModel
from typing import List

class PatientData(BaseModel):
    data: List[float]

@app.post("/predict")
def predict(patient: PatientData):
    arr = np.array(patient.data).reshape(1, -1)
    prediction = model.predict(arr)
    return {"prediction": int(prediction[0])}
