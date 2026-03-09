# Heart Disease Prediction System

🌟 **Live AI Dashboard:** [https://heart-disease-risk-predictor-mazm.vercel.app/](https://heart-disease-risk-predictor-mazm.vercel.app/)

A professional machine learning system to predict heart disease risk using clinical patient data. 
This project follows an industry-standard ML structure suitable for a strong resume and GitHub portfolio.

## Key Features

* Developed a **machine learning system to predict heart disease risk** using clinical patient data.
* Implemented **data preprocessing, feature engineering, and model training pipelines**.
* Built a **FastAPI REST API for real-time predictions**.
* Structured using **modular ML architecture and best practices**.

## Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your raw dataset inside `data/raw/heart.csv`.

## Running the Pipeline

1. **Preprocess the Data**:
   ```bash
   python src/data_preprocessing.py
   ```
2. **Train the Model**:
   ```bash
   python src/train_model.py
   ```
3. **Evaluate the Model**:
   ```bash
   python src/evaluate_model.py
   ```
4. **Run a Sample Prediction**:
   ```bash
   python src/predict.py
   ```

## API Deployment

Run the FastAPI application:
```bash
uvicorn api.app:app --reload
```
Navigate to `http://localhost:8000/docs` to test the API.

## Advanced (GPU Support)
To use GPU acceleration, make sure you have CUDA installed and set up XGBoost GPU mode in `train_model.py`:
```python
from xgboost import XGBClassifier
model = XGBClassifier(tree_method="gpu_hist")
```
