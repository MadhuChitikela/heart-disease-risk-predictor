# ❤️ Heart Disease Risk Prediction System

🌟 **Live AI Dashboard:** [https://heart-disease-risk-predictor-mazm.vercel.app/](https://heart-disease-risk-predictor-mazm.vercel.app/)

A professional, end-to-end machine learning system designed to predict heart disease risk utilizing clinical patient data. This project demonstrates an industry-standard ML architecture spanning from iterative data preprocessing to a fully deployed serverless web application.

---

## 🏗️ Project Architecture & Workflow

The pipeline is split into three functional layers: 
1. **Data Layer (`src/`)**: Handles automated ingestion, missing value imputation, and feature matrix construction.
2. **Modeling Layer (`models/`)**: Generates optimized algorithmic inferences utilizing Random Forest classification to handle tabular relationships efficiently.
3. **Serving Layer (`api/` & `frontend_web/`)**: Exposes the trained model via a FastAPI REST interface and consumes it via a beautiful, Vanilla JS glassmorphism frontend.

### Workflow
1. Raw Data is parsed from `data/raw/` ➡️ `data_preprocessing.py`.
2. Processed arrays move to `feature_engineering.py` for target splitting.
3. `train_model.py` develops the ML model and serializes it securely into `models/`.
4. `api/app.py` boots up a Uvicorn server, deserializes the core model, mounts the static frontend UI, and handles asynchronous inference requests on the `/predict` endpoint.

---

## 🛠️ Technology Stack

* **Data Processing:** `pandas`, `numpy`
* **Machine Learning Engine:** Scikit-Learn (`RandomForestClassifier`)
* **Backend API framework:** `fastapi`, `uvicorn`, `pydantic`
* **Model Serialization:** `joblib`
* **Frontend Web Application:** HTML5, CSS3 Variables (Modern Glassmorphism UI), Vanilla JavaScript, Fetch API
* **Deployment/Orchestration:** Vercel Serverless Lambdas, Docker (`docker-compose`)

---

## 📂 Project Structure

```text
heart-disease-prediction/
│
├── api/                    # FastAPI Server configuration
│   └── app.py              # Main REST API and static mounting
│
├── data/
│   ├── raw/                # Unaltered source data
│   └── processed/          # Cleaned downstream features
│
├── frontend_web/           # Production Modern Web App UI
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/                 # Serialized Machine Learning artifacts
│   └── heart_model.pkl
│
├── src/                    # Data Science Pipeline Code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── Dockerfile.api          # Containerization scripts
├── Dockerfile.frontend
├── docker-compose.yml 
├── requirements.txt
└── main.py                 # Pipeline Orchestrator
```

---

## 🚀 Setup & Installation (Local Development)

1. **Clone the repository and set up a Virtual Environment**:
   ```bash
   git clone https://github.com/MadhuChitikela/heart-disease-risk-predictor.git
   cd heart-disease-risk-predictor
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete Data & Training Pipeline**:
   ```bash
   python main.py
   ```

4. **Boot up the Live Web Application**:
   ```bash
   uvicorn api.app:app --reload
   ```
   *Navigate to `http://localhost:8000/` to use the User Interface!*
   *Navigate to `http://localhost:8000/docs` to test the API securely via Swagger UI.*
