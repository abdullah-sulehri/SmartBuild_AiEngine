# SmartBuild AI Engine (FastAPI Backend)

SmartBuild is a B2B construction material procurement platform. This repository contains the **AI Recommendation Engine & Backend**, built with FastAPI. It serves as the intelligent "brain" behind the Next.js frontend, utilizing multiple machine learning models to optimize logistics, predict market trends, and score supplier reliability.

## 🧠 Core AI Architecture

To ensure enterprise-grade reliability and 100% billing accuracy, this engine employs a **hybrid architecture** that separates probabilistic machine learning from deterministic financial mathematics.

1.  **Landed Cost Predictor (XGBoost)**
    * Rather than predicting the total transaction value (which leads to ML interpolation errors on massive quantities), the XGBoost regression tree is strictly tasked with predicting the highly variable **logistics/freight overhead** based on distance, truck type, and volume constraints. 
    * The API then merges this AI prediction with deterministic Python math (`Price * Quantity`) to generate an airtight, mathematically perfect Landed Cost invoice.
2.  **Supplier Trust Classifier (Random Forest)**
    * A classification model that balances supplier ratings against historical dispute counts, utilizing `class_weight='balanced'` to handle industry-standard imbalanced datasets. It outputs a probabilistic trust score (0.0 to 1.0).
3.  **Market Trend Forecaster (Prophet / SARIMAX)**
    * A time-series forecasting pipeline trained individually on 24 distinct construction materials. It factors in baseline inflation and seasonal anomalies (e.g., the July-September monsoon construction dip) to predict next week's baseline market price.

---

## 📂 Repository Structure

* `app.py` - The FastAPI server and API endpoint definitions.
* `config.py` - Single source of truth for constants, inflation rates, and system parameters.
* `generate_datasets.py` - Pipeline **Phase 1**: Generates synthetically anchored historical data (Orders, Prices, Suppliers).
* `pipeline_preprocessing.py` - Pipeline **Phase 2**: Engineers features, applies encoders/scalers, and persists `joblib` transformers to prevent train/serve skew.
* `train_models.py` - Pipeline **Phase 3**: Trains the XGBoost, Random Forest, and SARIMAX/Prophet models and saves the `.joblib` files.
* `/models/` - Ignored in `.gitignore` initially, this folder holds the compiled `.joblib` files required for the API to run.
* `/artifacts/` - Auto-generated evaluation matrices, SHAP values, and feature importance CSVs for FYP defense transparency.

---

## 🚀 Local Installation & Setup

**1. Create a Virtual Environment**
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

**2. Install Dependencies**
    pip install fastapi uvicorn pandas numpy scikit-learn xgboost statsmodels prophet shap joblib

*(Note: If Prophet fails to install due to C++ compiler issues, the pipeline automatically falls back to SARIMAX without crashing.)*

---

## ⚙️ Running the ML Pipeline

If you are cloning this repository for the first time, you must build the models before starting the server. Run these scripts in exact order:

    # 1. Generate the historical supply chain data
    python generate_datasets.py

    # 2. Preprocess, scale, and encode the data
    python pipeline_preprocessing.py

    # 3. Train the models and generate evaluation artifacts
    python train_models.py

Once Phase 3 completes, your `/models` folder will be populated and the API is ready to launch.

---

## 🌐 Starting the API Server

Start the FastAPI server using Uvicorn:

    uvicorn app:app --reload --port 8000

* **API Base URL:** `http://127.0.0.1:8000`
* **Swagger UI (Interactive Docs):** `http://127.0.0.1:8000/docs`

---

## 📡 API Endpoints

### `GET /api/v1/forecast-trend/{material}`
Returns the predicted market baseline price for the requested material for the upcoming week.
* **Example Input:** `/api/v1/forecast-trend/Cement`

### `POST /api/v1/score-supplier`
Evaluates a supplier's rating and dispute count to return an AI trust probability score.

### `POST /api/v1/predict-cost`
Calculates the exact subtotal using deterministic math, predicts the physical freight/logistics cost using XGBoost, and returns a mathematically perfect Total Landed Cost.

### `POST /api/v1/smart-recommendation`
The core ranking engine. Accepts an array of competing supplier quotes and ranks them using a composite algorithm:
* **Cost Score:** Rewards the most affordable landed cost.
* **Trust Score:** Penalizes suppliers with high dispute histories.
* **Trend Score:** A "Too Good To Be True" mechanism that severely penalizes scam quotes attempting to undercut the AI's market benchmark by > 20%.

---
.
