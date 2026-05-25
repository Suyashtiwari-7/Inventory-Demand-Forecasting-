# Inventory Demand Forecasting & Price Optimizer 📦📈

A production-grade Machine Learning system that forecasts weekly product demand and optimizes promotional pricing to maximize retail revenue. Built using **Python**, **LightGBM**, **Streamlit**, and a compiled **Go REST API**.

---

## 📊 Project Overview
This project solves a classic retail supply chain challenge: predicting product demand to optimize inventory levels and promotional pricing. By accurately forecasting weekly demand (`units_sold`) for any combination of store and product SKU, the system helps minimize stockouts and reduce inventory holding costs.

The architecture uses a **hybrid design**:
1. **Python** handles model training, target encoding, and hyperparameter tuning.
2. **Streamlit** provides an interactive web dashboard for business exploration.
3. **Go (Golang)** loads the trained LightGBM model and serves predictions through a high-performance REST API.

---

## 🗂️ Dataset Specifications
The model is trained on **150,150 weekly transaction records** consisting of:
* **76 Active Stores** (`store_id`)
* **28 Unique Products** (`sku_id`)
* **Price & Promotion Features:**
  * `base_price`: The standard retail price.
  * `total_price`: The actual selling price (including discounts).
  * `is_featured_sku`: Binary flag indicating if the product was featured in the catalog.
  * `is_display_sku`: Binary flag indicating if the product had in-store stand display.
* **Target Variable:** `units_sold` (weekly demand volume).

---

## ⚙️ Technical Implementation & Pipelines

### 1. Data Preprocessing & Custom Imputation
* Imputed a single missing value in the `total_price` column by copying the corresponding `base_price` for that record (representing no discount), rather than using a continuous statistical mean.
* Filtered out extreme outliers (above the 99th percentile of `units_sold`) to prevent the model from fitting to supply-chain anomalies.

### 2. Feature Engineering & Lags
* **Promotional Features:**
  * `price_difference`: The absolute discount amount (`base_price - total_price`).
  * `discount_ratio`: The percentage discount offered (`price_difference / base_price`).
  * `is_discounted`: Boolean flag indicating if a discount was active (`total_price < base_price`).
* **Temporal Features:** Extracted `day`, `month`, `year`, `week_of_year`, and `day_of_year` from the `week` date.
* **Exogenous Lag Features:** Shifted price and promotional variables (`total_price_lag_1`, `total_price_lag_2`, `is_featured_sku_lag_1`, `is_display_sku_lag_1`, and `discount_ratio_lag_1`) per store-SKU combination.

### 3. Native Categorical Variable Support
* Instead of sparse one-hot encoding which increases dataset dimensionality, `store_id` and `sku_id` are cast to Pandas `category` type. This allows LightGBM to perform optimal multi-way categorical splits natively.
* Mappings are stored to align the categories of any incoming validation or single-row simulation data at inference time.

### 4. Out-of-Fold (OOF) Target Encoding
* Computed the average sales of `units_sold` per `store_sku`, `store`, and `sku` combinations.
* Implemented a **5-Split K-Fold Out-of-Fold (OOF) encoder** during training to prevent target leakage, and exported the global mappings for test set inference.

### 5. Hyperparameter Tuning
* Optimized using a cross-validated Randomized Search to select the best parameters:
  * `n_estimators`: `800`
  * `learning_rate`: `0.1`
  * `num_leaves`: `128`
  * `subsample`: `0.8`
  * `colsample_bytree`: `0.9`

---

## 📈 Model Performance Metrics

| Model Configuration | R² Score | RMSE | Error Reduction |
| :--- | :---: | :---: | :---: |
| **Baseline Model** (Original notebook code) | `84.16%` | `16.99` | Baseline |
| **Optimized Model** (Native categories + Feature Engineering) | `87.79%` | `14.92` | -12.2% |
| **Optimized & Tuned Model** (OOF CV Target Encoding + Lags) | **`89.46%`** | **`13.85`** | **-18.5%** |

---

## ⚡ High-Performance Go REST API (`go-api/`)
To serve predictions in a high-concurrency production environment, a microservice is written in Go. It uses a pure Go implementation of LightGBM (`leaves`), avoiding slow C bindings (CGO). It loads the model text file (`models/lgbm_model.txt`) and target encoding JSON (`models/mappings.json`) directly into memory on startup for sub-millisecond inference speeds.

### 1. Compile and Run the Go API
*Prerequisite: Go must be installed on the system.*
```bash
cd go-api
go mod tidy
go run main.go
```
The server will start listening on port **`8080`**.

### 2. Example API Call
Send a POST request to `/predict` with the JSON feature payload:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 8023,
    "sku_id": 216233,
    "total_price": 99.0,
    "base_price": 110.0,
    "is_featured_sku": 0,
    "is_display_sku": 0,
    "day": 15,
    "month": 6,
    "year": 2013,
    "week_of_year": 24,
    "day_of_year": 166,
    "total_price_lag_1": 99.0,
    "total_price_lag_2": 99.0,
    "is_featured_sku_lag_1": 0,
    "is_display_sku_lag_1": 0,
    "discount_ratio_lag_1": 0.1
  }'
```
Response:
```json
{
  "predicted_demand": 127.75134279911771,
  "status": "success"
}
```

---

## 🖥️ Streamlit Web Application (`app.py`)
A custom dark-themed interactive dashboard designed for decision support:
1. **Analytics Dashboard:** Visualizes top-selling products, top stores, and price-to-demand scatter plots.
2. **Forecast Explorer:** Interactive Plotly charts displaying historical sales vs. predictions on the test set for any Store-SKU combination.
3. **Price & Revenue Optimizer:** A live simulator with a price slider to estimate sales volume, with a revenue curve showing the **Optimal Pricing Point** (maximizing Price × Predicted Demand).
4. **Model Retrainer:** Trigger retraining directly from the UI with real-time progress indicators.

---

## 🚀 Getting Started (Python Web Dashboard)

### 1. Installation
Install Python dependencies:
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Run the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
* **Credentials:** Log in using username `admin` and password `password`, or click **Guest Access** to log in instantly.

---

## 🗂️ Modular Code Structure
* **`src/features.py`**: Preprocessing, lag features, and target encoding functions.
* **`src/model.py`**: Model training, saving, loading, and prediction functions.
* **`app.py`**: Streamlit dashboard web interface.
* **`go-api/`**: Production REST API microservice.
