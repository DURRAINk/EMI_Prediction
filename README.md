# 📊 EMI Prediction

a comprehensive financial risk assessment platform that integrates machine learning models with MLflow experiment tracking to create an interactive web application for EMI prediction.
Nowadays, people struggle to pay EMI due to poor financial planning and inadequate risk assessment. This project aims to solve this critical issue by providing data-driven insights for better loan decisions.

## Visit to the streamlit application 
https://emi-prediction-app.streamlit.app/
---

## 🚀 Features
- **Data preprocessing**: Handling missing values, scaling, and encoding  
- **Financial feature engineering**: Debt‑to‑income, expense‑to‑income, EMI ratios  
- **Models**: XGBoost, Random Forest, and regression/classification pipelines  
- **Deployment**: Streamlit app for interactive EMI prediction  
- **Artifacts**: Pre‑trained models (`emi_classifier_model.json`, `emi_regressor_model.json`), scaler (`scaler.pkl`), and dataset (`emi_prediction_dataset.csv`)  

---

## 📂 Repository Structure
EMI_Prediction/
│── app.py                  # Streamlit app for EMI prediction
│── doc.ipynb               # Jupyter notebook with experiments
│── emi_prediction_dataset.csv  # Dataset used for training
│── emi_classifier_model.json   # Saved classifier model
│── emi_regressor_model.json    # Saved regressor model
│── scaler.pkl              # Pre-fitted MinMaxScaler
│── requirements.txt        # Project dependencies

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/DURRAINk/EMI_Prediction.git
cd EMI_Prediction
pip install -r requirements.txt

### 4. Start MLflow Tracking UI (Optional)
In `eda.ipynb`, there is the code for MLflow
1. launch the UI:
```bash
   mlflow ui
```
2. copy and paste your localhost uri in the notebook:
```python
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
```
3. Run the code
