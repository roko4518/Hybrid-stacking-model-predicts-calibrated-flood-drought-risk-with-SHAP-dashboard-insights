# Hybrid-stacking-model-predicts-calibrated-flood-drought-risk-with-SHAP-dashboard-insights
This system uses a hybrid stacking ensemble combining Random Forest, XGBoost, and Ridge Regression to deliver highly accurate flood and drought risk predictions with calibrated 0–100% probabilities. An interactive Streamlit dashboard with SHAP interpretability explains each prediction, ensuring transparency and informed decision-making.
🌊 Automated Flood & Drought Prediction System
An AI-powered Hydrological Risk Prediction Dashboard built using Streamlit, Machine Learning, and Stacking Ensemble Models to predict flood and drought risk probabilities from environmental datasets.

📌 Project Overview
This project predicts regional hydro-risk levels (Flood/Drought risk) using:

🌡 Temperature
💧 Evaporation Rate
🌧 Rainfall
🔁 Engineered interaction features
The system:

Trains a Stacking Regressor Model
Saves the trained model as .pkl
Loads the model into a Streamlit Web Dashboard
Generates live performance metrics
Displays risk probability visualizations
Provides detailed risk assessment tables
🧠 Machine Learning Architecture
The model uses a Stacking Regressor with:

Base Models:
Random Forest Regressor
XGBoost Regressor
Final Estimator:
RidgeCV
Feature Engineering:
temp_evap_interaction = temp × evap_rate
This interaction improves prediction accuracy by modeling combined environmental effects.

📂 Project Structure
├── app.py                    # Streamlit Dashboard Application
├── CPP FINAL CODE.py         # Model Training Script
├── trained_hydro_model.pkl   # Saved ML Model (Generated after training)
├── Val_hydro_data.csv        # Training Dataset
├── hydro_*.csv               # Validation/Test datasets
└── README.md
⚙️ Installation
1️⃣ Clone the Repository
git clone <your-repo-link>
cd <project-folder>
2️⃣ Install Dependencies
pip install -r requirements.txt
Or manually install:

pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn joblib
🚀 How to Run the Project
Step 1: Train the Model
Run:

python "CPP FINAL CODE.py"
This will:

Train the stacking model
Generate trained_hydro_model.pkl
Step 2: Launch the Dashboard
streamlit run app.py
The web app will open in your browser.

📊 Dashboard Features
✅ Live Performance Metrics
Accuracy (MAPE-based)
R² Score
RMSE
MSE
MAE
📈 Visualizations
Risk Probability Distribution
Rainfall vs Predicted Risk Scatter Plot
🚨 Risk Monitoring
Average Regional Risk
High Risk Alerts (> 70%)
Total Records Processed
📋 Detailed Risk Table
Color-coded:

🟢 Low Risk (< 40%)
🟠 Medium Risk (40–70%)
🔴 High Risk (> 70%)
📌 Required Dataset Format
Your CSV files must contain:

Column Name	Description
temp	Temperature
evap_rate	Evaporation Rate
rainfall	Rainfall Amount
target	Risk Score (Training Only)
⚠ If required columns are missing, the app will show an error.

🔬 Risk Probability Formula
Predicted risk score is converted to probability using: Risk= 1/1+e^−score/5 1​image

This scales output between 0–100%.

📦 Performance Metrics Used
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R² Score
MAPE-based Accuracy
🛠 Technologies Used
Python 3.12
Streamlit
Scikit-learn
XGBoost
Pandas
NumPy
Matplotlib
Seaborn
Joblib
🎯 Key Highlights
✔ Ensemble Learning (Stacking) ✔ Feature Engineering ✔ Live Metric Evaluation ✔ Automated Dataset Detection ✔ Interactive Risk Dashboard ✔ Scalable for large datasets (2000+ records)

🧩 Future Improvements
Add real-time weather API integration
Add classification-based flood/drought alerts
Deploy to Streamlit Cloud / AWS
Add geospatial risk heatmaps
Improve model explainability using SHAP
