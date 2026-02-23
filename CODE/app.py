import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Fixes NameError: joblib is not defined
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Automated Hydro-Risk System", layout="wide")
st.title("🌊 Automated Flood & Drought Prediction System")

# SIDEBAR: Only show files related to the project
all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
hydro_files = [f for f in all_files if "hydro" in f or "Val" in f]
selected_file = st.sidebar.selectbox("Select Dataset", hydro_files)

if not hydro_files:
    st.error("No valid hydrological datasets found in the directory.")
else:
    try:
        # Load Model and selected Data
        model = joblib.load('trained_hydro_model.pkl')
        df = pd.read_csv(selected_file)

        # 1. VALIDATION: Check for required columns
        required_cols = ['temp', 'evap_rate', 'rainfall', 'target']
        if all(col in df.columns for col in required_cols):
            
            # Feature Engineering for the specific file
            df['temp_evap_interaction'] = df['temp'] * df['evap_rate']
            X = df.drop('target', axis=1)
            y_true = df['target']
            
            # Generate Live Predictions
            y_pred = model.predict(X)

            # 2. DYNAMIC METRIC CALCULATION (Fixes 'Same Values' error)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # FIX: Clipped Accuracy Formula (Fixes -94.39% error)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) 
            accuracy = max(0, 100 - (mape * 100))

            # 3. DISPLAY PERFORMANCE
            st.subheader(f"📊 Live Performance Metrics: {selected_file}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{accuracy:.2f}%")
            m2.metric("R² Score", f"{r2:.4f}")
            m3.metric("RMSE", f"{rmse:.4f}")
            m4.metric("MSE", f"{mse:.4f}")
            m5.metric("MAE", f"{mae:.4f}")

            st.markdown("---")

            # 4. RISK SUMMARY & VISUALIZATIONS
            df['Risk_Probability'] = 1 / (1 + np.exp(-y_pred / 5))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Average Regional Risk", f"{df['Risk_Probability'].mean():.2%}")
            c2.metric("High-Risk Alerts", len(df[df['Risk_Probability'] > 0.7]))
            c3.metric("Records Processed", len(df))

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Risk Distribution")
                fig, ax = plt.subplots()
                sns.histplot(df['Risk_Probability'], bins=20, kde=True, color='teal', ax=ax)
                st.pyplot(fig)
            with col_r:
                st.subheader("Rainfall vs. Predicted Risk")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(data=df, x='rainfall', y='Risk_Probability', hue='temp', palette='YlOrRd', ax=ax2)
                st.pyplot(fig2)

            # 5. DATA TABLE
            st.subheader("Detailed Risk Assessment")
            st.dataframe(df.style.format({'Risk_Probability': '{:.2%}'}))
            
        else:
            # Prevents crashing on 'breast-cancer.csv'
            st.error(f"Selected file '{selected_file}' is missing required columns like 'temp' or 'target'.")

    except Exception as e:
        st.error(f"System Error: {e}")