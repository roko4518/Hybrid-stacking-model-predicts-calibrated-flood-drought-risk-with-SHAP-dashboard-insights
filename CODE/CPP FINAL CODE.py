{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "89c82a6c-b45c-47ec-8cc1-792e68f85c60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Visualization libraries installed! Please restart your kernel if needed.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "'C:\\Users\\POLEMONI' is not recognized as an internal or external command,\n",
      "operable program or batch file.\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "# Install necessary visualization libraries to the current kernel\n",
    "!{sys.executable} -m pip install matplotlib seaborn --quiet\n",
    "\n",
    "print(\"✅ Visualization libraries installed! Please restart your kernel if needed.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6f452014-db30-4e38-8bc2-bb2f5a710797",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-02-23 13:17:11.601 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:11.602 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.352 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\POLEMONI ABHIRAM\\AppData\\Roaming\\Python\\Python312\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-02-23 13:17:12.353 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.353 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.356 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.358 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.359 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.360 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.361 Session state does not function when running a script without `streamlit run`\n",
      "2026-02-23 13:17:12.361 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.364 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:12.365 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:14.435 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:14.436 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-23 13:17:14.437 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "\n",
    "st.set_page_config(page_title=\"Automated Hydro-Risk System\", layout=\"wide\")\n",
    "\n",
    "st.title(\"🌊 Automated Flood & Drought Prediction System\")\n",
    "\n",
    "# SIDEBAR: Dataset Selector\n",
    "st.sidebar.header(\"Configuration\")\n",
    "# Automatically finds all CSV files in your folder\n",
    "csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]\n",
    "selected_file = st.sidebar.selectbox(\"Select Dataset\", csv_files if csv_files else [\"No CSV found\"])\n",
    "\n",
    "# 1. LOAD MODEL AND DATA\n",
    "try:\n",
    "    model = joblib.load('trained_hydro_model.pkl')\n",
    "    # Now loads the file you picked in the sidebar (e.g., hydro_data_2000.csv)\n",
    "    df = pd.read_csv(selected_file)\n",
    "    \n",
    "    # Pre-processing: Apply feature engineering to the dataset\n",
    "    df_features = df.copy()\n",
    "    df_features['temp_evap_interaction'] = df_features['temp'] * df_features['evap_rate']\n",
    "    \n",
    "    # Ensure we only drop 'target' if it exists in the file\n",
    "    X = df_features.drop('target', axis=1) if 'target' in df_features.columns else df_features\n",
    "    \n",
    "    # 2. GENERATE PREDICTIONS AUTOMATICALLY\n",
    "    df['Predicted_Risk_Score'] = model.predict(X)\n",
    "    \n",
    "    # Scale to 0-100% Probability\n",
    "    df['Risk_Probability'] = 1 / (1 + np.exp(-df['Predicted_Risk_Score'] / 5))\n",
    "    \n",
    "    # 3. SUMMARY METRICS\n",
    "    avg_risk = df['Risk_Probability'].mean()\n",
    "    high_risk_count = len(df[df['Risk_Probability'] > 0.7])\n",
    "    \n",
    "    c1, c2, c3 = st.columns(3)\n",
    "    c1.metric(\"Average Regional Risk\", f\"{avg_risk:.2%}\")\n",
    "    c2.metric(\"High-Risk Alerts\", high_risk_count)\n",
    "    c3.metric(\"Data Records Processed\", len(df))\n",
    "\n",
    "    st.markdown(\"---\")\n",
    "\n",
    "    # 4. VISUALIZATIONS\n",
    "    col_left, col_right = st.columns(2)\n",
    "    \n",
    "    with col_left:\n",
    "        st.subheader(\"Risk Distribution Across Dataset\")\n",
    "        fig, ax = plt.subplots()\n",
    "        sns.histplot(df['Risk_Probability'], bins=20, kde=True, color='teal', ax=ax)\n",
    "        ax.set_xlabel(\"Probability\")\n",
    "        st.pyplot(fig)\n",
    "\n",
    "    with col_right:\n",
    "        st.subheader(\"Rainfall vs. Predicted Risk\")\n",
    "        fig2, ax2 = plt.subplots()\n",
    "        sns.scatterplot(data=df, x='rainfall', y='Risk_Probability', hue='temp', palette='YlOrRd', ax=ax2)\n",
    "        st.pyplot(fig2)\n",
    "\n",
    "    # 5. AUTOMATED RISK TABLE\n",
    "    st.subheader(f\"Detailed Risk Assessment: {selected_file}\")\n",
    "    def color_risk(val):\n",
    "        color = 'red' if val > 0.7 else 'orange' if val > 0.4 else 'green'\n",
    "        return f'color: {color}'\n",
    "\n",
    "    # Using st.dataframe instead of st.table for larger datasets like 2000 rows\n",
    "    st.dataframe(df.style.applymap(color_risk, subset=['Risk_Probability']).format({'Risk_Probability': '{:.2%}'}))\n",
    "\n",
    "except Exception as e:\n",
    "    st.error(f\"Error loading system: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff074ec1-474c-46ef-a7d8-d59a413f4b75",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "72e662c6-cd80-4461-a6ce-8b125729a5cc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model trained and 'trained_hydro_model.pkl' has been saved.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
    "from sklearn.ensemble import RandomForestRegressor, StackingRegressor\n",
    "from sklearn.linear_model import RidgeCV\n",
    "from xgboost import XGBRegressor\n",
    "\n",
    "# 1. Load the primary training dataset\n",
    "# Ensure this file is in your project folder\n",
    "df = pd.read_csv('Val_hydro_data.csv')\n",
    "\n",
    "# Feature Engineering\n",
    "df['temp_evap_interaction'] = df['temp'] * df['evap_rate']\n",
    "\n",
    "X = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "\n",
    "# 2. Define Model Architecture (Fixes NameError: stacking_model)\n",
    "base_models = [\n",
    "    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),\n",
    "    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05))\n",
    "]\n",
    "stacking_model = StackingRegressor(estimators=base_models, final_estimator=RidgeCV())\n",
    "\n",
    "# 3. Train the model\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "stacking_model.fit(X_train, y_train)\n",
    "\n",
    "# 4. Save the Model\n",
    "# We save only the model because the dashboard will calculate metrics live\n",
    "joblib.dump(stacking_model, 'trained_hydro_model.pkl')\n",
    "\n",
    "print(\"✅ Model trained and 'trained_hydro_model.pkl' has been saved.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "befb8dd2-803c-41da-9cc3-73826227f1cc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1ebe109-0177-496c-92be-844b38b37d46",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
