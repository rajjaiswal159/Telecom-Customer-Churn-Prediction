<h2>Telecom Customer Churn Prediction & Strategy Simulator</h2>
This project aims to help telecom companies identify customers at risk of leaving (churning) and provides a simulation tool to test the impact of retention strategies. It consists of a data science workflow for model building and a Streamlit-based web application for real-time predictions and business analysis.

<h2>🚀 Overview</h2>
The repository is structured into two main components:

1. Model Training Workflow (CustomerChurn.ipynb): A complete Jupyter Notebook covering data cleaning, Exploratory Data Analysis (EDA), and training multiple machine learning models (Random Forest, LightGBM, etc.) to predict churn.

2. Interactive Web App (app.py): A Streamlit application that allows users to predict churn for individual customers and simulate how changes in service factors (like price or support calls) affect overall churn rates.

<h2>🛠️ Features</h2>
● Single Customer Prediction: Input specific customer attributes (Age, Tenure, Usage Frequency, etc.) to get an instant churn prediction with probability.

● Segment Strategy Simulator: Adjust business variables (e.g., reducing payment delays or increasing tenure) across a segment to simulate the potential reduction in churn rates.

● Explainable AI: Integrated SHAP values to explain why a specific customer is predicted to churn.

● Automated Model Selection: Uses Optuna for hyperparameter tuning and compares models like RandomForest and LightGBM.

<h2>📊 Dataset Insights</h2>
The model is trained on a telecom dataset with over 500,000 records. Key features include:

● Customer Demographics: Age, Gender.

● Usage Metrics: Tenure, Usage Frequency, Total Spend.

● Service Interactions: Support Calls, Payment Delay, Last Interaction.

● Contract Details: Subscription Type (Basic, Standard, Premium), Contract Length (Monthly, Quarterly, Annual).

<h2>⚙️ Installation & Usage</h2>
Prerequisites:
● Python 3.8+

● Required libraries: streamlit, pandas, joblib, shap, matplotlib, scikit-learn, lightgbm, optuna.

<h2>Setup</h2>

1. Clone the repository:
```Bash
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction
```

2. Install dependencies:
```Bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```Bash
streamlit run app.py
```

<h2>📈 Model Performance</h2>
Based on the experimental results in the notebook, both RandomForest and LightGBM demonstrated high accuracy and comparable F1-scores, making them reliable for business deployment.

📁 Project Structure
● CustomerChurn.ipynb: Notebook for data preprocessing, EDA, and model training.

● app.py: Streamlit application script.

● model.pkl: The serialized trained model (used by the app).
