<h1>📊 Telecom Customer Churn Prediction</h1>
<h2>📌 Project Overview</h2>

Customer churn is one of the biggest challenges in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project builds a Machine Learning pipeline to predict whether a telecom customer will churn based on demographic, behavioral, and subscription-related features.

The final solution is deployed using Streamlit with SHAP-based explainability to provide transparent predictions.

<h2>🎯 Business Problem</h2>

Telecom companies face revenue loss due to customer churn.

The objective of this project is:

To accurately predict whether a customer will churn so that the company can take proactive retention actions.

<h2>📂 Dataset Information</h2>

📊 Total Rows: 505,207

📌 Total Features: 12

</h3>Key Features:</h3>

Age

Gender

Tenure

Usage Frequency

Support Calls

Payment Delay

Subscription Type (Basic, Standard, Premium)

Contract Length (Monthly, Quarterly, Annual)

Total Spend

Last Interaction

Target Variable: Churn (0/1)

🔎 Exploratory Data Analysis (EDA) Insights

Key business insights discovered:

📌 Customers with Monthly contracts have higher churn rate.

📌 Customers with more than 4 support calls show strong churn tendency.

📌 Payment delays greater than 20 days significantly increase churn probability.

📌 Customers with Total Spend < 500 are more likely to churn.

📌 Customers above 50 years have higher churn probability.

🛠 Tech Stack

Python

Pandas

NumPy

Matplotlib & Seaborn

Scikit-learn

XGBoost

LightGBM

Optuna (Hyperparameter Tuning)

SHAP (Explainable AI)

Streamlit (Deployment)

⚙️ Data Preprocessing

Dropped CustomerID

Removed null values

Converted float features to integer

Used ColumnTransformer

OneHotEncoder → Gender

OrdinalEncoder → Subscription Type & Contract Length

Pipeline used for clean preprocessing + modeling

🤖 Model Selection

The following models were evaluated using cross-validation:

Logistic Regression

Decision Tree

Random Forest

XGBoost

LightGBM

📈 Best Performing Models:

Random Forest

LightGBM

🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using Optuna for:

Random Forest

LightGBM

Both models showed comparable performance with similar:

Accuracy

Precision

Recall

F1-score

📊 Model Evaluation

Evaluation Metrics Used:

Accuracy

Precision

Recall

F1-Score

Classification Report

Both RandomForest and LightGBM achieved strong and balanced performance on the test set.

🚀 Streamlit Web App

An interactive web application was built using Streamlit where users can:

Enter customer details

Get churn prediction

View churn probability

See SHAP Waterfall explanation for transparency

🔍 Features of the App:

✔ Real-time prediction
✔ Probability score
✔ SHAP explainability
✔ Clean UI

🧠 Explainable AI (SHAP)

To ensure model transparency:

Used shap.TreeExplainer

Generated SHAP Waterfall plots

Identified top contributing features for each prediction

This makes the model production-ready and trustworthy.

📁 Project Structure
Customer-Churn-Prediction/
│
├── CustomerChurn.ipynb
├── app.py
├── model.pkl
├── mydata.csv
└── README.md

💻 How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit app
streamlit run app.py

📸 App Preview

(Add screenshots of your Streamlit app here)

🔮 Future Improvements

Add ROC-AUC visualization

Deploy on Streamlit Cloud / AWS / Render

Add model comparison dashboard

Add feature importance visualization inside app

Handle class imbalance using advanced sampling techniques

📌 Key Learning Outcomes

End-to-end ML Pipeline creation

Feature engineering & preprocessing

Hyperparameter tuning with Optuna

Model comparison & evaluation

Explainable AI using SHAP

Model deployment using Streamlit

👨‍💻 Author

Raj Jaiswal
B.Tech (Computer Science & Engineering)
Aspiring Data Scientist

⭐ If you found this project useful, consider giving it a star!
