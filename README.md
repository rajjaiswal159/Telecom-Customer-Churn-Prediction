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

 • 📊 Total Rows: 505,207

 • 📌 Total Features: 12

Key Features:

 • Age

 • Gender

 • Tenure

 • Usage Frequency

 • Support Calls

 • Payment Delay

 • Subscription Type (Basic, Standard, Premium)

 • Contract Length (Monthly, Quarterly, Annual)

 • Total Spend

 • Last Interaction

 • Target Variable: Churn (0/1)

<h2>🔎 Exploratory Data Analysis (EDA) Insights</h2>

Key business insights discovered:

📌 Customers with Monthly contracts have higher churn rate.

📌 Customers with more than 4 support calls show strong churn tendency.

📌 Payment delays greater than 20 days significantly increase churn probability.

📌 Customers with Total Spend < 500 are more likely to churn.

📌 Customers above 50 years have higher churn probability.

<h2>🛠 Tech Stack</h2>

 • Python

 • Pandas

 • NumPy

 • Matplotlib & Seaborn

 • Scikit-learn

 • XGBoost

 • LightGBM

 • Optuna (Hyperparameter Tuning)

 • SHAP (Explainable AI)

 • Streamlit (Deployment)

<h2>⚙️ Data Preprocessing</h2>

 • Dropped CustomerID

 • Removed null values

 • Converted float features to integer

 • Used ColumnTransformer

 • OneHotEncoder → Gender

 • OrdinalEncoder → Subscription Type & Contract Length

 • Pipeline used for clean preprocessing + modeling

<h2>🤖 Model Selection</h2>

The following models were evaluated using cross-validation:

 • Logistic Regression

 • Decision Tree

 • Random Forest

 • XGBoost

 • LightGBM

<h2>📈 Best Performing Models:</h2>

 • Random Forest

 • LightGBM

<h2>🔧 Hyperparameter Tuning</h2>

Hyperparameter tuning was performed using Optuna for:

 • Random Forest

 • LightGBM

Both models showed comparable performance with similar:

 • Accuracy

 • Precision

 • Recall

 • F1-score

<h2>📊 Model Evaluation</h2>

Evaluation Metrics Used:

 • Accuracy

 • Precision

 • Recall

 • F1-Score

 • Classification Report

Both RandomForest and LightGBM achieved strong and balanced performance on the test set.

<h2>🚀 Streamlit Web App</h2>

An interactive web application was built using Streamlit where users can:

 • Enter customer details

 • Get churn prediction

 • View churn probability

 • See SHAP Waterfall explanation for transparency

<h2>🔍 Features of the App:</h2>

✔ Real-time prediction

✔ Probability score

✔ SHAP explainability

✔ Clean UI

<h2>🧠 Explainable AI (SHAP)</h2>

To ensure model transparency:

 • Used shap.TreeExplainer

 • Generated SHAP Waterfall plots

 • Identified top contributing features for each prediction

This makes the model production-ready and trustworthy.

<h2>📁 Project Structure</h2>
Customer-Churn-Prediction/
│
├── CustomerChurn.ipynb
├── app.py
├── model.pkl
├── mydata.csv
└── README.md

<h2>💻 How to Run Locally</h2>
```bash
git clone https://github.com/rajjaiswal159/Telecom-Customer-Churn-Prediction.git
cd customer-churn-prediction
```

<h2>2️⃣ Install dependencies</h2>
pip install -r requirements.txt

<h2>3️⃣ Run Streamlit app</h2>
streamlit run app.py

<h2>📌 Key Learning Outcomes</h2>

End-to-end ML Pipeline creation

Feature engineering & preprocessing

Hyperparameter tuning with Optuna

Model comparison & evaluation

Explainable AI using SHAP

Model deployment using Streamlit

<h2>👨‍💻 Author</h2>

Raj Jaiswal
B.Tech (Computer Science & Engineering)
Aspiring Data Scientist

⭐ If you found this project useful, consider giving it a star!
