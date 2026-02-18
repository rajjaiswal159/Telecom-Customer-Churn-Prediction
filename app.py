# import necessary files
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# load model
lgbm_model = joblib.load('model.pkl')


# -------------------- UI --------------------
st.header('Telecom Customer Churn Prediction')

st.subheader('Enter customer age')
age = st.text_input('', key='age')

st.subheader('Select customer Gender')
gender = st.selectbox('', ['Male', 'Female'])

st.subheader('How many months the customer has stayed with company')
tenure = st.text_input('', key='tenure')

st.subheader('How many times (in a month) a customer contacted customer support')
calls = st.text_input('', key='calls')

st.subheader('How many days (in a month) that customer actively used the service')
freq = st.text_input('', key='freq')

st.subheader('Enter number of days that customer delayed their payment')
d_pay = st.text_input('', key='d_pay')

st.subheader('Select customer Subscription Type')
subscription = st.selectbox('', ['Basic', 'Premium', 'Standard'])

st.subheader('Select contract length')
contract = st.selectbox('', ['Monthly', 'Quarterly', 'Annual'])

st.subheader('How much total amount of money a customer has paid to the company')
t_spend = st.text_input('', key='t_spend')

st.subheader('Enter days since last interaction with customer')
interact = st.text_input('', key='interact')


# -------------------- Prediction --------------------
if st.button('Predict'):

    # Validate numeric inputs
    try:
        age = int(age)
        tenure = int(tenure)
        calls = int(calls)
        freq = int(freq)
        d_pay = int(d_pay)
        t_spend = int(t_spend)
        interact = int(interact)
    except:
        st.error("Please enter valid numerical values.")
        st.stop()

    # Create input dataframe
    inp = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Tenure': [tenure],
        'Usage Frequency': [freq],
        'Support Calls': [calls],
        'Payment Delay': [d_pay],
        'Subscription Type': [subscription],
        'Contract Length': [contract],
        'Total Spend': [t_spend],
        'Last Interaction': [interact]
    })

    # Extract pipeline components
    preprocessor = lgbm_model.named_steps['preprocessor']
    model = lgbm_model.named_steps['model']

    # Transform input
    x_transformed = preprocessor.transform(inp)
    feature_names = preprocessor.get_feature_names_out()

    x_transformed_df = pd.DataFrame(
        x_transformed,
        columns=feature_names
    )

    # Make prediction
    prediction = model.predict(x_transformed_df)[0]
    proba = model.predict_proba(x_transformed_df)[0][1]

    st.markdown("### 📊 Model Prediction")

    if prediction == 1:
        st.error(f"🚨 Customer WILL Churn (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Customer will NOT Churn (Probability: {proba:.2%})")

    # -------------------- SHAP Explanation --------------------
    st.subheader("🔍 SHAP Waterfall Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_transformed_df)

    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=x_transformed_df.iloc[0],
        feature_names=feature_names
    )

    # Create figure properly (important for Streamlit)
    fig = plt.figure()
    shap.plots.waterfall(shap_exp, show=False)

    st.pyplot(fig)
    plt.clf()

