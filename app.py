import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.predictor import predict_customer
from src.shap_explainer import generate_shap_explanation
from src.llm_explainer import generate_business_explanation
from src.simulator import (
    apply_modifications,
    calculate_churn_rate
)


# -------------------- Page Config --------------------
st.set_page_config(page_title="Customer Churn Platform", layout="wide")


# -------------------- Load Resources --------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("clean_df.csv")


model_pipeline = load_model()
original_df = load_data()


# -------------------- App Mode Selection --------------------
st.sidebar.title("Navigation")

app_mode = st.sidebar.radio(
    "Choose Application",
    [
        "Single Customer Prediction",
        "Customer Segment Strategy Simulator"
    ]
)


# ==========================================================
#               1️⃣ INDIVIDUAL CUSTOMER PREDICTION
# ==========================================================
if app_mode == "Single Customer Prediction":

    st.header("👤 Telecom Customer Churn Prediction")

    with st.form("prediction_form"):

        st.subheader("Enter customer age")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )
        
            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                value=12
            )
        
            freq = st.number_input(
                "Usage Frequency",
                min_value=0,
                value=15
            )
        
            subscription = st.selectbox(
                "Subscription Type",
                ["Basic", "Standard", "Premium"]
            )
        
            t_spend = st.number_input(
                "Total Spend",
                min_value=0,
                value=500,
                step=100
            )

        with col2:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )
        
            calls = st.number_input(
                "Support Calls",
                min_value=0,
                value=2
            )
        
            d_pay = st.number_input(
                "Payment Delay (Days)",
                min_value=0,
                value=3
            )
        
            contract = st.selectbox(
                "Contract Length",
                ["Monthly", "Quarterly", "Annual"]
            )
        
            interact = st.number_input(
                "Last Interaction",
                min_value=0,
                value=5
            )



    # -------------------- Prediction --------------------
        submitted = st.form_submit_button(
            "🔍 Predict Churn",
            use_container_width=True
        )
    
    if submitted:

        inp = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Tenure": [tenure],
            "Usage Frequency": [freq],
            "Support Calls": [calls],
            "Payment Delay": [d_pay],
            "Subscription Type": [subscription],
            "Contract Length": [contract],
            "Total Spend": [t_spend],
            "Last Interaction": [interact]
        })

        (
            prediction,
            churn_proba,
            no_churn_proba,
            x_transformed_df,
            feature_names,
            model,
        ) = predict_customer(model_pipeline, inp)


        st.subheader("📊 Prediction Result")
        
        probability = churn_proba if prediction == 1 else no_churn_proba
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
        
            if prediction == 1:
                st.error("🚨 High Churn Risk")
            else:
                st.success("✅ Low Churn Risk")
        
        with col2:
        
            st.metric(
                "Confidence",
                f"{probability:.2%}"
            )
        
        with col3:
        
            if churn_proba >= 0.80:
                st.metric("Risk Level", "🔴 High")
        
            elif churn_proba >= 0.50:
                st.metric("Risk Level", "🟡 Medium")
        
            else:
                st.metric("Risk Level", "🟢 Low")


        # -------------------- SHAP --------------------
        fig, top_features = generate_shap_explanation(
            model,
            x_transformed_df,
            feature_names
        )
        

        with st.spinner("Generating AI explanation..."):

            explanation = generate_business_explanation(
                prediction,
                churn_proba if prediction == 1 else no_churn_proba,
                top_features
            )
        
        st.divider()

        with st.container(border=True):
        
            st.subheader("🤖 AI Business Insights")
        
            st.markdown(explanation)
        
        st.divider()

        with st.expander("🔬 Technical Model Explanation (For Data Scientists)"):

            st.pyplot(fig)
    
            technical_df = top_features.copy()

            technical_df.columns = [
                "Feature",
                "Contribution",
                "Importance"
            ]
            
            st.dataframe(
                technical_df,
                use_container_width=True
            )



# ==========================================================
#              2️⃣ CUSTOMER SEGMENT STRATEGY SIMULATOR
# ==========================================================
elif app_mode == "Customer Segment Strategy Simulator":

    st.header("📉 Strategy Impact Simulator")

    st.write(
        "Simulate business strategies by modifying customer features "
        "and observe their effect on churn rate."
    )


    # -------------------- Data Modification --------------------


    # -------------------- Churn Calculation --------------------



    # -------------------- Sidebar Controls --------------------
    st.sidebar.header("Strategy Controls")

    st.sidebar.subheader("Customer Behavior")

    tenure_adj = st.sidebar.slider("Add Months to Tenure", 0, 24, 0)

    usage_adj = st.sidebar.slider("Increase Usage Frequency", -10, 20, 0)
    spend_adj = st.sidebar.slider("Increase Total Spend", -500, 2000, 0)

    interaction_adj = st.sidebar.slider("Change Last Interaction (days)", -30, 30, 0)

    st.sidebar.subheader("Customer Issues")

    calls_adj = st.sidebar.slider("Reduce Support Calls", -5, 0, 0)
    delay_adj = st.sidebar.slider("Reduce Payment Delay", -15, 0, 0)

    st.sidebar.subheader("Plan Strategy")

    contract = st.sidebar.selectbox(
        "Change Contract Length",
        ["No Change", "Monthly", "Quarterly", "Annual"]
    )

    subscription = st.sidebar.selectbox(
        "Change Subscription Type",
        ["No Change", "Basic", "Standard", "Premium"]
    )


    # -------------------- Analysis --------------------
    st.subheader("Compare Churn Rates")

    if st.button("🚀 Run Impact Analysis"):

        with st.spinner("Running simulation..."):

            modified_df = apply_modifications(
                original_df,
                tenure_adj,
                calls_adj,
                delay_adj,
                usage_adj,
                spend_adj,
                interaction_adj,
                contract,
                subscription
            )

            original_rate = calculate_churn_rate(
                model_pipeline,
                original_df
            )
            new_rate = calculate_churn_rate(
                model_pipeline,
                modified_df
            )

            diff = new_rate - original_rate
            reduction = original_rate - new_rate

        col1, col2, col3 = st.columns(3)

        col1.metric("Baseline Churn", f"{original_rate:.2f}%")
        col2.metric("Simulated Churn", f"{new_rate:.2f}%", delta=f"{diff:.2f}%", delta_color="inverse")
        col3.metric("Churn Reduced", f"{max(0, reduction):.2f}%")

        if diff < 0:
            st.success(f"✅ Strategy may reduce churn by {abs(diff):.2f}%")

        elif diff > 0:
            st.warning(f"⚠️ Strategy may increase churn by {diff:.2f}%")

        else:
            st.info("No significant change detected.")

        with st.expander("View Modified Data Sample"):
            st.dataframe(modified_df.head(10))

    else:
        st.info("Adjust strategy controls and run the simulation.")