import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


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

    st.subheader("Enter customer age")
    age = st.text_input("", key="age")

    st.subheader("Select customer Gender")
    gender = st.selectbox("", ["Male", "Female"])

    st.subheader("How many months the customer has stayed with company")
    tenure = st.text_input("", key="tenure")

    st.subheader("How many times customer contacted support")
    calls = st.text_input("", key="calls")

    st.subheader("How many days customer used the service")
    freq = st.text_input("", key="freq")

    st.subheader("Enter number of days payment delay")
    d_pay = st.text_input("", key="d_pay")

    st.subheader("Select customer Subscription Type")
    subscription = st.selectbox("", ["Basic", "Premium", "Standard"])

    st.subheader("Select contract length")
    contract = st.selectbox("", ["Monthly", "Quarterly", "Annual"])

    st.subheader("Total money customer spent")
    t_spend = st.text_input("", key="t_spend")

    st.subheader("Days since last interaction")
    interact = st.text_input("", key="interact")



    # -------------------- Prediction --------------------
    if st.button("Predict Churn"):

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

        preprocessor = model_pipeline.named_steps["preprocessor"]
        model = model_pipeline.named_steps["model"]

        x_transformed = preprocessor.transform(inp)
        feature_names = preprocessor.get_feature_names_out()

        x_transformed_df = pd.DataFrame(
            x_transformed,
            columns=feature_names
        )

        prediction = model.predict(x_transformed_df)[0]
        churn_proba = model.predict_proba(x_transformed_df)[0][1]
        no_churn_proba = model.predict_proba(x_transformed_df)[0][0]

        st.markdown("### 📊 Model Prediction")

        if prediction == 1:
            st.error(f"🚨 Customer WILL Churn (Probability: {churn_proba:.2%})")
        else:
            st.success(f"✅ Customer will NOT Churn (Probability: {no_churn_proba:.2%})")


        # -------------------- SHAP --------------------
        st.subheader("🔍 SHAP Explanation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_transformed_df)

        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=x_transformed_df.iloc[0],
            feature_names=feature_names
        )

        fig = plt.figure()
        shap.plots.waterfall(shap_exp, show=False)

        st.pyplot(fig)
        plt.clf()



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
    def apply_modifications(
        df,
        tenure_adj,
        calls_adj,
        delay_adj,
        usage_adj,
        spend_adj,
        interaction_adj,
        contract,
        subscription
    ):

        df = df.copy()

        df["Tenure"] = df["Tenure"] + tenure_adj
        df["Support Calls"] = (df["Support Calls"] + calls_adj).clip(lower=0)
        df["Payment Delay"] = (df["Payment Delay"] + delay_adj).clip(lower=0)

        df["Usage Frequency"] = (df["Usage Frequency"] + usage_adj).clip(lower=0)
        df["Total Spend"] = (df["Total Spend"] + spend_adj).clip(lower=0)
        df["Last Interaction"] = (df["Last Interaction"] + interaction_adj).clip(lower=0)

        if contract != "No Change":
            df["Contract Length"] = contract

        if subscription != "No Change":
            df["Subscription Type"] = subscription

        return df


    # -------------------- Churn Calculation --------------------
    def calculate_churn_rate(df):

        preprocessor = model_pipeline.named_steps["preprocessor"]
        model = model_pipeline.named_steps["model"]

        X = df.drop(columns=["Churn"], errors="ignore")

        X_transformed = preprocessor.transform(X)

        preds = model.predict(X_transformed)

        return preds.mean() * 100



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

            original_rate = calculate_churn_rate(original_df)
            new_rate = calculate_churn_rate(modified_df)

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