# Apply strategy changes to the customer dataset
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

    df["Tenure"] += tenure_adj

    df["Support Calls"] = (
        df["Support Calls"] + calls_adj
    ).clip(lower=0)

    df["Payment Delay"] = (
        df["Payment Delay"] + delay_adj
    ).clip(lower=0)

    df["Usage Frequency"] = (
        df["Usage Frequency"] + usage_adj
    ).clip(lower=0)

    df["Total Spend"] = (
        df["Total Spend"] + spend_adj
    ).clip(lower=0)

    df["Last Interaction"] = (
        df["Last Interaction"] + interaction_adj
    ).clip(lower=0)

    if contract != "No Change":
        df["Contract Length"] = contract

    if subscription != "No Change":
        df["Subscription Type"] = subscription

    return df


# Calculate predicted churn rate for the dataset
def calculate_churn_rate(
    model_pipeline,
    df
):

    # Extract preprocessing pipeline and trained model
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    X = df.drop(
        columns=["Churn"],
        errors="ignore"
    )

    # Transform input features
    X = preprocessor.transform(X)

    # Predict churn for all customers
    prediction = model.predict(X)

    # Return churn rate as a percentage
    return prediction.mean() * 100
