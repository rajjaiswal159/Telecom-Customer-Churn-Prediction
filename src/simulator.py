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

def calculate_churn_rate(
    model_pipeline,
    df
):

    preprocessor = model_pipeline.named_steps["preprocessor"]

    model = model_pipeline.named_steps["model"]

    X = df.drop(
        columns=["Churn"],
        errors="ignore"
    )

    X = preprocessor.transform(X)

    prediction = model.predict(X)

    return prediction.mean() * 100