import pandas as pd


# Predict churn for a single customer
def predict_customer(model_pipeline, input_df):

    # Extract preprocessing pipeline and trained model
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    # Transform input features
    X_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    # Convert transformed data into a DataFrame
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    # Predict customer churn
    prediction = model.predict(X_transformed_df)[0]

    # Predict class probabilities
    probabilities = model.predict_proba(X_transformed_df)[0]

    # Extract individual class probabilities
    no_churn_probability = probabilities[0]
    churn_probability = probabilities[1]

    # Return prediction results and model artifacts
    return (
        prediction,
        churn_probability,
        no_churn_probability,
        X_transformed_df,
        feature_names,
        model
    )
