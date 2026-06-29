import pandas as pd


def predict_customer(model_pipeline, input_df):

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    prediction = model.predict(X_transformed_df)[0]

    probabilities = model.predict_proba(X_transformed_df)[0]

    no_churn_probability = probabilities[0]
    churn_probability = probabilities[1]

    return (
        prediction,
        churn_probability,
        no_churn_probability,
        X_transformed_df,
        feature_names,
        model
    )