# Import required libraries
import shap
import matplotlib.pyplot as plt
import pandas as pd


# Generate SHAP explanations for a prediction
def generate_shap_explanation(model, x_transformed_df, feature_names):

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(x_transformed_df)

    # Create SHAP explanation object
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=x_transformed_df.iloc[0],
        feature_names=feature_names
    )

    # Generate SHAP waterfall plot
    fig = plt.figure()
    shap.plots.waterfall(shap_exp, show=False)

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })

    # Calculate absolute SHAP values
    importance_df["Abs SHAP"] = importance_df["SHAP Value"].abs()

    # Select top contributing features
    top_features = (
        importance_df
        .sort_values("Abs SHAP", ascending=False)
        .head(6)
    )

    # Return plot and top features
    return fig, top_features
