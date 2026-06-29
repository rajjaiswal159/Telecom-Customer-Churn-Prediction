import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def generate_shap_explanation(model, x_transformed_df, feature_names):
    """
    Returns:
    - matplotlib figure
    - top SHAP features dataframe
    """

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

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })

    importance_df["Abs SHAP"] = importance_df["SHAP Value"].abs()

    top_features = (
        importance_df
        .sort_values("Abs SHAP", ascending=False)
        .head(6)
    )

    return fig, top_features