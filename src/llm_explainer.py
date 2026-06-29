import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_business_explanation(
    prediction,
    probability,
    top_features
):
    """
    Generate stakeholder-friendly explanation using Gemini.
    """

    feature_text = ""

    for _, row in top_features.iterrows():
        feature_text += (
            f"{row['Feature']} : {row['SHAP Value']:.4f}\n"
        )

    status = (
        "Customer WILL Churn"
        if prediction == 1
        else "Customer will NOT Churn"
    )

    prompt = f"""
You are a senior telecom business analyst.

A machine learning model predicted:

Prediction:
{status}

Probability:
{probability:.2%}

Top factors:

{feature_text}

Instructions:

1. Explain in simple business language.
2. Do NOT mention SHAP.
3. Explain positive and negative factors.
4. Maximum 180 words.
5. End with 3 actionable retention recommendations.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text