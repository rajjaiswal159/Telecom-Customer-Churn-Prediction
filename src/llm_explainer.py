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
You are a Senior Telecom Business Consultant.

Prediction:
{status}

Probability:
{probability:.2%}

Key Factors:

{feature_text}

Write a report for a BUSINESS STAKEHOLDER.

Rules:

- Never mention SHAP.
- Never mention Machine Learning.
- Never mention AI model.
- Never mention feature importance.
- Never mention algorithms.
- Use simple English.
- Maximum 150 words.

Return exactly in this format:

## Why is this customer at risk?

(Explain in simple language.)

## Business Insights

- Point 1
- Point 2
- Point 3

## Recommended Actions

- Action 1
- Action 2
- Action 3
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text