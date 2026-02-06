from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from google import genai
import shap
import time
import json

# ------------------ GEMINI SETUP ------------------
client = genai.Client(api_key="AIzaSyAkVCom9C8UUm0YpUE0MmuKT0uqfssEv4I")

def generate_gemini_strategy(reasons, churn_prob, domain):
    reasons = [str(r) for r in reasons]

    prompt = f"""
    Domain: {domain}
    You are a customer retention expert.
    A customer's churn probability is {churn_prob:.2f} (where 1.0 = highest risk).
    
    Based on this risk level and the SHAP feature impact values provided, give ONLY the TOP 3 most important retention strategies.

    STRICT RULES:
    - Give exactly 3 strategies
    - Each strategy must be only 1 to 3 sentences maximum
    - Each strategy must start on a NEW LINE
    - Be direct, practical, and action-oriented
    - Do NOT give explanations, categories, headings, or long paragraphs
    - Do NOT repeat the churn probability
    - Do NOT add introductions or conclusions
    
    SHAP Important Features:
    {', '.join(reasons)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.replace('.','\n')


# ------------------ FLASK APP ------------------
app = Flask(__name__)

def load_domain_assets(domain):
    model = joblib.load(f"models/{domain}_model.pkl")
    features = joblib.load(f"models/{domain}_features.pkl")
    transformer = None

    if domain in ["bank", "ecom"]:
        transformer = joblib.load(f"models/{domain}_scaler.pkl")

    return model, transformer, features

def load_feature_options():
    try:
        with open("models/feature_options.json") as f:
            return json.load(f)
    except Exception:
        return {}

UNWANTED_FEATURES = {
    "bank": {"CLIENTNUM"},
    "ecom": {"CustomerID"},
    "telecom": {"Customer Value"}
}

def get_numeric_features(domain, feature_names):
    options = load_feature_options().get(domain, {})
    if domain == "telecom":
        return [f for f in feature_names if f not in options.keys() and f not in UNWANTED_FEATURES.get(domain, set())]

    prefixes = [f"{key}_" for key in options.keys()]

    numeric = []
    for f in feature_names:
        if not any(f.startswith(p) for p in prefixes) and f not in UNWANTED_FEATURES.get(domain, set()):
            numeric.append(f)
    return numeric

def build_input_from_form(domain, feature_names, form_data):
    options = load_feature_options().get(domain, {})

    # start with all zeros
    input_data = {f: 0.0 for f in feature_names}

    # numeric fields
    numeric_features = get_numeric_features(domain, feature_names)
    for f in numeric_features:
        if f in form_data:
            input_data[f] = float(form_data.get(f, 0))

    # categorical fields
    for cat_name, values in options.items():
        selected = form_data.get(cat_name)
        if selected is None:
            continue

        if domain == "telecom":
            if cat_name in feature_names:
                input_data[cat_name] = float(selected)
        else:
            col_name = f"{cat_name}_{selected}"
            if col_name in input_data:
                input_data[col_name] = 1.0

    return input_data


@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/features/<domain>")
def get_features(domain):
    try:
        feature_names = joblib.load(f"models/{domain}_features.pkl")
        return get_numeric_features(domain, feature_names)

    except:
        return []


@app.route("/feature-options/<domain>")
def feature_options(domain):
    try:
        data = load_feature_options()
        return data.get(domain, {})
    except:
        return {}




@app.route("/predict", methods=["POST"])
def predict():
    try:
        domain = request.form["domain"]
        model, transformer, feature_names = load_domain_assets(domain)

        # ---------- DYNAMIC INPUT BUILDING ----------
        input_data = build_input_from_form(domain, feature_names, request.form)
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]

        # ---------- TRANSFORM INPUT ----------
        processed_input = transformer.transform(input_df) if transformer is not None else input_df

        # ---------- PREDICTION ----------
        prob = model.predict_proba(processed_input)[0][1]

        if prob < 0.4:
            risk = "Low Risk"
        elif prob < 0.7:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(input_df)

        # Handle binary/multi-output shapes safely
        if hasattr(shap_exp, "values") and getattr(shap_exp, "values").ndim == 3:
            # shape: (n_samples, n_features, n_outputs)
            shap_single = shap_exp.values[0, :, 1]
            base_value = shap_exp.base_values[0, 1]
        else:
            # shape: (n_samples, n_features)
            shap_single = shap_exp.values[0]
            base_value = shap_exp.base_values[0] if hasattr(shap_exp, "base_values") else explainer.expected_value

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_single,
                base_values=base_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            show=False
        )
        shap_filename = f"shap_plot_{int(time.time())}.png"
        shap_path = os.path.join("static", shap_filename)
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()

        feature_names = input_df.columns

        # Pair feature with absolute impact
        feature_impact = list(zip(feature_names, shap_single))

        # Sort by strongest impact
        feature_impact_sorted = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)

        # Get top N features
        top_n = 3
        top_features = [f[0] for f in feature_impact_sorted[:top_n]]


        # ---------- GEMINI STRATEGY ----------
        try:
            retention_action = generate_gemini_strategy(top_features, prob, domain)
        except Exception as e:
            retention_action = f"Retention strategy unavailable: {e}"

        return render_template(
            "output.html",
            prediction_text=f"{domain.upper()} Churn Probability: {prob:.2f}",
            risk_level=risk,
            retention_action=retention_action,
            shap_plot=shap_filename
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)

