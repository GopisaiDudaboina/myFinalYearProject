from flask import Flask, render_template, request, Response
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
from google import genai
import shap
import time
import json
import io
import numpy as np
from dotenv import load_dotenv

# ------------------ 1. SECURITY & CONFIG ------------------
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Securely set configurations
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_fallback")
API_KEY = os.getenv("GEMINI_API_KEY")

ARTIFACTS_DIR = "artifacts"
DATASETS_DIR = "datasets"

# Check for API Key
if not API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found. AI strategies will not work.")
    client = None
else:
    client = genai.Client(api_key=API_KEY)

# ------------------ 2. HELPER FUNCTIONS ------------------

def cleanup_shap_images():
    """Deletes old SHAP plots from static folder to save space."""
    try:
        files = glob.glob("static/shap_*.png")
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass # Ignore if file is in use
    except Exception as e:
        print(f"Cleanup Error: {e}")

def load_assets(domain):
    """Loads Model, Scaler, Features, and Encoders for a specific domain."""
    try:
        model = joblib.load(f"{ARTIFACTS_DIR}/{domain}_model.pkl")
        features = joblib.load(f"{ARTIFACTS_DIR}/{domain}_features.pkl")
        
        transformer = None
        encoders = None
        
        if os.path.exists(f"{ARTIFACTS_DIR}/{domain}_scaler.pkl"):
            transformer = joblib.load(f"{ARTIFACTS_DIR}/{domain}_scaler.pkl")
            
        if os.path.exists(f"{ARTIFACTS_DIR}/{domain}_encoders.pkl"):
            encoders = joblib.load(f"{ARTIFACTS_DIR}/{domain}_encoders.pkl")
            
        return model, transformer, features, encoders
    except FileNotFoundError as e:
        raise RuntimeError(f"Artifacts missing for {domain}. Please run training scripts.") from e

def load_config():
    """Loads the UI configuration (dropdown options)."""
    path = f"{ARTIFACTS_DIR}/ui_config.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_model_metadata(domain):
    """Loads model accuracy and training info."""
    path = f"{ARTIFACTS_DIR}/{domain}_metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"model_name": "Unknown", "accuracy": 0, "samples": 0}

def parse_numeric_features(domain, all_features):
    """Separates features into Numeric and Categorical for the UI."""
    config = load_config().get(domain, {})
    options = config.get('options', {})
    # Identify categorical columns based on One-Hot prefixes (e.g., Country_France)
    cat_prefixes = [f"{key}_" for key in options.keys()]
    
    numeric = []
    for f in all_features:
        # If feature matches a dropdown key or starts with a One-Hot prefix, it's categorical
        is_cat = f in options or any(f.startswith(p) for p in cat_prefixes)
        if not is_cat:
            numeric.append(f)
    return numeric

def generate_gemini_strategy(reasons, churn_prob, domain):
    """Uses Gemini AI to generate retention strategies."""
    if not client:
        return "AI Strategy unavailable (Missing API Key)."
        
    try:
        reasons_text = ', '.join([str(r) for r in reasons])
        prompt = f"""
        Role: Senior Retention Strategist for {domain} industry.
        Context: Customer Churn Risk is {churn_prob:.2%}.
        Key Risk Drivers: {reasons_text}
        
        Task: Provide 3 distinct, high-impact retention strategies.
        Format:
        - Strategy 1: [Actionable Advice]
        - Strategy 2: [Actionable Advice]
        - Strategy 3: [Actionable Advice]
        
        Constraints: No intro/outro. Direct and professional tone.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        return response.text.replace('.', '\n')
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "AI Strategy currently unavailable. Please rely on risk factors."

# ------------------ 3. ROUTE DEFINITIONS ------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/config/<domain>")
def get_domain_config(domain):
    """API to send form configuration to frontend."""
    try:
        all_features = joblib.load(f"{ARTIFACTS_DIR}/{domain}_features.pkl")
        config = load_config().get(domain, {}).get('options', {})
        numeric = parse_numeric_features(domain, all_features)
        return {"options": config, "numeric": numeric}
    except:
        return {"options": {}, "numeric": []}

@app.route("/api/template/<domain>")
def download_template(domain):
    """Generates a CSV template for batch upload."""
    try:
        all_features = joblib.load(f"{ARTIFACTS_DIR}/{domain}_features.pkl")
        # For the template, we want the raw categorical names, not one-hot encoded ones
        numeric = parse_numeric_features(domain, all_features)
        config = load_config().get(domain, {}).get('options', {})
        cols = numeric + list(config.keys())
        
        df = pd.DataFrame(columns=cols)
        output = io.StringIO()
        df.to_csv(output, index=False)
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-disposition": f"attachment; filename={domain}_template.csv"})
    except Exception as e:
        return str(e), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Handles Bulk CSV Uploads."""
    try:
        domain = request.form.get("domain")
        file = request.files.get("file")
        
        if not file or not domain: return "Missing file or domain", 400

        model, transformer, features, encoders = load_assets(domain)
        df = pd.read_csv(file)
        input_df = df.copy()
        
        # --- PREPROCESSING PIPELINES ---
        if domain == "telecom" and encoders:
            # Label Encoding for Telecom
            for col, le in encoders.items():
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
        elif domain == "retail":
            # SPECIFIC CLEANING FOR RETAIL MODEL
            # 1. Drop IDs and Leakage columns if present
            for col in ["CustomerID", "DaySinceLastOrder", "Complain"]:
                if col in input_df.columns:
                    input_df.drop(col, axis=1, inplace=True)
            
            # 2. Normalize Categories (Match training logic)
            if "PreferredLoginDevice" in input_df.columns:
                input_df["PreferredLoginDevice"] = input_df["PreferredLoginDevice"].replace("Phone", "Mobile Phone")
            if "PreferedOrderCat" in input_df.columns:
                input_df["PreferedOrderCat"] = input_df["PreferedOrderCat"].replace("Mobile", "Mobile Phone")
                
            input_df = pd.get_dummies(input_df)
            
        elif domain == "bank": 
            input_df = pd.get_dummies(input_df)
            
        # Align columns with training data (Fill missing with 0, drop extras)
        input_df = input_df.reindex(columns=features, fill_value=0)
        
        # Scale
        if transformer: 
            data_processed = transformer.transform(input_df)
        else: 
            data_processed = input_df
            
        # Predict
        probs = model.predict_proba(data_processed)[:, 1]
        df["Churn_Score"] = probs
        df["Risk_Category"] = pd.cut(probs, bins=[-1, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
        
        # Return CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-disposition": "attachment; filename=analysis_results.csv"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Batch processing failed: {str(e)}", 500

@app.route("/predict/single", methods=["POST"])
def predict_single():
    """Handles Single Customer Prediction via Form."""
    try:
        domain = request.form.get("domain")
        model, transformer, features, encoders = load_assets(domain)
        config = load_config().get(domain, {}).get('options', {})
        
        # Load Metadata for Report Banner
        meta = load_model_metadata(domain)

        # 1. Parse Form Data
        input_data = {f: 0.0 for f in features} # Start with 0s
        numeric_cols = parse_numeric_features(domain, features)
        
        # Fill Numeric
        for f in numeric_cols: 
            val = request.form.get(f)
            if val:
                input_data[f] = float(val)
            
        # Fill Categorical
        for cat, opts in config.items():
            val = request.form.get(cat)
            if not val: continue
            
            if domain == "telecom" and encoders and cat in encoders:
                try: input_data[cat] = encoders[cat].transform([val])[0]
                except: input_data[cat] = 0
            else:
                # One-Hot Encoding Logic (e.g. Country_France = 1)
                col_name = f"{cat}_{val}"
                if col_name in input_data: input_data[col_name] = 1.0

        # 2. Create DataFrame & Scale
        df = pd.DataFrame([input_data])
        processed = transformer.transform(df) if transformer else df
            
        # 3. Predict
        prob = model.predict_proba(processed)[0][1]
        risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        
        # 4. Generate SHAP Plot
        cleanup_shap_images() # Delete old images first
        shap_file = f"shap_{int(time.time())}.png"
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(df)
            
            # Handle different SHAP output formats
            vals = shap_values.values
            if vals.ndim == 3: vals = vals[:,:,1] # For binary classification
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=vals[0], 
                    base_values=shap_values.base_values[0] if vals.ndim==2 else shap_values.base_values[0][1], 
                    data=df.iloc[0], 
                    feature_names=features
                ), 
                show=False
            )
            plt.savefig(f"static/{shap_file}", bbox_inches="tight")
            plt.close()
            
            # Identify Top 3 Factors for Gemini
            impact = sorted(zip(features, vals[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
            top_factors = [f"{n}" for n, v in impact]
        except Exception as e:
            print(f"SHAP Error: {e}")
            shap_file = None
            top_factors = []
            
        # 5. Generate AI Strategy
        strategy = generate_gemini_strategy(top_factors, prob, domain)
        
        # 6. Render Report
        return render_template(
            "report.html",
            score=f"{prob:.1%}",
            risk=risk,
            strategy=strategy,
            chart=shap_file,
            model_name=meta.get("model_name"),
            accuracy=meta.get("accuracy"),
            samples=meta.get("samples")
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Prediction Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)