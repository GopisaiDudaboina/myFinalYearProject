# 🔮 ChurnIntel AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask)
![Machine Learning](https://img.shields.io/badge/AI-Scikit--Learn-orange?style=for-the-badge)
![Gemini AI](https://img.shields.io/badge/GenAI-Gemini%20Flash-purple?style=for-the-badge)

**ChurnIntel AI** is an enterprise-grade Machine Learning application designed to predict **Customer Churn** (attrition) across three major industries: **Telecom, Banking, and Retail.**

Unlike traditional predictors, this tool provides **Explainable AI (XAI)** insights using SHAP and generates **Actionable Retention Strategies** using Google's Gemini AI. It is designed to help businesses understand *why* customers leave and *how* to keep them.

---

## 🚀 Key Features

* **⚡ Multi-Domain Support:** specialized ML models for Telecom, Banking, and Retail sectors.
* **🧠 Explainable AI (SHAP):** Visualizes the exact reasons for churn (e.g., "High Monthly Charges", "Low Satisfaction Score").
* **🤖 Generative AI Strategies:** Integrated with **Google Gemini 2.0 Flash** to automatically generate personalized retention plans for at-risk customers.
* **📂 Dual-Mode Prediction:**
    * **Single Mode:** Interactive form for individual customer analysis.
    * **Batch Mode:** Upload CSV files to process thousands of customers at once.
* **🛡️ Secure & Optimized:** Uses environment variables for API security and auto-cleans visualization files to save storage.
* **🎨 Modern UI:** Features a "Midnight Fire" glassmorphism design with responsive animations and loading states.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript
* **Backend:** Flask (Python)
* **Machine Learning:**
    * **Telecom:** CatBoost Classifier
    * **Banking:** XGBoost Classifier
    * **Retail:** LightGBM (Gradient Boosting)
* **AI & Explainability:** SHAP (Shapley Additive Explanations), Google Gemini API

---

## 📂 Project Structure

```text
ChurnIntel-AI/
├── app.py                   # Main Flask Application
├── .env                     # API Keys (Security - Not uploaded to GitHub)
├── requirements.txt         # Python Dependencies
├── artifacts/               # Saved Models (.pkl), Scalers & Metadata
├── datasets/                # Raw CSV Training Data
├── training_scripts/        # Scripts to retrain models
│   ├── train_telecom.py
│   ├── train_banking.py
│   └── train_retail.py
├── static/                  # CSS, JS, and Images
│   ├── style.css
│   ├── script.js
│   └── shap_*.png           # Generated plots (auto-deleted)
└── templates/               # HTML Pages
    ├── index.html           # Landing Page
    ├── dashboard.html       # Prediction Interface
    └── report.html          # Analysis Results Page
```

## ⚙️ Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/churnintel-ai.git](https://github.com/your-username/churnintel-ai.git)
cd churnintel-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Security Keys
Create a file named .env in the root folder (same level as app.py) and add your keys:
```TOML
# .env file content
GEMINI_API_KEY = your_google_gemini_key_here
FLASK_SECRET_KEY = any_random_secret_string_for_security
```

### 4. Train the Models
If the artifacts/ folder is empty, generate the models by running:
```bash
python training_scripts/train_telecom.py
python training_scripts/train_banking.py
python training_scripts/train_retail.py
```

### 5. Run the Application
```bash
python app.py
```
Open your browser and visit: http://127.0.0.1:5000

## 📊 How to Use

Single Prediction
1. Go to the Dashboard.

2. Select the industry (Telecom, Banking, or Retail).

3. Fill in the customer details (e.g., Credit Score, Tenure, Monthly Charges).

4. Click Analyze Risk.

5. View the Churn Score, SHAP Graph, and AI Strategy.

Batch Prediction
1. Go to the Dashboard and switch to "Batch Upload".

2. Download the CSV Template for your selected industry.

3. Fill the CSV with customer data.

4. Upload the file.

5. The app will return a downloadable CSV with Churn Scores and Risk Categories for every customer.

## 🛡️ Security Note

This project uses a .env file to store sensitive API keys. Do not upload your .env file to GitHub. The .gitignore file included in this repository prevents this automatically.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

### Built with ❤️ by Gopisai Dudaboina
