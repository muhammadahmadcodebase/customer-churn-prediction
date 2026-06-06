# AI-Powered Customer Churn Prediction & Retention Intelligence Platform

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Models Trained](#models-trained)
- [Final Model](#final-model)
- [Business Intelligence Features](#business-intelligence-features)
- [Explainable AI](#explainable-ai)
- [Dashboard Features](#dashboard-features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Key Business Insights](#key-business-insights)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [LICENSE](#license)

## Project Overview

This repository delivers an end-to-end customer churn prediction and retention intelligence platform. It combines machine learning, business intelligence, explainable AI, and interactive analytics to help businesses identify customers likely to churn and recommend the best retention actions.

The system is built to support:

- Customer churn prediction
- Risk segmentation and revenue exposure analysis
- Personalized retention recommendations
- SHAP-based model explainability
- Executive decision-making via an interactive dashboard

## Dataset

- **Dataset:** Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Problem Type:** Binary classification
- **Target Variable:** `Churn`
- **Classes:** `Yes`, `No`

## Feature Engineering

The platform includes engineered features designed to improve predictive power and business interpretability.

| Feature                         | Description                                     | Business Importance                                   |
| ------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| `CLV` (Customer Lifetime Value) | Monthly spending multiplied by tenure           | Identifies customer value and long-term churn risk    |
| `AvgRevenuePerMonth`            | Total charges divided by tenure plus one        | Normalizes revenue performance across customer tenure |
| `ContractRiskScore`             | Risk score assigned by contract type            | Captures contract-driven retention exposure           |
| `AutoPay`                       | Flag for automatic payment methods              | Indicates convenience and retention loyalty           |
| `HighRiskCustomer`              | Flag for month-to-month contract and low tenure | Highlights customers with immediate churn risk        |
| `TotalServices`                 | Count of active service subscriptions           | Measures customer engagement and product adoption     |
| `TenureGroup`                   | Customer tenure bucket                          | Segments customers by lifecycle stage                 |
| `MonthlySpendCategory`          | Monthly spend bucket                            | Identifies spend behavior and value tier              |
| `RevenueSegment`                | Total charges bucket                            | Distinguishes customers by historical revenue impact  |

Each engineered feature increases model performance while improving business interpretability for analytics teams.

## Machine Learning Pipeline

The platform follows a complete ML lifecycle from raw data to production-ready inference.

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Exploratory Data Analysis
5. Feature Engineering
6. Feature Selection
7. Data Preprocessing
8. Model Training
9. Hyperparameter Tuning
10. Threshold Optimization
11. Model Evaluation
12. Model Comparison
13. Model Persistence
14. Deployment Preparation

## Models Trained

Multiple candidate models were evaluated to identify the most reliable tradeoff between accuracy, interpretability, and business value.

| Model                  | Purpose                                     |
| ---------------------- | ------------------------------------------- |
| Logistic Regression    | Baseline explainable model                  |
| Random Forest          | Robust ensemble baseline                    |
| Extra Trees            | Fast ensemble model with variance reduction |
| Hist Gradient Boosting | High-performance gradient boosting          |
| LightGBM               | Efficient tree boosting for medium data     |
| CatBoost               | Categorical-aware boosting model            |
| XGBoost                | Final production champion                   |

Evaluating multiple algorithms ensures the final model selection is defensible and suited to business outcomes.

## Final Model

- **Production Model:** Tuned XGBoost
- **Why Selected:** Best balance of accuracy, probability calibration, and explainability.

### Reported Performance

- **Accuracy:** ≈ 80%
- **Precision:** ≈ 65%
- **Recall:** ≈ 56%
- **F1 Score:** ≈ 60%
- **ROC-AUC:** ≈ 84%

Final selection was driven by business usefulness, strong overall performance, and interpretability for customer retention decisions.

## Business Intelligence Features

This project extends churn prediction into actionable business intelligence.

### Customer Risk Segmentation

- **Low Risk**
- **Medium Risk**
- **High Risk**
- **Critical Risk**

This segmentation helps customer success teams prioritize the highest-risk customers first.

### Revenue Risk Analysis

The dashboard estimates annual revenue at risk by combining churn probability with monthly customer revenue.

### Customer Health Score

A 0–100 health score reflects churn risk, contract status, and revenue engagement.

### Retention Recommendation Engine

The system generates personalized retention strategies such as:

- Contract upgrade offers
- Discount and loyalty incentives
- Tech support promotions
- Security package recommendations

These recommendations are designed to translate predictive insight into immediate action.

## Dashboard Features

The interface is built for business users, analysts, and retention teams.

### Executive Dashboard

Displays:

- Total customers
- Churn risk distribution
- Revenue risk exposure
- KPI metrics for retention and growth

### Customer Prediction

Allows manual customer entry and returns:

- Churn probability
- Risk segment
- Revenue risk
- Health score
- Personalized recommendations

### Explainable

Displays SHAP explanations for customer-level decisions and supports deeper analysis.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Customer_Analysis.py
│   ├── 3_Batch_Prediction.py
│   ├── 4_Model_Performance.py
│   ├── 5_About_Project.py
│   └── customer_prediction.py
├── models/
│   ├── xgboost_tuned.pkl
│   ├── preprocessor.pkl
│   └── threshold.json
├── utils/
│   ├── feature_engineering.py
│   ├── model_loader.py
│   ├── prediction_engine.py
│   ├── recommendation_engine.py
│   ├── risk_calculator.py
│   ├── report_generator.py
│   └── visualizations.py
├── data/
│   ├── telco_churn.csv
│   └── dashboard_data.csv
├── assets/
│   └── logo.png
├── reports/
│   └── dashboard-screenshoots/
├── requirements.txt
├── README.md
└── .venv/
```

## Tech Stack

![Python]
![Pandas]
![NumPy]
![Scikit-Learn]
![XGBoost]
![LightGBM]
![CatBoost]
![SHAP]
![Plotly]
![Streamlit]
![Joblib]
![Matplotlib]
![Seaborn]

## Installation

Follow these steps to install and run the dashboard locally.

```bash
# Clone repository
git clone https://github.com/muhammadahmadcodebase/customer-churn-prediction.git
cd customer-churn-prediction

# Create and activate environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py
```

> macOS / Linux users:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## Usage

1. Train or load the saved model artifacts.
2. Place required model files in `models/`.
3. Launch the Streamlit dashboard.
4. Enter customer profile values in the `Customer Prediction` page.
5. Review churn probability, risk segment, revenue risk, health score and recommendations.
6. Explore SHAP explanations to understand feature impact.

## Screenshots

![Dashboard Screenshots](reports/dashboard-screenshoots/)

## Key Business Insights

- Identifies high-risk customers before they churn
- Quantifies revenue loss exposure from churn
- Prioritizes retention efforts based on customer risk
- Translates predictive insights into actionable recommendations
- Boosts stakeholder confidence with explainable model outputs

## Future Improvements

- Real-time prediction API
- Cloud deployment with containerization
- CRM integration for retention campaigns
- Automated customer outreach workflows
- Customer lifetime revenue forecasting
- Deep learning and sequential modeling
- MLOps pipeline with monitoring and retraining

## Author

**Muhammad Ahmad**

- LinkedIn: [www.linkedin.com/in/ahmadvision](https://www.linkedin.com/in/ahmadvision)
- GitHub: [https://github.com/muhammadahmadcodebase](https://github.com/muhammadahmadcodebase)
- Email: [muhammadahmadcodebase@gmail.com]

## License

Copyright © 2026 Muhammad Ahmad

All Rights Reserved.

This project is provided for viewing and educational reference only. No permission is granted to copy, modify, distribute, or reuse any part of the source code without explicit written permission from the author.

For permission requests, contact: [muhammadahmadcodebase@gmail.com]
