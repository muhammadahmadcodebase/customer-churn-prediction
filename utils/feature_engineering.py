import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features required by the trained customer churn preprocessor.
    Ensures consistency between training and inference time.
    """
    df_eng = df.copy()

    # 1. CLV (Customer Lifetime Value)
    # CLV = MonthlyCharges * tenure
    monthly_charges = pd.to_numeric(df_eng.get('MonthlyCharges', 0.0), errors='coerce').fillna(0.0)
    tenure = pd.to_numeric(df_eng.get('tenure', 0), errors='coerce').fillna(0).astype(int)
    df_eng['CLV'] = monthly_charges * tenure

    # 2. AvgRevenuePerMonth
    # AvgRevenuePerMonth = TotalCharges / (tenure + 1)
    total_charges = pd.to_numeric(df_eng.get('TotalCharges', 0.0), errors='coerce').fillna(0.0)
    df_eng['AvgRevenuePerMonth'] = total_charges / (tenure + 1)

    # 3. TotalServices
    # Count how many service columns contain "Yes" (case-insensitive, stripped)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available_cols = [col for col in service_cols if col in df_eng.columns]
    if available_cols:
        df_eng['TotalServices'] = df_eng[available_cols].apply(
            lambda row: sum(str(val).strip().lower() == 'yes' for val in row), axis=1
        )
    else:
        df_eng['TotalServices'] = 0

    # 4. AutoPay
    # 1 if PaymentMethod contains the word "automatic" (case-insensitive), 0 otherwise
    if 'PaymentMethod' in df_eng.columns:
        df_eng['AutoPay'] = df_eng['PaymentMethod'].apply(
            lambda val: 1 if 'automatic' in str(val).lower() else 0
        )
    else:
        df_eng['AutoPay'] = 0

    # 5. ContractRiskScore
    # Month-to-month -> 3, One year -> 2, Two year -> 1
    contract_mapping = {
        "month-to-month": 3,
        "one year": 2,
        "two year": 1
    }
    if 'Contract' in df_eng.columns:
        df_eng['ContractRiskScore'] = df_eng['Contract'].apply(
            lambda val: contract_mapping.get(str(val).strip().lower(), 1)
        )
    else:
        df_eng['ContractRiskScore'] = 1

    # 6. TenureGroup
    # 0-12 months -> New, 13-24 months -> Developing, 25-48 months -> Established, 49+ months -> Loyal
    bins_tenure = [-1, 12, 24, 48, float('inf')]
    labels_tenure = ["New", "Developing", "Established", "Loyal"]
    df_eng['TenureGroup'] = pd.cut(tenure, bins=bins_tenure, labels=labels_tenure).astype(str)

    # 7. MonthlySpendCategory
    # Low, Medium, High, Premium (4 bins using quantiles derived from training data)
    bins_spend = [-float('inf'), 35.5, 70.35, 89.85, float('inf')]
    labels_spend = ['Low', 'Medium', 'High', 'Premium']
    df_eng['MonthlySpendCategory'] = pd.cut(monthly_charges, bins=bins_spend, labels=labels_spend).astype(str)

    # 8. RevenueSegment
    # LowValue, MidValue, HighValue, VIP (4 bins using quantiles derived from training data)
    bins_rev = [-float('inf'), 402.225, 1397.475, 3786.6, float('inf')]
    labels_rev = ['LowValue', 'MidValue', 'HighValue', 'VIP']
    df_eng['RevenueSegment'] = pd.cut(total_charges, bins=bins_rev, labels=labels_rev).astype(str)

    # 9. HighRiskCustomer
    # Contract is Month-to-month and tenure < 12
    if 'Contract' in df_eng.columns:
        contract_is_m2m = df_eng['Contract'].astype(str).str.strip().str.lower() == 'month-to-month'
        df_eng['HighRiskCustomer'] = np.where(contract_is_m2m & (tenure < 12), 1, 0)
    else:
        df_eng['HighRiskCustomer'] = 0

    return df_eng


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the exact engineered features used during training.
    This wraps the existing inference-time feature engineering logic.
    """
    return engineer_features(df)
