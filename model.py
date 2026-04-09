"""
model.py
--------
Reusable credit risk model — preprocessing, training, and prediction.
Import this module in app.py or the notebook.
"""

import pandas as pd
import numpy as np
import warnings
import os

# Absolute path to the data file — works locally and on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'german_credit_data.csv')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ── Ordinal mappings (must stay identical across training and prediction)
ORDINAL_MAPPINGS = {
    'status_account': [
        '< 0 DM', '0 to < 200 DM', '>= 200 DM', 'no checking account'
    ],
    'status_savings': [
        '< 100 DM', '100 to < 500 DM', '500 to < 1000 DM',
        '>= 1000 DM', 'unknown/ no savings account'
    ],
    'years_employment': [
        'unemployed', '< 1 year', '1 to < 4 years',
        '4 to < 7 years', '>= 7 years'
    ],
    'credit_history': [
        'critical account/ other credits existing (not at this bank)',
        'delay in paying off in the past',
        'existing credits paid back duly till now',
        'all credits at this bank paid back duly',
        'no credits taken/ all credits paid back duly'
    ]
}

OHE_COLS = [
    'purpose', 'status_and_sex', 'secondary_obligor',
    'collateral', 'other_installment_plans', 'housing', 'job'
]

NUMERIC_COLS = [
    'age', 'credit_amount', 'month_duration',
    'payment_to_income_ratio', 'residence_since',
    'n_credits', 'n_guarantors'
]

LGD = 0.45   # Basel II unsecured retail standard


def build_model(data_path=None):
    if data_path is None:
        data_path = DATA_PATH
    df = pd.read_csv(data_path)
    df['target'] = df['target'].map({'good': 0, 'bad': 1})

    # Ordinal encoding
    for col, order in ORDINAL_MAPPINGS.items():
        df[col] = df[col].map({val: i for i, val in enumerate(order)})

    # Binary encoding
    df['telephone'] = df['telephone'].map(
        {'yes, registered under the customers name': 1, 'none': 0})
    df['is_foreign_worker'] = df['is_foreign_worker'].map(
        {'yes': 1, 'no': 0})

    # One-hot encoding
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    lr.fit(X_train, y_train)

    return lr, scaler, X_train, X_test, y_test


def preprocess_client(client_dict, X_train_columns, scaler):
    """
    Apply same preprocessing pipeline to a new client dictionary.
    Returns: preprocessed DataFrame ready for prediction.
    """
    df_c = pd.DataFrame([client_dict])

    # Ordinal encoding
    for col, order in ORDINAL_MAPPINGS.items():
        mapping = {val: i for i, val in enumerate(order)}
        df_c[col] = df_c[col].map(mapping)

    # Binary encoding
    df_c['telephone'] = df_c['telephone'].map(
        {'yes, registered under the customers name': 1, 'none': 0})
    df_c['is_foreign_worker'] = df_c['is_foreign_worker'].map(
        {'yes': 1, 'no': 0})

    # One-hot encoding
    df_c = pd.get_dummies(df_c, columns=OHE_COLS)

    # Align columns
    df_c = df_c.reindex(columns=X_train_columns, fill_value=0)

    # Scale
    df_c[NUMERIC_COLS] = scaler.transform(df_c[NUMERIC_COLS])

    return df_c


def assess_client(client_dict, lr, scaler, X_train):
    """
    Full assessment pipeline for one client.
    Returns dict with PD, band, EL, decision.
    """
    df_c     = preprocess_client(client_dict, X_train.columns, scaler)
    PD       = lr.predict_proba(df_c)[0][1]
    EAD      = client_dict['credit_amount']
    EL       = PD * LGD * EAD

    if   PD <= 0.20: band = "Very Low"
    elif PD <= 0.40: band = "Low"
    elif PD <= 0.60: band = "Medium"
    elif PD <= 0.80: band = "High"
    else:            band = "Very High"

    if   PD < 0.35: decision = "APPROVE"
    elif PD < 0.60: decision = "REFER FOR REVIEW"
    else:           decision = "DECLINE"

    return {
        "PD"       : round(PD * 100, 2),
        "band"     : band,
        "EAD"      : EAD,
        "LGD"      : LGD * 100,
        "EL"       : round(EL, 2),
        "decision" : decision
    }
