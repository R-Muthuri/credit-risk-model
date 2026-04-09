# 🏦 Credit Risk Model — Probability of Default

A full end-to-end credit risk modelling project built in Python, using the UCI German Credit Dataset. The model predicts the **Probability of Default (PD)** for individual loan applicants and calculates **Expected Loss** using the Basel II formula.

---

## 📊 Live Dashboard

> **[Launch Streamlit App →](https://your-app-name.streamlit.app)**
> *(Replace this link after deploying to Streamlit Community Cloud)*

---

## 🎯 Project Objectives

- Build a binary classification model to predict loan default (good vs bad)
- Achieve an AUC-ROC score above **80%**
- Calculate Expected Loss per borrower: **EL = PD × LGD × EAD**
- Segment borrowers into risk bands (Very Low → Very High)
- Deploy as an interactive dashboard for credit officers

---

## 📁 Project Structure

```
credit-risk-model/
│
├── app.py                  ← Streamlit dashboard
├── model.py                ← Preprocessing and model logic
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── data/
│   └── german_credit_data.csv
│
├── outputs/
│   ├── eda_3_1_target.png
│   ├── eda_3_2_numeric.png
│   ├── eda_3_3_categorical.png
│   ├── eda_3_4_boxplots.png
│   ├── eval_7_1_roc.png
│   ├── eval_7_2_confusion.png
│   ├── eval_7_3_expected_loss.png
│   └── eval_8_feature_importance.png
│
└── notebook/
    └── credit_risk_model.ipynb
```

---

## 🔬 Methodology

### Dataset
- **Source:** UCI Statlog German Credit Dataset
- **Size:** 1,000 loan applications × 20 features
- **Target:** Binary — good (repaid) vs bad (default)
- **Class split:** 70% good, 30% default

### Preprocessing
| Step | Technique |
|------|-----------|
| Target encoding | good → 0, bad → 1 |
| Ordinal encoding | status_account, status_savings, years_employment, credit_history |
| Binary encoding | telephone, is_foreign_worker |
| One-hot encoding | purpose, housing, job, collateral, and others |
| Feature scaling | StandardScaler on all numeric columns |

### Models Trained
| Model | Test AUC | Mean CV AUC |
|-------|----------|-------------|
| **Logistic Regression (C=0.1)** | **0.8101** | **0.787** |
| Random Forest | 0.8090 | 0.781 |
| XGBoost (tuned) | 0.7980 | 0.784 |

### Expected Loss Formula (Basel II)
```
EL = PD × LGD × EAD

PD  = Probability of Default    (model output, 0–1)
LGD = Loss Given Default        (45% — Basel II unsecured retail)
EAD = Exposure at Default       (loan amount in DM)
```

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Final Model AUC | **0.8101** ✅ |
| Portfolio Exposure (EAD) | DM 861,392 |
| Total Expected Loss | DM 144,766 |
| Portfolio EL Rate | 16.81% |
| Average PD | 30.98% |

### Risk Band Summary
| Band | Borrowers | Avg PD | Total EL |
|------|-----------|--------|----------|
| Very Low | 102 | 10.5% | DM 11,362 |
| Low | 73 | 28.4% | DM 34,834 |
| Medium | 38 | 50.6% | DM 32,070 |
| High | 29 | 68.8% | DM 38,496 |
| Very High | 8 | 85.1% | DM 28,004 |

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit dashboard
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 🌐 Deploying to Streamlit Community Cloud (Free)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repository, branch (`main`), and set main file to `app.py`
5. Click **Deploy** — live in ~2 minutes

---

## 🏦 Business Applications

This model architecture is directly applicable to:
- **Kenyan fintechs** — Tala, Branch, M-KOPA, Lipa Later
- **Commercial banks** — Equity Bank, NCBA, KCB, Co-operative Bank
- **SACCOs** — automated loan scoring at member intake
- **Microfinance institutions** — portfolio risk monitoring

---

## 📚 Skills Demonstrated

`Python` `pandas` `scikit-learn` `XGBoost` `Streamlit` `Logistic Regression`
`Credit Risk` `Basel II` `Expected Loss` `AUC-ROC` `Feature Engineering`
`EDA` `Data Visualisation` `Machine Learning` `Financial Modelling`

---

## 👤 Author

**[Your Name]**
- LinkedIn: [your-linkedin]
- Email: [your-email]

---

## 📄 License

MIT License — free to use and adapt with attribution.
