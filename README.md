# Credit Risk Model — Probability of Default

A full end-to-end credit risk modelling project built in Python, using the UCI German Credit Dataset. The model predicts the **Probability of Default (PD)** for individual loan applicants and calculates **Expected Loss** using the Basel II formula.

---

## Live Dashboard

> **[Launch Streamlit App →]()**


---

## Project Objectives

- Build a binary classification model to predict loan default (good vs bad)
- Achieve an AUC-ROC score above **80%**
- Calculate Expected Loss per borrower: **EL = PD × LGD × EAD**
- Segment borrowers into risk bands (Very Low → Very High)
- Deploy as an interactive dashboard for credit officers

---


##  Methodology

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

## Key Results

| Metric | Value |
|--------|-------|
| Final Model AUC | **0.8101**  |
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



## Business Applications

This model architecture is directly applicable to:
- **Kenyan fintechs** — Tala, Branch, M-KOPA, Lipa Later
- **Commercial banks** — Equity Bank, NCBA, KCB, Co-operative Bank
- **SACCOs** — automated loan scoring at member intake
- **Microfinance institutions** — portfolio risk monitoring

---

## Skills Demonstrated

`Python` `pandas` `scikit-learn` `XGBoost` `Streamlit` `Logistic Regression`
`Credit Risk` `Basel II` `Expected Loss` `AUC-ROC` `Feature Engineering`
`EDA` `Data Visualisation` `Machine Learning` `Financial Modelling`

---

## Author

**[Raphael Muthuri Nteere]**
- LinkedIn: [linkedin.com/in/raphael-nteere-3b2284274]
- Email: [nteerejoshua9@gmail.com]

---


