"""
app.py
------
Streamlit dashboard for the Probability of Default Credit Risk Model.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from model import (build_model, assess_client, ORDINAL_MAPPINGS, LGD)

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Credit Risk Model",
    page_icon="🏦",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
#  LOAD AND CACHE MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return build_model()

lr, scaler, X_train, X_test, y_test = load_model()
y_prob = lr.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png", width=60)
st.sidebar.title("Credit Risk Model")
st.sidebar.markdown("**Probability of Default**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [" Model Overview",
     " Predict New Client",
     " EDA & Visuals",
     " Portfolio Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.metric("AUC Score",     f"{auc:.4f}")
st.sidebar.metric("Training Rows", "750")
st.sidebar.metric("Test Rows",     "250")
st.sidebar.metric("Features",      "37")


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — MODEL OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == " Model Overview":

    st.title(" Credit Risk — Probability of Default Model")
    st.markdown("Built on the **UCI German Credit Dataset** · 1,000 loan applications · Logistic Regression (C=0.1)")
    st.markdown("---")

    # ── KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC Score",        f"{auc:.4f}",  "Above 80% target ✅")
    col2.metric("Dataset Size",     "1,000 rows",  "No missing values")
    col3.metric("Default Rate",     "30%",         "Class imbalance noted")
    col4.metric("LGD Assumption",   "45%",         "Basel II standard")

    st.markdown("---")

    # ── ROC Curve + Confusion Matrix side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 4))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color='#2980b9', lw=2,
                label=f'AUC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.08, color='#2980b9')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve — Logistic Regression')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Confusion Matrix")
        y_pred = (y_prob >= 0.5).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Good', 'Default'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Confusion Matrix (threshold = 0.50)')
        ax.grid(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Feature Importance
    st.subheader("Top 15 Feature Importances")
    feat_imp = pd.DataFrame({
        'Feature'    : X_train.columns,
        'Coefficient': lr.coef_[0]
    })
    feat_imp['Abs'] = feat_imp['Coefficient'].abs()
    feat_imp = feat_imp.nlargest(15, 'Abs').sort_values('Coefficient')

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ['#e74c3c' if c > 0 else '#2ecc71' for c in feat_imp['Coefficient']]
    ax.barh(feat_imp['Feature'], feat_imp['Coefficient'],
            color=colors, edgecolor='white', height=0.65)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title('Red = increases default risk  |  Green = reduces default risk')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT NEW CLIENT
# ══════════════════════════════════════════════════════════════
elif page == " Predict New Client":

    st.title(" New Client Credit Assessment")
    st.markdown("Fill in the applicant details below and click **Run Assessment**.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account & Credit")
        status_account = st.selectbox(
            "Checking Account Status",
            ['< 0 DM', '0 to < 200 DM', '>= 200 DM', 'no checking account'])
        status_savings = st.selectbox(
            "Savings Account Status",
            ['< 100 DM', '100 to < 500 DM', '500 to < 1000 DM',
             '>= 1000 DM', 'unknown/ no savings account'])
        credit_amount = st.number_input(
            "Loan Amount (DM)", min_value=250, max_value=20000,
            value=5000, step=250)
        month_duration = st.slider(
            "Loan Duration (months)", 6, 72, 24)
        credit_history = st.selectbox(
            "Credit History",
            ['critical account/ other credits existing (not at this bank)',
             'delay in paying off in the past',
             'existing credits paid back duly till now',
             'all credits at this bank paid back duly',
             'no credits taken/ all credits paid back duly'])
        purpose = st.selectbox(
            "Loan Purpose",
            ['radio/television', 'education', 'furniture/equipment',
             'car (new)', 'car (used)', 'business',
             'domestic appliances', 'repairs', 'others', 'retraining'])

    with col2:
        st.subheader("Personal Details")
        age = st.slider("Age", 18, 80, 32)
        status_and_sex = st.selectbox(
            "Personal Status & Sex",
            ['male : single', 'male : married/widowed',
             'male : divorced/separated',
             'female : divorced/separated/married'])
        years_employment = st.selectbox(
            "Years Employed",
            ['unemployed', '< 1 year', '1 to < 4 years',
             '4 to < 7 years', '>= 7 years'])
        housing = st.selectbox(
            "Housing", ['own', 'rent', 'for free'])
        is_foreign_worker = st.selectbox(
            "Foreign Worker", ['yes', 'no'])
        telephone = st.selectbox(
            "Telephone Registered",
            ['yes, registered under the customers name', 'none'])

    with col3:
        st.subheader("Financial Details")
        payment_to_income_ratio = st.slider(
            "Payment to Income Ratio", 1, 4, 2)
        secondary_obligor = st.selectbox(
            "Secondary Obligor / Guarantor",
            ['none', 'guarantor', 'co-applicant'])
        collateral = st.selectbox(
            "Collateral",
            ['none', 'car', 'real estate',
             'savings agreement/life insurance'])
        other_installment_plans = st.selectbox(
            "Other Installment Plans", ['none', 'bank', 'stores'])
        residence_since = st.slider("Residence Since (years)", 1, 4, 2)
        n_credits = st.slider("Number of Existing Credits", 1, 4, 1)
        n_guarantors = st.slider("Number of Guarantors", 1, 2, 1)
        job = st.selectbox(
            "Job Type",
            ['skilled employee/ official',
             'unskilled - resident',
             'management/ self-employed/highly qualified employee',
             'unemployed/ unskilled - non-resident'])

    st.markdown("---")

    if st.button(" Run Credit Assessment", use_container_width=True):

        client = {
            'status_account'          : status_account,
            'month_duration'          : month_duration,
            'credit_history'          : credit_history,
            'purpose'                 : purpose,
            'credit_amount'           : credit_amount,
            'status_savings'          : status_savings,
            'years_employment'        : years_employment,
            'payment_to_income_ratio' : payment_to_income_ratio,
            'status_and_sex'          : status_and_sex,
            'secondary_obligor'       : secondary_obligor,
            'residence_since'         : residence_since,
            'collateral'              : collateral,
            'age'                     : age,
            'other_installment_plans' : other_installment_plans,
            'housing'                 : housing,
            'n_credits'               : n_credits,
            'job'                     : job,
            'n_guarantors'            : n_guarantors,
            'telephone'               : telephone,
            'is_foreign_worker'       : is_foreign_worker
        }

        result = assess_client(client, lr, scaler, X_train)

        # ── Decision colour
        if result['decision'] == "APPROVE":
            colour = "🟢"
            box_colour = "normal"
        elif result['decision'] == "REFER FOR REVIEW":
            colour = "🟡"
            box_colour = "warning"
        else:
            colour = "🔴"
            box_colour = "error"

        st.markdown("### Assessment Result")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Probability of Default", f"{result['PD']}%")
        c2.metric("Risk Band",              result['band'])
        c3.metric("Loan Amount (EAD)",      f"DM {result['EAD']:,.0f}")
        c4.metric("Expected Loss",          f"DM {result['EL']:,.2f}")
        c5.metric("Decision",               f"{colour} {result['decision']}")

        if result['decision'] == "APPROVE":
            st.success(f"✅ **APPROVED** — PD of {result['PD']}% is below the 35% threshold. Expected Loss is DM {result['EL']:,.2f}.")
        elif result['decision'] == "REFER FOR REVIEW":
            st.warning(f"⚠️ **REFER FOR MANUAL REVIEW** — PD of {result['PD']}% is between 35% and 60%. A credit officer should review this application.")
        else:
            st.error(f"❌ **DECLINED** — PD of {result['PD']}% exceeds the 60% threshold. Expected Loss of DM {result['EL']:,.2f} is too high.")

        # ── PD Gauge
        st.markdown("---")
        st.markdown("**PD Score Gauge**")
        gauge_col, _ = st.columns([2, 1])
        with gauge_col:
            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.barh(0, 100, color='#f0f0f0', height=0.5)
            ax.barh(0, result['PD'], color=(
                '#2ecc71' if result['PD'] < 35 else
                '#f39c12' if result['PD'] < 60 else '#e74c3c'),
                height=0.5)
            ax.axvline(35, color='#f39c12', lw=1.5, linestyle='--')
            ax.axvline(60, color='#e74c3c', lw=1.5, linestyle='--')
            ax.text(35, 0.35, 'Refer', ha='center', fontsize=8, color='#f39c12')
            ax.text(60, 0.35, 'Decline', ha='center', fontsize=8, color='#e74c3c')
            ax.text(result['PD'], -0.35,
                    f"{result['PD']}%", ha='center', fontsize=9, fontweight='bold')
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel('Probability of Default (%)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — EDA & VISUALS
# ══════════════════════════════════════════════════════════════
    st.title(" Exploratory Data Analysis")
    st.markdown("Visual summary of the German Credit Dataset.")
    st.markdown("---")

    df_eda = pd.read_csv(DATA_PATH)

    # ── 1. Target Distribution
    st.subheader("1. Target Distribution")
    counts = df_eda['target'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(counts.index, counts.values,
                color=['#2ecc71','#e74c3c'], edgecolor='white', width=0.4)
    axes[0].set_title('Loan Outcome Count', fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].pie(counts.values, labels=counts.index,
                colors=['#2ecc71','#e74c3c'],
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Loan Outcome Proportion', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("700 good borrowers vs 300 defaulters — 70/30 class imbalance")

    st.markdown("---")

    # ── 2. Numeric Distributions
    st.subheader("2. Numeric Feature Distributions")
    num_cols = ['age', 'credit_amount', 'month_duration',
                'payment_to_income_ratio', 'n_credits', 'n_guarantors']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df_eda[col], bins=30, color='#3498db',
                     edgecolor='white', alpha=0.85)
        mean_val = df_eda[col].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--',
                        lw=1.5, label='Mean: ' + str(round(mean_val, 1)))
        axes[i].set_title(col, fontweight='bold')
        axes[i].set_ylabel('Count')
        axes[i].legend(fontsize=8)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    plt.suptitle('Distribution of Numeric Features', fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Right-skewed credit amounts and age distributions typical of microfinance")

    st.markdown("---")

    # ── 3. Default Rate by Category
    st.subheader("3. Default Rate by Categorical Feature")
    cat_cols = ['status_account', 'credit_history',
                'purpose', 'housing', 'years_employment']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        rate = df_eda.groupby(col)['target'].apply(
            lambda x: (x == 'bad').mean() * 100
        ).sort_values(ascending=True)
        colors = ['#e74c3c' if v >= 30 else '#3498db' for v in rate.values]
        axes[i].barh(rate.index, rate.values,
                     color=colors, edgecolor='white', height=0.6)
        axes[i].axvline(30, color='black', linestyle='--',
                        lw=1.2, label='Avg 30%')
        axes[i].set_title('Default Rate by ' + col, fontweight='bold')
        axes[i].set_xlabel('Default Rate (%)')
        axes[i].legend(fontsize=8)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    axes[5].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Red bars exceed the 30% average — these categories carry above-average default risk")

    st.markdown("---")

    # ── 4. Boxplots
    st.subheader("4. Age & Credit Amount vs Default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    good = df_eda[df_eda['target'] == 'good']
    bad  = df_eda[df_eda['target'] == 'bad']
    axes[0].boxplot([good['age'].values, bad['age'].values],
                    labels=['Good', 'Default'],
                    boxprops=dict(color='#2c3e50'),
                    medianprops=dict(color='red', lw=2),
                    patch_artist=True,
                    )
    axes[0].set_title('Age by Loan Outcome', fontweight='bold')
    axes[0].set_ylabel('Age')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].boxplot([good['credit_amount'].values, bad['credit_amount'].values],
                    labels=['Good', 'Default'],
                    boxprops=dict(color='#2c3e50'),
                    medianprops=dict(color='red', lw=2),
                    patch_artist=True)
    axes[1].set_title('Credit Amount by Loan Outcome', fontweight='bold')
    axes[1].set_ylabel('Credit Amount (DM)')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Defaulters tend to be younger and borrow larger amounts on average")

# ══════════════════════════════════════════════════════════════
#  PAGE 4 — PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == " Portfolio Analysis":

    st.title(" Portfolio Expected Loss Analysis")
    st.markdown("Expected Loss calculated on the 250-row test set using **EL = PD × LGD × EAD**")
    st.markdown("---")

    df_raw = pd.read_csv(DATA_PATH)
    EAD    = df_raw.loc[y_test.index, 'credit_amount'].values
    EL     = y_prob * LGD * EAD

    el_df = pd.DataFrame({
        'PD'            : (y_prob * 100).round(2),
        'EAD'           : EAD,
        'Expected_Loss' : EL.round(2),
        'Actual_Default': y_test.values,
        'Risk_Band'     : pd.cut(y_prob,
                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            labels=['Very Low','Low','Medium','High','Very High'])
    })

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Exposure (EAD)",  "DM " + f"{EAD.sum():,.0f}")
    c2.metric("Total Expected Loss",   "DM " + f"{EL.sum():,.0f}")
    c3.metric("Portfolio EL Rate",     str(round(EL.sum()/EAD.sum()*100, 2)) + "%")
    c4.metric("Average PD",            str(round(y_prob.mean()*100, 2)) + "%")

    st.markdown("---")

    # ── EL charts
    st.subheader("Expected Loss Visualisations")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Chart 1 — EL by Risk Band
    band_el = el_df.groupby('Risk_Band', observed=True)['Expected_Loss'].sum()
    colors  = ['#2ecc71','#3498db','#f39c12','#e67e22','#e74c3c']
    bars = axes[0].bar(band_el.index, band_el.values,
                       color=colors, edgecolor='white', width=0.6)
    axes[0].set_title('Total EL by Risk Band', fontweight='bold')
    axes[0].set_xlabel('Risk Band')
    axes[0].set_ylabel('Expected Loss (DM)')
    axes[0].tick_params(axis='x', rotation=30)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 200,
                     'DM ' + f'{bar.get_height():,.0f}',
                     ha='center', fontsize=7, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Chart 2 — EL vs Loan Amount
    scatter_c = ['#e74c3c' if d == 1 else '#2ecc71'
                 for d in el_df['Actual_Default']]
    axes[1].scatter(EAD, EL, c=scatter_c, alpha=0.5, s=25, edgecolors='none')
    axes[1].set_title('Expected Loss vs Loan Amount', fontweight='bold')
    axes[1].set_xlabel('Loan Amount / EAD (DM)')
    axes[1].set_ylabel('Expected Loss (DM)')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Chart 3 — PD Distribution
    axes[2].hist(y_prob[y_test == 0], bins=25, alpha=0.65,
                 color='#2ecc71', label='Good', edgecolor='white')
    axes[2].hist(y_prob[y_test == 1], bins=25, alpha=0.65,
                 color='#e74c3c', label='Default', edgecolor='white')
    axes[2].axvline(0.5, color='black', linestyle='--',
                    lw=1.5, label='Threshold 0.5')
    axes[2].set_title('PD Distribution by Actual Outcome', fontweight='bold')
    axes[2].set_xlabel('Predicted PD')
    axes[2].set_ylabel('Count')
    axes[2].legend(fontsize=9)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ── Risk Band Table
    st.subheader("Risk Band Breakdown")
    summary = el_df.groupby('Risk_Band', observed=True).agg(
        Borrowers       = ('PD',            'count'),
        Avg_PD          = ('PD',            'mean'),
        Total_EAD       = ('EAD',           'sum'),
        Total_EL        = ('Expected_Loss', 'sum'),
        Actual_Defaults = ('Actual_Default','sum')
    ).reset_index()
    summary['Avg_PD']    = summary['Avg_PD'].apply(lambda x: str(round(x, 1)) + '%')
    summary['Total_EAD'] = summary['Total_EAD'].apply(lambda x: 'DM ' + f'{x:,.0f}')
    summary['Total_EL']  = summary['Total_EL'].apply(lambda x: 'DM ' + f'{x:,.0f}')
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Full borrower table
    st.subheader("Full Borrower-Level EL Table")
    st.dataframe(
        el_df.rename(columns={
            'PD'            : 'PD (%)',
            'EAD'           : 'Loan Amount (DM)',
            'Expected_Loss' : 'Expected Loss (DM)',
            'Actual_Default': 'Actual Default',
            'Risk_Band'     : 'Risk Band'
        }),
        use_container_width=True,
        hide_index=True
    )
