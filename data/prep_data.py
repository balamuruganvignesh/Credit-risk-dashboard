"""
prep_data.py
Cleans the real credit_risk_dataset.csv and engineers PD, LGD, EAD
so it feeds directly into the EL model.
"""
import pandas as pd
import numpy as np
import os

# Paths relative to this script's location
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

df = pd.read_csv(os.path.join(DATA_DIR, 'credit_risk_dataset.csv'))

# ── 1. Clean ──────────────────────────────────────────────────────────────────
# Cap age outliers (144, 123 etc. are clearly data errors)
df['person_age'] = df['person_age'].clip(upper=85)

# Impute loan_int_rate nulls with grade median
grade_median_rate = df.groupby('loan_grade')['loan_int_rate'].median()
df['loan_int_rate'] = df.apply(
    lambda r: grade_median_rate[r['loan_grade']] if pd.isna(r['loan_int_rate']) else r['loan_int_rate'],
    axis=1
)

# Impute person_emp_length nulls with median
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())

# Encode prior default history
df['prior_default'] = (df['cb_person_default_on_file'] == 'Y').astype(int)

# ── 2. Engineer EAD ───────────────────────────────────────────────────────────
# EAD = loan amount (full exposure)
df['ead'] = df['loan_amnt'].astype(float)

# ── 3. Engineer PD ───────────────────────────────────────────────────────────
# Base PD calibrated directly from actual observed default rates by grade
observed_pd = {
    'A': 0.0996, 'B': 0.1628, 'C': 0.2073,
    'D': 0.5905, 'E': 0.6442, 'F': 0.7054, 'G': 0.9844
}

# Adjustments:
#   +8pp if prior default on file
#   scaled by loan_percent_income (DTI signal): each 10pp above 20% adds ~2pp
df['pd_base'] = df['loan_grade'].map(observed_pd)
df['pd_dti_adj'] = np.clip((df['loan_percent_income'] - 0.20) * 0.20, -0.02, 0.08)
df['pd_prior_adj'] = df['prior_default'] * 0.08
df['pd'] = np.clip(
    df['pd_base'] + df['pd_dti_adj'] + df['pd_prior_adj'] + np.random.normal(0, 0.005, len(df)),
    0.001, 0.999
).round(6)

# ── 4. Engineer LGD ──────────────────────────────────────────────────────────
# Grade base + interest rate signal (higher rate → riskier → higher loss)
lgd_grade_base = {
    'A': 0.20, 'B': 0.28, 'C': 0.38,
    'D': 0.50, 'E': 0.60, 'F': 0.68, 'G': 0.75
}
df['lgd_base'] = df['loan_grade'].map(lgd_grade_base)
# Int rate signal: normalised 0–1 relative to range 5–25%
df['lgd_rate_adj'] = ((df['loan_int_rate'] - 5) / 20).clip(0, 1) * 0.08
df['lgd'] = np.clip(
    df['lgd_base'] + df['lgd_rate_adj'] + np.random.normal(0, 0.015, len(df)),
    0.05, 0.95
).round(6)

# ── 5. Add synthetic origination dates (dataset has none) ────────────────────
start, end = pd.Timestamp('2021-01-01'), pd.Timestamp('2024-06-30')
days_range = (end - start).days
df['origination_date'] = [
    start + pd.Timedelta(days=int(d))
    for d in np.random.randint(0, days_range, len(df))
]

# ── 6. Rename for consistency with dashboard ──────────────────────────────────
df = df.rename(columns={
    'loan_intent':             'product_type',
    'person_home_ownership':   'region',
    'loan_grade':              'grade',
    'loan_amnt':               'loan_amnt_orig',
})

# Add loan_id
df.insert(0, 'loan_id', [f'LN{str(i).zfill(6)}' for i in range(1, len(df)+1)])

# ── 7. Save ───────────────────────────────────────────────────────────────────
keep = [
    'loan_id', 'origination_date', 'grade', 'region', 'product_type',
    'ead', 'pd', 'lgd',
    'loan_int_rate', 'loan_status', 'loan_percent_income',
    'person_age', 'person_income', 'person_emp_length',
    'cb_person_cred_hist_length', 'prior_default',
]
df[keep].to_csv(os.path.join(DATA_DIR, 'loan_portfolio.csv'), index=False)

print(f"✅ Cleaned {len(df):,} loans")
print(f"   Grades:   {sorted(df['grade'].unique())}")
print(f"   Products: {sorted(df['product_type'].unique())}")
print(f"   Regions:  {sorted(df['region'].unique())}")
print(f"   Nulls remaining: {df[keep].isnull().sum().sum()}")
print()
print("PD stats by grade:")
print(df.groupby('grade')['pd'].agg(['mean','min','max']).round(4))
print()
print("LGD stats by grade:")
print(df.groupby('grade')['lgd'].agg(['mean','min','max']).round(4))
