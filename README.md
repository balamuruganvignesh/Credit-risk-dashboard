# Credit Risk · Expected Loss Dashboard

A Python/SQL credit risk tool built on a real 32,581-loan portfolio dataset.
Calculates **EL = PD × LGD × EAD** at loan level, aggregates to portfolio insights,
and enables interactive stress scenario analysis.

## Dataset
- **Source:** Kaggle credit risk dataset (32,581 loans, 12 features)
- **Key fields:** loan amount, grade (A–G), loan intent, home ownership,
  default status, interest rate, debt-to-income ratio
- **EL model fields engineered from:** observed grade default rates (PD),
  grade + interest rate signal (LGD), loan amount (EAD)


## Project structure

```
credit_risk/
├── data/
│   ├── credit_risk_dataset.csv   ← raw Kaggle data
│   ├── prep_data.py              ← cleaning + PD/LGD/EAD engineering
│   ├── loans_clean.csv           ← cleaned loan-level data
│   ├── by_grade/region/product/quarter.csv
│   ├── portfolio_summary.json
│   └── portfolio.duckdb
├── model/
│   └── el_model.py               ← EL model + stress engine
├── dashboard/
│   └── app.py                    ← Streamlit dashboard
└── requirements.txt
```

## Portfolio summary (baseline)

| Metric | Value |
|--------|-------|
| Total EAD | ~$310M |
| Baseline EL | ~$35.4M |
| EL Coverage | 11.3% |
| Actual Default Rate | 21.8% |
| Loan Count | 32,581 |

## Stress scenarios

| Scenario | PD Mult | LGD Shock | EL Delta |
|----------|---------|-----------|----------|
| Baseline | 1.0x    | +0pp      | —        |
| Adverse  | 1.5x    | +5pp      | ~+62%    |
| Severe   | 2.25x   | +12pp     | ~+128%   |
| Extreme  | 3.5x    | +20pp     | ~+237%   |

## Dashboard features

- **KPI row** — EAD, baseline EL, stressed EL (live), EL coverage, actual default rate
- **Sidebar filters** — grade, home ownership, loan intent, origination date range
- **Stress controls** — preset scenarios or fully custom PD multiplier + LGD shock sliders
- **Charts** — EL by grade, scenario comparison, origination time series, model validation
  (modeled PD vs actual default rate), concentration heatmap, EL by loan intent,
  EL by home ownership, EL rate violin by grade
- **Loan-level table** — sortable by any metric, with DTI and interest rate columns
- **CSV export** — download the full stressed portfolio

