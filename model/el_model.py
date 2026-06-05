"""
el_model.py
Runs EL = PD × LGD × EAD on the cleaned loan portfolio,
builds portfolio/segment aggregations, runs stress scenarios,
and stores everything in DuckDB + CSV.
"""
import pandas as pd
import numpy as np
import duckdb, json, os

# Paths relative to project root (one level up from model/)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(
    os.path.join(DATA_DIR, 'loan_portfolio.csv'),
    parse_dates=['origination_date']
)

# ── 2. Baseline EL ────────────────────────────────────────────────────────────
df['el']      = (df['pd'] * df['lgd'] * df['ead']).round(4)
df['el_rate'] = (df['pd'] * df['lgd']).round(6)
df['quarter'] = df['origination_date'].dt.to_period('Q').astype(str)
df['year']    = df['origination_date'].dt.year

total_ead  = df['ead'].sum()
total_el   = df['el'].sum()
wtd_pd     = (df['pd']  * df['ead']).sum() / total_ead
wtd_lgd    = (df['lgd'] * df['ead']).sum() / total_ead
el_cov     = total_el / total_ead

portfolio_summary = {
    'total_ead':       round(total_ead, 2),
    'total_el':        round(total_el,  2),
    'wtd_avg_pd':      round(wtd_pd,  6),
    'wtd_avg_lgd':     round(wtd_lgd, 6),
    'el_coverage_pct': round(el_cov * 100, 4),
    'loan_count':      len(df),
    'actual_default_rate': round(df['loan_status'].mean(), 4),
}

# ── 3. Segment aggregations ───────────────────────────────────────────────────
def segment_agg(df, col):
    g = df.groupby(col).agg(
        loan_count  = ('loan_id', 'count'),
        total_ead   = ('ead',     'sum'),
        total_el    = ('el',      'sum'),
        actual_defaults = ('loan_status', 'sum'),
        wtd_pd  = ('pd',  lambda x: np.average(x, weights=df.loc[x.index, 'ead'])),
        wtd_lgd = ('lgd', lambda x: np.average(x, weights=df.loc[x.index, 'ead'])),
    ).reset_index()
    g['el_rate_pct']   = (g['total_el']  / g['total_ead'] * 100).round(4)
    g['ead_share_pct'] = (g['total_ead'] / total_ead       * 100).round(2)
    g['default_rate']  = (g['actual_defaults'] / g['loan_count'] * 100).round(2)
    return g.sort_values('total_el', ascending=False)

by_grade   = segment_agg(df, 'grade')
by_region  = segment_agg(df, 'region')
by_product = segment_agg(df, 'product_type')

by_quarter = df.groupby('quarter').agg(
    loan_count = ('loan_id',     'count'),
    total_ead  = ('ead',         'sum'),
    total_el   = ('el',          'sum'),
    defaults   = ('loan_status', 'sum'),
).reset_index().sort_values('quarter')
by_quarter['default_rate'] = (by_quarter['defaults'] / by_quarter['loan_count'] * 100).round(2)

# ── 4. Stress scenario engine ─────────────────────────────────────────────────
SCENARIOS = {
    'Baseline': {'pd_mult': 1.00, 'lgd_shock': 0.00},
    'Adverse':  {'pd_mult': 1.50, 'lgd_shock': 0.05},
    'Severe':   {'pd_mult': 2.25, 'lgd_shock': 0.12},
    'Extreme':  {'pd_mult': 3.50, 'lgd_shock': 0.20},
}

scenario_summary = {}
for name, p in SCENARIOS.items():
    pd_s  = np.clip(df['pd']  * p['pd_mult'],   0.001, 0.999)
    lgd_s = np.clip(df['lgd'] + p['lgd_shock'], 0.001, 0.999)
    el_s  = (pd_s * lgd_s * df['ead']).sum()
    scenario_summary[name] = {
        'total_el':    round(el_s, 2),
        'el_coverage': round(el_s / total_ead * 100, 4),
        'el_delta':    round((el_s - total_el) / total_el * 100, 2),
    }

# ── 5. Store in DuckDB ────────────────────────────────────────────────────────
con = duckdb.connect(os.path.join(DATA_DIR, 'portfolio.duckdb'))
for tbl in ['loans','by_grade','by_region','by_product','by_quarter']:
    con.execute(f"DROP TABLE IF EXISTS {tbl}")

con.register('loans_df',      df)
con.register('by_grade_df',   by_grade)
con.register('by_region_df',  by_region)
con.register('by_product_df', by_product)
con.register('by_quarter_df', by_quarter)
con.execute("CREATE TABLE loans      AS SELECT * FROM loans_df")
con.execute("CREATE TABLE by_grade   AS SELECT * FROM by_grade_df")
con.execute("CREATE TABLE by_region  AS SELECT * FROM by_region_df")
con.execute("CREATE TABLE by_product AS SELECT * FROM by_product_df")
con.execute("CREATE TABLE by_quarter AS SELECT * FROM by_quarter_df")
con.close()

# ── 6. Save CSVs + JSON ───────────────────────────────────────────────────────
df.to_csv(        os.path.join(DATA_DIR, 'loans_clean.csv'),  index=False)
by_grade.to_csv(  os.path.join(DATA_DIR, 'by_grade.csv'),     index=False)
by_region.to_csv( os.path.join(DATA_DIR, 'by_region.csv'),    index=False)
by_product.to_csv(os.path.join(DATA_DIR, 'by_product.csv'),   index=False)
by_quarter.to_csv(os.path.join(DATA_DIR, 'by_quarter.csv'),   index=False)

with open(os.path.join(DATA_DIR, 'portfolio_summary.json'),'w') as f:
    json.dump(portfolio_summary, f)
with open(os.path.join(DATA_DIR, 'scenario_results.json'),'w') as f:
    json.dump(scenario_summary, f)

# ── 7. Print summary ──────────────────────────────────────────────────────────
print("✅ EL model complete")
print(f"Portfolio: ${total_ead/1e9:.2f}B EAD | ${total_el/1e6:.1f}M EL | {el_cov*100:.2f}% coverage")
print(f"Actual default rate in dataset: {df['loan_status'].mean()*100:.1f}%")
print()
print("Scenario comparison:")
for name, r in scenario_summary.items():
    print(f"  {name:10s}: EL=${r['total_el']/1e6:.1f}M  coverage={r['el_coverage']:.2f}%  delta={r['el_delta']:+.1f}%")
print()
print("EL by grade:")
print(by_grade[['grade','loan_count','total_ead','total_el','el_rate_pct','default_rate']].to_string(index=False))
