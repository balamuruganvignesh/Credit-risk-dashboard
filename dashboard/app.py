import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
C = {
    'bg':      '#0f1117',
    'surface': '#1a1d27',
    'border':  '#2d3045',
    'accent':  '#4f8ef7',
    'green':   '#2ecc8a',
    'amber':   '#f5a623',
    'red':     '#e85d4a',
    'purple':  '#9b6dff',
    'text':    '#e8eaf6',
    'muted':   '#8b90a8',
    'grade':   {
        'A': '#2ecc8a', 'B': '#4f8ef7', 'C': '#9b6dff',
        'D': '#f5a623', 'E': '#f07848', 'F': '#e85d4a', 'G': '#7b1a1a'
    },
}

def rgba(hex_color, alpha=0.33):
    """Convert #rrggbb hex to rgba() string for Plotly compatibility."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'


st.markdown(f"""
<style>
html, body, .stApp {{ background-color:{C['bg']}; color:{C['text']}; font-family:'Inter',sans-serif; }}
[data-testid="stSidebar"] {{ background-color:{C['surface']}; border-right:1px solid {C['border']}; }}
[data-testid="stMetric"] {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px; padding:16px; }}
[data-testid="stMetricLabel"] {{ color:{C['muted']} !important; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
[data-testid="stMetricValue"] {{ color:{C['text']} !important; font-size:1.55rem; font-weight:700; }}
.block-container {{ padding:1.5rem 2rem; }}
div[data-testid="stHorizontalBlock"] > div {{ gap:0.8rem; }}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'data')

@st.cache_data
def load_data():
    df        = pd.read_csv(f'{BASE}/loans_clean.csv', parse_dates=['origination_date'])
    by_grade  = pd.read_csv(f'{BASE}/by_grade.csv')
    by_region = pd.read_csv(f'{BASE}/by_region.csv')
    by_prod   = pd.read_csv(f'{BASE}/by_product.csv')
    by_qtr    = pd.read_csv(f'{BASE}/by_quarter.csv')
    with open(f'{BASE}/portfolio_summary.json') as f:
        summary = json.load(f)
    return df, by_grade, by_region, by_prod, by_qtr, summary

df, by_grade, by_region, by_prod, by_qtr, summary = load_data()

GRADES   = ['A','B','C','D','E','F','G']
INTENTS  = sorted(df['product_type'].unique().tolist())
REGIONS  = sorted(df['region'].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{C['accent']};margin-bottom:4px'>🏦 Credit Risk</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{C['muted']};font-size:12px;margin-top:0'>Expected Loss · Real Portfolio</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📊 Portfolio Filters")
    sel_grades  = st.multiselect("Loan Grade",    GRADES,  default=GRADES)
    sel_regions = st.multiselect("Home Ownership",REGIONS, default=REGIONS)
    sel_intents = st.multiselect("Loan Intent",   INTENTS, default=INTENTS)
    date_range  = st.date_input("Origination Range",
        value=[df['origination_date'].min().date(), df['origination_date'].max().date()])

    st.divider()
    st.markdown("### ⚡ Stress Scenario")
    scenario_mode = st.radio("Mode", ["Preset", "Custom"], horizontal=True)

    if scenario_mode == "Preset":
        preset = st.selectbox("Scenario", ["Baseline","Adverse","Severe","Extreme"])
        presets = {
            "Baseline": (1.00, 0.00),
            "Adverse":  (1.50, 0.05),
            "Severe":   (2.25, 0.12),
            "Extreme":  (3.50, 0.20),
        }
        pd_mult, lgd_shock = presets[preset]
    else:
        preset    = "Custom"
        pd_mult   = st.slider("PD Multiplier",  1.0, 5.0, 1.0, 0.1)
        lgd_shock = st.slider("LGD Shock (+)",  0.0, 0.40, 0.0, 0.01)

    st.divider()
    st.caption(f"32,581 real loans · Kaggle credit risk dataset")

# ── Filter ────────────────────────────────────────────────────────────────────
mask = (
    df['grade'].isin(sel_grades) &
    df['region'].isin(sel_regions) &
    df['product_type'].isin(sel_intents)
)
if len(date_range) == 2:
    mask &= (df['origination_date'].dt.date >= date_range[0]) & \
            (df['origination_date'].dt.date <= date_range[1])
fdf = df[mask].copy()

# ── Stressed EL ───────────────────────────────────────────────────────────────
fdf['pd_s']  = np.clip(fdf['pd']  * pd_mult,   0.001, 0.999)
fdf['lgd_s'] = np.clip(fdf['lgd'] + lgd_shock, 0.001, 0.999)
fdf['el_s']  = fdf['pd_s'] * fdf['lgd_s'] * fdf['ead']

total_ead_f  = fdf['ead'].sum()
baseline_el  = fdf['el'].sum()
stressed_el  = fdf['el_s'].sum()
el_delta_pct = (stressed_el - baseline_el) / baseline_el * 100 if baseline_el > 0 else 0
el_cov_s     = stressed_el / total_ead_f * 100 if total_ead_f > 0 else 0
wtd_pd_s     = (fdf['pd_s'] * fdf['ead']).sum() / total_ead_f if total_ead_f > 0 else 0
actual_dr    = fdf['loan_status'].mean() * 100

# ── Plot layout ───────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=C['text'], family='Inter', size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=C['muted'])),
)
AX = dict(gridcolor=C['border'], zerolinecolor=C['border'])

def ax(fig, x=None, y=None):
    fig.update_xaxes(**(x if x else AX))
    fig.update_yaxes(**(y if y else AX))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:1.55rem;margin-bottom:.2rem'>Credit Risk · Expected Loss Dashboard</h1>", unsafe_allow_html=True)
sc_color = C['green'] if preset=="Baseline" else C['amber'] if preset=="Adverse" else C['red']
st.markdown(
    f"<span style='background:{sc_color}22;color:{sc_color};border:1px solid {sc_color}44;"
    f"border-radius:20px;padding:3px 14px;font-size:12px;font-weight:600'>● {preset} Scenario &nbsp;"
    f"PD ×{pd_mult:.1f} &nbsp; LGD +{lgd_shock*100:.0f}pp</span>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
with k1:
    st.metric("Total EAD", f"${total_ead_f/1e6:.1f}M", f"{len(fdf):,} loans")
with k2:
    st.metric("Baseline EL", f"${baseline_el/1e6:.2f}M", f"{baseline_el/total_ead_f*100:.2f}% of EAD")
with k3:
    sign = "+" if el_delta_pct >= 0 else ""
    st.metric("Stressed EL", f"${stressed_el/1e6:.2f}M",
              f"{sign}{el_delta_pct:.1f}% vs baseline", delta_color="inverse" if el_delta_pct>0 else "normal")
with k4:
    st.metric("EL Coverage", f"{el_cov_s:.2f}%", f"Wtd PD: {wtd_pd_s*100:.2f}%")
with k5:
    st.metric("Actual Default Rate", f"{actual_dr:.1f}%",
              f"{fdf['loan_status'].sum():,} defaulted loans")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: EL by Grade  +  Scenario Comparison ───────────────────────────────
r1a, r1b = st.columns(2)

with r1a:
    st.markdown("#### EL by Loan Grade")
    g_data = fdf.groupby('grade').agg(baseline=('el','sum'), stressed=('el_s','sum'))
    g_data = g_data.reindex([g for g in GRADES if g in g_data.index]).reset_index()
    fig = go.Figure()
    fig.add_bar(x=g_data['grade'], y=g_data['baseline']/1e6,
                name='Baseline EL', marker_color=C['accent'], opacity=0.8)
    fig.add_bar(x=g_data['grade'], y=g_data['stressed']/1e6,
                name='Stressed EL',  marker_color=C['red'],    opacity=0.9)
    fig.update_layout(**PL, barmode='group', yaxis_title='EL ($M)', height=300)
    ax(fig)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with r1b:
    st.markdown("#### Scenario Comparison")
    sc_names  = ['Baseline','Adverse','Severe','Extreme']
    sc_params = [(1.00,0.00),(1.50,0.05),(2.25,0.12),(3.50,0.20)]
    sc_els    = []
    for pm, ls in sc_params:
        el_s = (np.clip(fdf['pd']*pm,0.001,0.999) * np.clip(fdf['lgd']+ls,0.001,0.999) * fdf['ead']).sum()
        sc_els.append(el_s/1e6)
    sc_colors = [C['green'], C['amber'], C['red'], '#7b1a1a']
    fig2 = go.Figure()
    fig2.add_bar(x=sc_names, y=sc_els, marker_color=sc_colors, showlegend=False,
                 text=[f'${v:.1f}M' for v in sc_els], textposition='outside',
                 textfont=dict(color=C['text'], size=12))
    if preset in sc_names:
        idx = sc_names.index(preset)
        fig2.add_shape(type='rect', x0=idx-.45, x1=idx+.45, y0=0, y1=sc_els[idx]*1.08,
                       line=dict(color=sc_color, width=2), fillcolor='rgba(0,0,0,0)')
    fig2.update_layout(**PL, yaxis_title='Total EL ($M)', height=300)
    ax(fig2, y=dict(range=[0,max(sc_els)*1.22], gridcolor=C['border'], zerolinecolor=C['border']))
    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: EL over time  +  Model Validation (PD vs Actual Default) ───────────
r2a, r2b = st.columns([1.3, 1])

with r2a:
    st.markdown("#### EL by Origination Quarter")
    q = fdf.copy()
    q['quarter'] = fdf['origination_date'].dt.to_period('Q').astype(str)
    q_agg = q.groupby('quarter').agg(
        baseline=('el','sum'), stressed=('el_s','sum')).reset_index().sort_values('quarter')
    fig3 = go.Figure()
    fig3.add_scatter(x=q_agg['quarter'], y=q_agg['baseline']/1e6, name='Baseline',
                     line=dict(color=C['accent'], width=2.5), mode='lines+markers', marker=dict(size=5))
    fig3.add_scatter(x=q_agg['quarter'], y=q_agg['stressed']/1e6, name='Stressed',
                     line=dict(color=C['red'], width=2.5, dash='dash'), mode='lines+markers', marker=dict(size=5))
    fig3.add_traces([go.Scatter(
        x=list(q_agg['quarter'])+list(q_agg['quarter'])[::-1],
        y=list(q_agg['baseline']/1e6)+list(q_agg['stressed']/1e6)[::-1],
        fill='toself', fillcolor='rgba(232,93,74,0.07)', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    )])
    fig3.update_layout(**PL, yaxis_title='EL ($M)', height=310)
    ax(fig3, x=dict(tickangle=-30, gridcolor=C['border'], zerolinecolor=C['border']))
    st.plotly_chart(fig3, use_container_width=True)

with r2b:
    st.markdown("#### Model Validation: PD vs Actual Default Rate")
    grade_val = fdf.groupby('grade').agg(
        modeled_pd    = ('pd',           'mean'),
        actual_dr     = ('loan_status',  'mean'),
        loan_count    = ('loan_id',      'count'),
    ).reindex([g for g in GRADES if g in fdf['grade'].unique()]).reset_index()
    fig4 = go.Figure()
    for _, row in grade_val.iterrows():
        col = C['grade'].get(row['grade'], C['muted'])
        fig4.add_scatter(
            x=[row['modeled_pd']*100], y=[row['actual_dr']*100],
            mode='markers+text', text=[row['grade']],
            textposition='top center', textfont=dict(color=col, size=12, family='Inter'),
            marker=dict(size=max(10, row['loan_count']/200), color=col, opacity=0.85,
                        line=dict(color='white', width=1)),
            showlegend=False,
            hovertemplate=f"Grade {row['grade']}<br>Modeled PD: {row['modeled_pd']*100:.1f}%<br>Actual DR: {row['actual_dr']*100:.1f}%<extra></extra>"
        )
    # 45° perfect-fit line
    lo, hi = 0, max(grade_val['actual_dr'].max()*100, grade_val['modeled_pd'].max()*100) * 1.1
    fig4.add_scatter(x=[lo,hi], y=[lo,hi], mode='lines',
                     line=dict(color=C['muted'], dash='dot', width=1), showlegend=False)
    fig4.update_layout(**PL, xaxis_title='Modeled PD (%)', yaxis_title='Actual Default Rate (%)', height=310)
    ax(fig4, x=dict(range=[0, hi], gridcolor=C['border'], zerolinecolor=C['border']), y=dict(range=[0, hi], gridcolor=C['border'], zerolinecolor=C['border']))
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Concentration heatmap  +  EL by Loan Intent ───────────────────────
r3a, r3b = st.columns([1.1, 1])

with r3a:
    st.markdown("#### Concentration Heatmap (EAD %)")
    heat = fdf.groupby(['grade','product_type'])['ead'].sum().unstack(fill_value=0)
    heat_pct = (heat / heat.values.sum() * 100).round(2)
    heat_pct = heat_pct.reindex([g for g in GRADES if g in heat_pct.index])
    fig5 = go.Figure(go.Heatmap(
        z=heat_pct.values, x=heat_pct.columns.tolist(), y=heat_pct.index.tolist(),
        colorscale=[[0,'#1a1d27'],[0.3,'#2d3d7a'],[0.7,'#4f8ef7'],[1.0,'#e85d4a']],
        text=heat_pct.values, texttemplate='%{text:.1f}%',
        textfont=dict(size=9, color='white'), showscale=True,
        colorbar=dict(tickfont=dict(color=C['muted']), bgcolor='rgba(0,0,0,0)'),
    ))
    fig5.update_layout(**PL, height=310)
    ax(fig5, x=dict(tickangle=-25, gridcolor='rgba(0,0,0,0)', zerolinecolor='rgba(0,0,0,0)'), y=dict(gridcolor='rgba(0,0,0,0)', zerolinecolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig5, use_container_width=True)

with r3b:
    st.markdown("#### Stressed EL by Loan Intent")
    prod_data = fdf.groupby('product_type').agg(
        baseline=('el','sum'), stressed=('el_s','sum')).reset_index()
    prod_data = prod_data.sort_values('stressed', ascending=True)
    fig6 = go.Figure()
    fig6.add_bar(y=prod_data['product_type'], x=prod_data['baseline']/1e6,
                 name='Baseline', orientation='h', marker_color=C['accent'], opacity=0.8)
    fig6.add_bar(y=prod_data['product_type'], x=prod_data['stressed']/1e6,
                 name='Stressed',  orientation='h', marker_color=C['red'],    opacity=0.85)
    fig6.update_layout(**PL, barmode='overlay', xaxis_title='EL ($M)', height=310)
    ax(fig6, y=dict(gridcolor='rgba(0,0,0,0)', zerolinecolor='rgba(0,0,0,0)'))
    fig6.update_traces(marker_line_width=0)
    st.plotly_chart(fig6, use_container_width=True)

# ── Row 4: EL by Home Ownership  +  EL Rate violin ───────────────────────────
r4a, r4b = st.columns(2)

with r4a:
    st.markdown("#### EL by Home Ownership")
    reg_data = fdf.groupby('region').agg(
        baseline=('el','sum'), stressed=('el_s','sum')).reset_index()
    reg_data = reg_data.sort_values('stressed', ascending=False)
    fig7 = go.Figure()
    fig7.add_bar(x=reg_data['region'], y=reg_data['baseline']/1e6,
                 name='Baseline', marker_color=C['accent'], opacity=0.8)
    fig7.add_bar(x=reg_data['region'], y=reg_data['stressed']/1e6,
                 name='Stressed',  marker_color=C['red'],    opacity=0.9)
    fig7.update_layout(**PL, barmode='group', yaxis_title='EL ($M)', height=290)
    ax(fig7)
    fig7.update_traces(marker_line_width=0)
    st.plotly_chart(fig7, use_container_width=True)

with r4b:
    st.markdown("#### Stressed EL Rate Distribution by Grade")
    fig8 = go.Figure()
    for grade in [g for g in GRADES if g in fdf['grade'].unique()]:
        sub = fdf[fdf['grade']==grade]
        el_rate_pct = sub['el_s'] / sub['ead'] * 100
        if len(el_rate_pct) > 1:
            fig8.add_violin(y=el_rate_pct, name=grade, box_visible=True,
                            meanline_visible=True,
                            fillcolor=rgba(C['grade'][grade], 0.33),
                            line_color=C['grade'][grade], opacity=0.85)
    fig8.update_layout(**PL, yaxis_title='Stressed EL Rate (%)',
                       xaxis_title='Grade', height=290, showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

# ── Loan-level table ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📋 Loan-Level Detail")
ta, tb = st.columns([3,1])
with ta:
    sort_col = st.selectbox("Sort by",
        ['el_s','ead','pd_s','lgd_s','loan_percent_income','loan_int_rate'],
        format_func=lambda x: {
            'el_s':'Stressed EL','ead':'EAD','pd_s':'Stressed PD',
            'lgd_s':'Stressed LGD','loan_percent_income':'DTI Ratio','loan_int_rate':'Interest Rate'
        }[x])
with tb:
    n_rows = st.selectbox("Show rows", [25,50,100], index=0)

display_cols = ['loan_id','grade','region','product_type','ead',
                'loan_int_rate','loan_percent_income','pd','lgd','el','pd_s','lgd_s','el_s']
disp = fdf[display_cols].sort_values(sort_col, ascending=False).head(n_rows).copy()
disp.columns = ['Loan ID','Grade','Ownership','Intent','EAD','Int Rate','DTI',
                'PD','LGD','Baseline EL','Stressed PD','Stressed LGD','Stressed EL']
for col in ['EAD','Baseline EL','Stressed EL']:
    disp[col] = disp[col].apply(lambda x: f"${x:,.0f}")
for col in ['PD','LGD','Stressed PD','Stressed LGD','DTI']:
    disp[col] = disp[col].apply(lambda x: f"{x*100:.2f}%")
disp['Int Rate'] = disp['Int Rate'].apply(lambda x: f"{x:.2f}%")

st.dataframe(disp, use_container_width=True, height=360)

# ── Download ──────────────────────────────────────────────────────────────────
export = fdf[display_cols].copy()
csv = export.to_csv(index=False).encode()
st.download_button("⬇️  Download Stressed Portfolio CSV", csv, "stressed_portfolio.csv", "text/csv")
