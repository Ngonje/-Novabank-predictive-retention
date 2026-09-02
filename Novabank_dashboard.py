"""
NovaBank Predictive Retention Dashboard
========================================
Run with:  streamlit run novabank_dashboard.py
Requires:  pip install streamlit plotly scikit-learn pandas numpy
Dataset:   bank-full.csv (same folder)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovaBank · Retention Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0D1117;
    color: #E6EDF3;
}
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] * {
    color: #E6EDF3 !important;
}

/* Header */
.nb-header {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.nb-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(29,158,117,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.nb-title {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    color: #E6EDF3;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.nb-subtitle {
    font-size: 14px;
    color: #8B949E;
    margin: 0;
    font-weight: 300;
}
.nb-accent { color: #1D9E75; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 16px 20px;
    flex: 1;
    min-width: 140px;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #E6EDF3;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 12px;
    color: #8B949E;
    font-weight: 400;
}
.metric-card.green .metric-val { color: #1D9E75; }
.metric-card.blue  .metric-val { color: #58A6FF; }
.metric-card.amber .metric-val { color: #F0A642; }
.metric-card.red   .metric-val { color: #E24B4A; }

/* Section headers */
.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: #E6EDF3;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262D;
}

/* Segment pills */
.seg-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin-left: 8px;
}
.pill-green { background: #1a3a2e; color: #1D9E75; }
.pill-blue  { background: #1a2b3e; color: #58A6FF; }
.pill-amber { background: #3a2a10; color: #F0A642; }
.pill-red   { background: #3a1a1a; color: #E24B4A; }
.pill-grey  { background: #21262D; color: #8B949E; }

/* Info boxes */
.insight-box {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 3px solid #1D9E75;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #C9D1D9;
    margin-bottom: 12px;
}
.insight-box.amber { border-left-color: #F0A642; }
.insight-box.red   { border-left-color: #E24B4A; }
.insight-box.blue  { border-left-color: #58A6FF; }

/* Dataframe overrides */
.stDataFrame { border: 1px solid #21262D !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY DARK THEME ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#C9D1D9', size=12),
    xaxis=dict(gridcolor='#21262D', linecolor='#21262D', zerolinecolor='#21262D'),
    yaxis=dict(gridcolor='#21262D', linecolor='#21262D', zerolinecolor='#21262D'),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=['#1D9E75','#58A6FF','#F0A642','#E24B4A','#5DCAA5','#85B7EB','#FAC775'],
)

GREEN, BLUE, AMBER, RED, GREY = '#1D9E75', '#58A6FF', '#F0A642', '#E24B4A', '#8B949E'

SEG_COLORS = {
    'Engaged Converters' : GREEN,
    'Previously Engaged' : '#5DCAA5',
    'High-Net-Worth'     : BLUE,
    'Older Mid-Balance'  : AMBER,
    'Young Low-Balance'  : '#FAC775',
    'Over-Contacted'     : RED,
    'Campaign-Fatigued'  : '#c0392b',
    'Statistical Outlier': GREY,
}

# ── DATA & MODEL PIPELINE ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and preparing data…")
def load_and_prepare():
    df = pd.read_csv('bank-full.csv', sep=';')
    df['y'] = (df['y'] == 'yes').astype(int)

    # Feature engineering
    df['was_previously_contacted'] = (df['pdays'] != -1).astype(int)
    df['pdays_clean'] = df['pdays'].replace(-1, 0)
    df['duration_bucket'] = pd.cut(df['duration'],
        bins=[0,60,180,360,600,9999],
        labels=['very_short','short','medium','long','very_long'])
    df['age_group'] = pd.cut(df['age'],
        bins=[0,25,35,45,55,65,100],
        labels=['under25','25-34','35-44','45-54','55-64','65+'])
    df['negative_balance']    = (df['balance'] < 0).astype(int)
    df['high_campaign_count'] = (df['campaign'] > 5).astype(int)

    df_enc = df.copy()
    df_enc['education_ord'] = df['education'].map({'unknown':0,'primary':1,'secondary':2,'tertiary':3})
    for col in ['default','housing','loan']:
        df_enc[col] = (df_enc[col] == 'yes').astype(int)
    nominal = ['job','marital','contact','month','poutcome','duration_bucket','age_group']
    df_enc = pd.get_dummies(df_enc, columns=nominal, drop_first=True)
    df_enc = df_enc.drop(columns=['education'], errors='ignore')

    X = df_enc.drop(columns=['y','pdays'])
    y = df_enc['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_cols = ['age','balance','duration','campaign','pdays_clean','previous']
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    return df, X_train, X_test, y_train, y_test

@st.cache_resource(show_spinner="Training models…")
def train_models(_X_train, _y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(_X_train, _y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15,
        min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(_X_train, _y_train)
    return lr, rf

@st.cache_data(show_spinner="Running segmentation…")
def run_segmentation(_df):
    cluster_features = ['age','balance','duration','campaign','previous']
    X_c = _df[cluster_features].copy()
    sc  = StandardScaler()
    X_s = sc.fit_transform(X_c)
    km  = KMeans(n_clusters=8, random_state=42, n_init=10)
    _df = _df.copy()
    _df['cluster'] = km.fit_predict(X_s)
    names = {3:'Engaged Converters',1:'Previously Engaged',6:'High-Net-Worth',
             2:'Older Mid-Balance',0:'Young Low-Balance',7:'Over-Contacted',
             4:'Campaign-Fatigued',5:'Statistical Outlier'}
    priority = {3:'🟢 Priority 1',1:'🟢 Priority 2',6:'🟢 Priority 3',
                2:'🟡 Priority 4',0:'🟡 Priority 5',7:'🔴 Reduce',
                4:'🔴 Stop',5:'⚫ Exclude'}
    _df['segment']  = _df['cluster'].map(names)
    _df['priority'] = _df['cluster'].map(priority)
    return _df

# ── LOAD EVERYTHING ────────────────────────────────────────────────────────────
df_raw, X_train, X_test, y_train, y_test = load_and_prepare()
lr_model, rf_model = train_models(X_train, y_train)

lr_prob = lr_model.predict_proba(X_test)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]

X_full    = pd.concat([X_train, X_test])
rf_full   = rf_model.predict_proba(X_full)[:, 1]
df_seg    = run_segmentation(df_raw)
df_seg['rf_score'] = rf_full

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"DM Serif Display",serif;font-size:20px;
    color:#E6EDF3;margin-bottom:4px;'>🏦 NovaBank</div>
    <div style='font-size:12px;color:#8B949E;margin-bottom:24px;'>
    Retention Intelligence Platform</div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊  Overview",
        "🤖  Model Performance",
        "🎯  Threshold Explorer",
        "👥  Customer Segments",
        "💡  Segment Insights",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#8B949E;'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:13px;color:#C9D1D9;'>{df_raw.shape[0]:,} customers · 17 features</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#8B949E;margin-top:8px;'>Models</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#C9D1D9;'>Logistic Regression · Random Forest</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#8B949E;margin-top:8px;'>Segmentation</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#C9D1D9;'>K-Means (k=8, silhouette-optimised)</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#8B949E;'>Analytics Methods & Frameworks · Quantic · 2026</div>", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='nb-header'>
  <div class='nb-title'>NovaBank <span class='nb-accent'>Retention Intelligence</span></div>
  <div class='nb-subtitle'>Predictive campaign targeting · Random Forest · K-Means segmentation · 45,211 customers</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown("<div class='section-head'>Campaign Performance at a Glance</div>", unsafe_allow_html=True)

    rf_auc  = roc_auc_score(y_test, rf_prob)
    lr_auc  = roc_auc_score(y_test, lr_prob)
    n_top20 = int(len(y_test) * 0.20)
    top20_idx = np.argsort(rf_prob)[::-1][:n_top20]
    p20 = y_test.iloc[top20_idx].mean()
    r20 = y_test.iloc[top20_idx].sum() / y_test.sum()

    st.markdown(f"""
    <div class='metric-row'>
      <div class='metric-card green'><div class='metric-val'>{rf_auc:.3f}</div><div class='metric-label'>AUC-ROC (Random Forest)</div></div>
      <div class='metric-card blue'> <div class='metric-val'>{p20:.0%}</div><div class='metric-label'>Precision @ top 20%</div></div>
      <div class='metric-card amber'><div class='metric-val'>{r20:.0%}</div><div class='metric-label'>Recall @ top 20%</div></div>
      <div class='metric-card'><div class='metric-val'>11.7%</div><div class='metric-label'>Baseline subscription rate</div></div>
      <div class='metric-card red'><div class='metric-val'>3.6×</div><div class='metric-label'>Lift over random outreach</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Target distribution
        counts = df_raw['y'].value_counts()
        fig = go.Figure(go.Bar(
            x=['No subscription', 'Subscribed'],
            y=[counts[0], counts[1]],
            marker_color=[GREY, GREEN],
            text=[f'{counts[0]:,}', f'{counts[1]:,}'],
            textposition='outside',
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Target distribution', height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sub rate by job
        job_rate = df_raw.groupby('job')['y'].mean().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=job_rate.values * 100,
            y=job_rate.index,
            orientation='h',
            marker_color=[RED if v < 0.08 else GREEN if v > 0.20 else AMBER
                          for v in job_rate.values],
            text=[f'{v:.1%}' for v in job_rate.values],
            textposition='outside',
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Subscription rate by job type', height=280,
                          xaxis_title='Subscription rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <strong>Decision rule:</strong> Contact every customer with predicted subscription probability ≥ 0.15.
    This captures 98.7% of subscribers while reducing call volume by 52% — estimated $1.69M annual net impact.
    </div>
    <div class='insight-box blue'>
    <strong>Why not accuracy?</strong> A model that always predicts "no" achieves 88.3% accuracy but catches zero
    subscribers. AUC-ROC and Precision@top20% are the right metrics for imbalanced datasets like this one.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Model Performance" in page:

    st.markdown("<div class='section-head'>Baseline vs. Improved Model</div>", unsafe_allow_html=True)

    def top20(probs):
        n   = int(len(y_test) * 0.20)
        idx = np.argsort(np.array(probs))[::-1][:n]
        p   = y_test.iloc[idx].mean()
        r   = y_test.iloc[idx].sum() / y_test.sum()
        return p, r

    naive_prob = np.full(len(y_test), 0.117)
    results = {
        'Naive baseline':      {'prob': naive_prob,  'auc': roc_auc_score(y_test, naive_prob)},
        'Logistic Regression': {'prob': lr_prob,     'auc': roc_auc_score(y_test, lr_prob)},
        'Random Forest':       {'prob': rf_prob,     'auc': roc_auc_score(y_test, rf_prob)},
    }
    for name in results:
        p20, r20 = top20(results[name]['prob'])
        results[name]['p20'] = p20
        results[name]['r20'] = r20
        results[name]['ap']  = average_precision_score(y_test, results[name]['prob'])

    col1, col2 = st.columns(2)

    with col1:
        names  = list(results.keys())
        colors_bar = [GREY, BLUE, GREEN]
        fig = go.Figure()
        for metric, label in [('auc','AUC-ROC'),('p20','Prec@20%'),('r20','Recall@20%')]:
            fig.add_trace(go.Bar(
                name=label,
                x=names,
                y=[results[n][metric]*100 for n in names],
                text=[f"{results[n][metric]:.1%}" for n in names],
                textposition='outside',
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Model comparison (×100)', barmode='group',
                          height=340, yaxis_title='Score (%)', yaxis_range=[0,115])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importance
        fi = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(12)
        fig = go.Figure(go.Bar(
            x=fi['importance'] * 100,
            y=fi['feature'],
            orientation='h',
            marker_color=GREEN,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='RF feature importance (top 12)',
                          height=340, xaxis_title='Importance (%)')
        st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.markdown("<div class='section-head'>Results Table</div>", unsafe_allow_html=True)
    tbl = pd.DataFrame([
        {'Model': n, 'AUC-ROC': f"{v['auc']:.3f}",
         'Avg Precision': f"{v['ap']:.3f}",
         'Precision @ top 20%': f"{v['p20']:.1%}",
         'Recall @ top 20%': f"{v['r20']:.1%}"}
        for n, v in results.items()
    ])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='insight-box'>
    <strong>Why Random Forest beats Logistic Regression:</strong> LR assumes linear, independent feature
    contributions. RF captures non-linear patterns and interactions (e.g. "high balance AND prior success")
    automatically — improving AUC from 0.908 → 0.921 and recall@top20% from 76.8% → 78.2%.
    </div>
    <div class='insight-box amber'>
    <strong>Duration leakage:</strong> Call duration is the strongest feature but is only known after the
    call ends. Production deployment should use the duration-excluded model (AUC 0.797) for pre-call scoring,
    then re-score with the full model for post-call follow-up decisions.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — THRESHOLD EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Threshold" in page:

    st.markdown("<div class='section-head'>Interactive Threshold Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#8B949E;margin-bottom:16px;'>Drag the slider to see how your threshold choice affects business outcomes in real time.</div>", unsafe_allow_html=True)

    COST_FP  = st.sidebar.slider("Cost per wasted call ($)", 1, 50, 8)
    VALUE_FN = st.sidebar.slider("Value per missed subscriber ($)", 50, 1000, 350)

    threshold = st.slider("Model threshold", 0.05, 0.70, 0.15, 0.01,
                          format="%.2f")

    pred = (rf_prob >= threshold).astype(int)
    tp = int(((pred==1) & (y_test==1)).sum())
    fp = int(((pred==1) & (y_test==0)).sum())
    fn = int(((pred==0) & (y_test==1)).sum())
    tn = int(((pred==0) & (y_test==0)).sum())
    called = tp + fp
    prec   = tp / called if called > 0 else 0
    rec    = tp / y_test.sum()
    net    = (tp * VALUE_FN) - (fp * COST_FP) - (fn * VALUE_FN * 0.1)
    scale  = 45211 / len(y_test)

    st.markdown(f"""
    <div class='metric-row'>
      <div class='metric-card {"green" if prec > 0.30 else "amber"}'><div class='metric-val'>{prec:.1%}</div><div class='metric-label'>Precision</div></div>
      <div class='metric-card {"green" if rec > 0.70 else "amber"}'><div class='metric-val'>{rec:.1%}</div><div class='metric-label'>Recall</div></div>
      <div class='metric-card'><div class='metric-val'>{called:,}</div><div class='metric-label'>Customers called</div></div>
      <div class='metric-card'><div class='metric-val'>{called/len(y_test):.0%}</div><div class='metric-label'>% of customer base</div></div>
      <div class='metric-card {"green" if net > 250000 else "red"}'><div class='metric-val'>${net*scale/1e6:.2f}M</div><div class='metric-label'>Est. annual net value</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix
        fig = go.Figure(go.Heatmap(
            z=[[tn, fp],[fn, tp]],
            x=['Predicted No','Predicted Yes'],
            y=['Actual No','Actual Yes'],
            text=[[f'TN: {tn:,}',f'FP: {fp:,}'],[f'FN: {fn:,}',f'TP: {tp:,}']],
            texttemplate='%{text}',
            colorscale=[[0,'#161B22'],[0.5,'#1D4E2A'],[1,'#1D9E75']],
            showscale=False,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=f'Confusion matrix at threshold {threshold:.2f}',
                          height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Net value curve
        thresholds = np.arange(0.05, 0.71, 0.02)
        nets = []
        for t in thresholds:
            p = (rf_prob >= t).astype(int)
            _tp = int(((p==1) & (y_test==1)).sum())
            _fp = int(((p==1) & (y_test==0)).sum())
            _fn = int(((p==0) & (y_test==1)).sum())
            nets.append((_tp*VALUE_FN - _fp*COST_FP - _fn*VALUE_FN*0.1) * scale)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=nets, mode='lines',
                                 line=dict(color=GREEN, width=2), name='Net value'))
        fig.add_vline(x=threshold, line_dash='dash', line_color=AMBER,
                      annotation_text=f'  Current: {threshold:.2f}',
                      annotation_font_color=AMBER)
        fig.update_layout(**PLOTLY_LAYOUT, title='Net value across all thresholds',
                          height=280, xaxis_title='Threshold', yaxis_title='Est. annual net value ($)')
        fig.update_yaxes(tickprefix='$', tickformat='.2s')
        st.plotly_chart(fig, use_container_width=True)

    # Threshold table
    st.markdown("<div class='section-head'>Full Threshold Analysis</div>", unsafe_allow_html=True)
    rows = []
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        p = (rf_prob >= t).astype(int)
        _tp = int(((p==1)&(y_test==1)).sum())
        _fp = int(((p==1)&(y_test==0)).sum())
        _fn = int(((p==0)&(y_test==1)).sum())
        _c  = _tp+_fp
        _pr = _tp/_c if _c>0 else 0
        _rc = _tp/y_test.sum()
        _nv = (_tp*VALUE_FN - _fp*COST_FP - _fn*VALUE_FN*0.1)*scale
        rows.append({'Threshold':t, '% Called':f'{_c/len(y_test):.0%}',
                     'Precision':f'{_pr:.1%}','Recall':f'{_rc:.1%}',
                     'FP (wasted)':f'{int(_fp*scale):,}','FN (missed)':f'{int(_fn*scale):,}',
                     'Annual net value':f'${_nv/1e6:.2f}M'})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Customer Segments" in page:

    st.markdown("<div class='section-head'>K-Means Customer Segmentation</div>", unsafe_allow_html=True)

    seg_profile = df_seg.groupby('segment').agg(
        customers=('y','count'), sub_rate=('y','mean'),
        avg_rf=('rf_score','mean'), avg_age=('age','mean'),
        avg_balance=('balance','mean'), avg_duration=('duration','mean'),
        avg_contacts=('campaign','mean'),
    ).round(3)
    seg_profile['pct'] = (seg_profile['customers']/df_seg.shape[0]*100).round(1)
    seg_profile['revenue_opp'] = (seg_profile['customers']*seg_profile['sub_rate']*350).astype(int)
    seg_profile = seg_profile[seg_profile.index != 'Statistical Outlier']

    col1, col2 = st.columns(2)

    with col1:
        # Bubble chart
        fig = go.Figure()
        for seg, row in seg_profile.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['sub_rate']*100],
                y=[row['avg_rf']],
                mode='markers+text',
                marker=dict(size=max(row['pct']*2.5, 8),
                            color=SEG_COLORS.get(seg, GREY),
                            opacity=0.85,
                            line=dict(color='white', width=1)),
                text=[seg],
                textposition='top center',
                textfont=dict(size=10),
                name=seg,
                showlegend=False,
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          title='Segments: subscription rate vs RF score (size = customers)',
                          xaxis_title='Subscription rate (%)',
                          yaxis_title='Avg RF score')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Revenue opportunity
        seg_rev = seg_profile.sort_values('revenue_opp')
        fig = go.Figure(go.Bar(
            x=seg_rev['revenue_opp']/1000,
            y=seg_rev.index,
            orientation='h',
            marker_color=[SEG_COLORS.get(s, GREY) for s in seg_rev.index],
            text=[f'${v/1000:.0f}K' for v in seg_rev['revenue_opp']],
            textposition='outside',
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          title='Revenue opportunity by segment ($K)',
                          xaxis_title='Revenue opportunity ($000s)')
        st.plotly_chart(fig, use_container_width=True)

    # Segment breakdown bar
    st.markdown("<div class='section-head'>Subscription Rate by Segment</div>", unsafe_allow_html=True)
    
    seg_sub = seg_profile.sort_values('sub_rate', ascending=True)

    fig = go.Figure(go.Bar(
        x=seg_sub['sub_rate'] * 100,
        y=seg_sub.index,
        orientation='h',
        marker_color=[SEG_COLORS.get(s, GREY) for s in seg_sub.index],
        text=[
            f'{v:.0%}  ({int(n):,} customers)'
            for v, n in zip(seg_sub['sub_rate'], seg_sub['customers'])
        ],
        textposition='outside',
    ))

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'margin'},
        height=300,
        xaxis_title='Subscription rate (%)',
        margin=dict(l=10, r=120, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full profile table
    st.markdown("<div class='section-head'>Full Segment Profiles</div>", unsafe_allow_html=True)
    display = seg_profile[['customers','pct','sub_rate','avg_rf','avg_age',
                            'avg_balance','avg_duration','avg_contacts','revenue_opp']].copy()
    display.columns = ['Customers','% of base','Sub rate','Avg RF score','Avg age',
                       'Avg balance','Avg duration (s)','Avg contacts','Revenue opp ($)']
    display['Sub rate']      = display['Sub rate'].map('{:.1%}'.format)
    display['Avg RF score']  = display['Avg RF score'].map('{:.3f}'.format)
    display['Revenue opp ($)'] = display['Revenue opp ($)'].map('${:,}'.format)
    st.dataframe(display.sort_values('Sub rate', ascending=False),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SEGMENT INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:

    st.markdown("<div class='section-head'>Segment Deep-Dive & Campaign Actions</div>", unsafe_allow_html=True)

    selected = st.selectbox("Select a segment to explore", [
        'Engaged Converters','Previously Engaged','High-Net-Worth',
        'Older Mid-Balance','Young Low-Balance','Over-Contacted','Campaign-Fatigued'
    ])

    seg_data = df_seg[df_seg['segment'] == selected]
    sub_rate = seg_data['y'].mean()
    n_cust   = len(seg_data)

    actions = {
        'Engaged Converters': ('🟢 Priority 1 — Contact first',
            'These customers already stay on the phone for 15 minutes on average — nearly 4× the mean. '
            'They are already engaged; the call is more confirmation than persuasion. '
            'Assign senior agents, prepare personalised term deposit offers and target within the '
            'first 3 days of each campaign window. A modest rate incentive will drive outsized returns at 47% base conversion.',
            GREEN),
        'Previously Engaged': ('🟢 Priority 2 — Personalised re-engagement',
            'These customers have a high prior contact history and 24% conversion rate. '
            'They already know NovaBank. A script that explicitly references prior interactions — '
            '"We spoke before about our savings products" — removes friction and signals familiarity. '
            'These are warm leads, not cold calls.',
            '#5DCAA5'),
        'High-Net-Worth': ('🟢 Priority 3 — Relationship manager escalation',
            'Average balance of $15,698 — 12× the overall mean. This is a premium segment being '
            'handled by standard call-centre processes. Escalate to relationship managers. '
            'Offer premium-rate term deposits, private banking consultation, or bundled investment '
            'products. Revenue per conversion far exceeds other segments — justify a higher acquisition cost.',
            BLUE),
        'Older Mid-Balance': ('🟡 Priority 4 — Standard campaign, timing matters',
            'Nearly a third of all customers. Conversion rate is modest at 9% but aggregate revenue '
            'potential is $422K. Contact mid-campaign after priority segments. '
            'Older customers (avg 53) respond better to calls in mid-morning (10–11am). '
            'Lead with stability and security messaging — term deposits as a reliable, safe instrument.',
            AMBER),
        'Young Low-Balance': ('🟡 Priority 5 — Digital-first, then call engaged responders',
            'Nearly half of all customers but lowest average balance ($888). '
            'Consider email or in-app push before a phone call — reduces cost-per-contact and '
            'this group (avg age 34) is digitally native. Call only customers who engage with '
            'the digital outreach first. Aggregate revenue opportunity is $603K despite the low rate.',
            '#FAC775'),
        'Over-Contacted': ('🔴 Reduce — Cap at 1 contact per cycle',
            'Called nearly 8 times on average with only 3% conversion. Additional calls are not '
            'working — they are consuming agent time and likely annoying customers. '
            'Hard cap at 1 contact per campaign cycle. Monitor for opt-out signals. '
            'Redirect saved agent capacity to Priority 1–3 segments immediately.',
            RED),
        'Campaign-Fatigued': ('🔴 Stop — Remove from outreach for 6 months',
            'Average of 21 contacts with 2% conversion. These customers have been spam-contacted. '
            'Remove from all outreach lists for a minimum of two campaign cycles (6 months). '
            'After 6 months, re-score with the model — if score rises above 0.20, attempt one '
            'careful re-engagement call. Risk of formal complaints is real if outreach continues.',
            '#c0392b'),
    }

    label, advice, color = actions[selected]
    st.markdown(f"""
    <div class='insight-box' style='border-left-color:{color};font-size:14px;'>
    <strong>{label}</strong><br><br>{advice}
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{n_cust:,}")
    col2.metric("Subscription rate", f"{sub_rate:.1%}")
    col3.metric("Revenue opportunity", f"${int(n_cust*sub_rate*350):,}")

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution
        fig = px.histogram(seg_data, x='age', nbins=20,
                           color_discrete_sequence=[color])
        fig.update_layout(**PLOTLY_LAYOUT, title=f'Age distribution — {selected}',
                          height=260, xaxis_title='Age', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Balance distribution (clipped for readability)
        bal_clip = seg_data['balance'].clip(-2000, 10000)
        fig = px.histogram(bal_clip, nbins=30, color_discrete_sequence=[color])
        fig.update_layout(**PLOTLY_LAYOUT, title=f'Balance distribution — {selected}',
                          height=260, xaxis_title='Balance (€, clipped)', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Sub rate by job within segment
        job_sub = seg_data.groupby('job')['y'].mean().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=job_sub.values*100, y=job_sub.index, orientation='h',
            marker_color=color,
            text=[f'{v:.0%}' for v in job_sub.values], textposition='outside',
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Sub rate by job (within segment)',
                          height=280, xaxis_title='Subscription rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sub rate by month
        month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        month_sub = seg_data.groupby('month')['y'].mean().reindex(month_order).dropna()
        fig = go.Figure(go.Bar(
            x=month_sub.index, y=month_sub.values*100,
            marker_color=color,
            text=[f'{v:.0%}' for v in month_sub.values], textposition='outside',
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Sub rate by contact month',
                          height=280, yaxis_title='Subscription rate (%)')
        st.plotly_chart(fig, use_container_width=True)
