import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time, warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import (load_breast_cancer, load_iris, load_wine,
                               fetch_california_housing, load_diabetes)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
  .block-container { padding-top: 1.5rem; }

  .hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px; padding: 32px; text-align: center; margin-bottom: 24px;
  }
  .hero h1 { color: #00d2ff; font-size: 2.2em; margin: 0; letter-spacing: 2px; }
  .hero p  { color: #a0c4d8; margin: 10px 0 0 0; font-size: 1.05em; }

  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 14px; text-align: center;
  }
  .metric-val  { font-size: 1.9em; font-weight: 700; color: #58a6ff; }
  .metric-name { font-size: 0.85em; color: #8b949e; margin-top: 4px; }

  .best-card {
    background: #0d2818; border: 1px solid #238636;
    border-radius: 10px; padding: 16px 20px; margin: 12px 0;
    color: #3fb950; font-size: 1.05em;
  }
  .section-title {
    font-size: 1.3em; font-weight: 700; margin: 24px 0 12px 0;
  }
  div[data-testid="stMetricValue"] > div { color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Palettes ──────────────────────────────────────────────────────────────────
COLORS = ['#58a6ff', '#3fb950', '#ff7b72', '#d2a8ff']
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',   'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',       'axes.titlecolor': '#e6edf3',
    'grid.color': '#21262d',       'grid.linewidth': 0.8,
})

# ── Registry ──────────────────────────────────────────────────────────────────
CLF_LOADERS = {
    'Breast Cancer': load_breast_cancer,
    'Iris':          load_iris,
    'Wine':          load_wine,
}
REG_LOADERS = {
    'California Housing': fetch_california_housing,
    'Diabetes':           load_diabetes,
}
CLF_MODELS = {
    'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       lambda: DecisionTreeClassifier(random_state=42),
    'Random Forest':       lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 lambda: SVC(kernel='rbf', probability=True, random_state=42),
}
REG_MODELS = {
    'Linear Regression': lambda: LinearRegression(),
    'Decision Tree':     lambda: DecisionTreeRegressor(random_state=42),
    'Random Forest':     lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR':               lambda: SVR(kernel='rbf'),
}

# ── Cached training ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_classification(ds_name):
    data   = CLF_LOADERS[ds_name](as_frame=True)
    X, y   = data.data.values, data.target
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    is_bin = len(np.unique(y)) == 2
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, stratify=y, random_state=42)
    rows, preds, probs = [], {}, {}
    for mname, make in CLF_MODELS.items():
        clf = make(); clf.fit(X_tr, y_tr)
        yp = clf.predict(X_te); yprob = clf.predict_proba(X_te)
        avg = 'binary' if is_bin else 'weighted'
        roc = roc_auc_score(y_te, yprob[:,1] if is_bin else yprob,
                            multi_class='raise' if is_bin else 'ovr',
                            average=None if is_bin else 'weighted')
        rows.append({'Model': mname,
            'Accuracy':  round(accuracy_score(y_te, yp)*100, 2),
            'Precision': round(precision_score(y_te, yp, average=avg, zero_division=0)*100, 2),
            'Recall':    round(recall_score(y_te, yp, average=avg)*100, 2),
            'F1 Score':  round(f1_score(y_te, yp, average=avg)*100, 2),
            'ROC-AUC':   round(float(roc), 4)})
        preds[mname] = (yp, yprob, y_te)
    df = pd.DataFrame(rows).set_index('Model')
    return df, preds, scaler, data.target_names.tolist(), is_bin, X_sc, y

@st.cache_data(show_spinner=False)
def train_regression(ds_name):
    data   = REG_LOADERS[ds_name](as_frame=True)
    X, y   = data.data.values, data.target
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    rows, preds = [], {}
    for mname, make in REG_MODELS.items():
        reg = make(); reg.fit(X_tr, y_tr)
        yp = reg.predict(X_te)
        rows.append({'Model': mname,
            'MAE':      round(mean_absolute_error(y_te, yp), 4),
            'MSE':      round(mean_squared_error(y_te, yp), 4),
            'RMSE':     round(np.sqrt(mean_squared_error(y_te, yp)), 4),
            'R² Score': round(r2_score(y_te, yp), 4)})
        preds[mname] = (yp, y_te)
    df = pd.DataFrame(rows).set_index('Model')
    return df, preds, X_sc, y

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ML Dashboard")
    st.markdown("---")
    task = st.radio("**Select Task**",
                    ["🔵 Classification", "🟠 Regression"],
                    label_visibility="collapsed")
    st.markdown("---")
    if "Classification" in task:
        ds_name = st.selectbox("**Dataset**", list(CLF_LOADERS.keys()))
        show_cm  = st.checkbox("Confusion Matrices", value=True)
        show_roc = st.checkbox("ROC Curves", value=True)
        show_lc  = st.checkbox("Learning Curves", value=True)
    else:
        ds_name  = st.selectbox("**Dataset**", list(REG_LOADERS.keys()))
        show_avp = st.checkbox("Actual vs Predicted", value=True)
        show_lc  = st.checkbox("Learning Curves", value=True)
    st.markdown("---")
    st.markdown("<p style='color:#8b949e; font-size:0.8em;'>B.Tech AI & DS · MBM University<br>Jodhpur, Rajasthan · 2026</p>",
                unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🤖 ML MODEL COMPARISON DASHBOARD</h1>
  <p>Classification &amp; Regression &nbsp;|&nbsp; 5 Datasets &nbsp;|&nbsp; 4 Algorithms &nbsp;|&nbsp; All Metrics</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
if "Classification" in task:
    st.markdown(f"<div class='section-title' style='color:#58a6ff;'>🔵 Classification — {ds_name}</div>",
                unsafe_allow_html=True)

    with st.spinner(f"Training all models on {ds_name}..."):
        df_r, preds, scaler, target_names, is_bin, X_sc, y = train_classification(ds_name)

    best = df_r['F1 Score'].idxmax()

    # ── Top metric cards ──
    cols = st.columns(5)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    for col, m in zip(cols, metrics):
        val = df_r.loc[best, m]
        suffix = '%' if m != 'ROC-AUC' else ''
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val'>{val}{suffix}</div>
          <div class='metric-name'>{m}<br><span style='color:#58a6ff; font-size:0.85em;'>{best}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='best-card'>🏆 &nbsp;<b>Best Model: {best}</b> &nbsp;·&nbsp; F1 = {df_r.loc[best,'F1 Score']:.2f}% &nbsp;·&nbsp; ROC-AUC = {df_r.loc[best,'ROC-AUC']:.4f}</div>",
                unsafe_allow_html=True)

    # ── Results table ──
    st.markdown("#### 📋 All Models Comparison")
    styled = df_r.style\
        .highlight_max(subset=['Accuracy','Precision','Recall','F1 Score','ROC-AUC'],
                       color='#1a4731')\
        .format({'Accuracy':'{:.2f}%','Precision':'{:.2f}%',
                 'Recall':'{:.2f}%','F1 Score':'{:.2f}%','ROC-AUC':'{:.4f}'})\
        .set_properties(**{'text-align':'center'})
    st.dataframe(styled, use_container_width=True)

    # ── Metric bar charts ──
    st.markdown("#### 📊 Metric Comparison")
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    fig.patch.set_facecolor('#0d1117')
    for ax, m in zip(axes, metrics):
        vals = df_r[m]
        bars = ax.bar(vals.index, vals.values, color=COLORS, edgecolor='#0d1117', width=0.5)
        ax.set_title(m, fontsize=10, fontweight='bold')
        ax.set_xticklabels(vals.index, rotation=30, ha='right', fontsize=7)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7, color='#e6edf3')
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Confusion Matrices ──
    if show_cm:
        st.markdown("#### 🔥 Confusion Matrices")
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.patch.set_facecolor('#0d1117')
        for ax, (mname, (yp, yprob, yte)) in zip(axes, preds.items()):
            cm = confusion_matrix(yte, yp)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        cmap='Blues', xticklabels=target_names,
                        yticklabels=target_names, linewidths=0.5,
                        linecolor='#0d1117', cbar=False,
                        annot_kws={'size':11,'weight':'bold'})
            ax.set_title(mname, fontsize=10, fontweight='bold', color='#58a6ff', pad=8)
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
            ax.tick_params(axis='x', rotation=25)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── ROC Curves ──
    if show_roc and is_bin:
        st.markdown("#### 📉 ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0d1117')
        for (mname, (yp, yprob, yte)), color in zip(preds.items(), COLORS):
            fpr, tpr, _ = roc_curve(yte, yprob[:,1])
            roc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{mname}  (AUC={roc_val:.3f})')
            ax.fill_between(fpr, tpr, alpha=0.05, color=color)
        ax.plot([0,1],[0,1],'--', color='#484f58', lw=1.5)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontsize=12, fontweight='bold', color='#58a6ff')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Learning Curves ──
    if show_lc:
        st.markdown("#### 📚 Learning Curves")
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.patch.set_facecolor('#0d1117')
        for ax, (mname, make) in zip(axes, CLF_MODELS.items()):
            clf = make()
            sizes, tr_sc, va_sc = learning_curve(
                clf, X_sc, y, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 8),
                scoring='f1_weighted', n_jobs=-1)
            tr_m = tr_sc.mean(1); va_m = va_sc.mean(1)
            ax.plot(sizes, tr_m, 'o-', color='#58a6ff', lw=2, label='Train')
            ax.plot(sizes, va_m, 's-', color='#ff7b72', lw=2, label='Validation')
            ax.fill_between(sizes, tr_sc.min(1), tr_sc.max(1), alpha=0.1, color='#58a6ff')
            ax.fill_between(sizes, va_sc.min(1), va_sc.max(1), alpha=0.1, color='#ff7b72')
            ax.set_title(mname, fontsize=9, fontweight='bold', color='#58a6ff')
            ax.set_xlabel('Training Size', fontsize=8)
            ax.set_ylabel('F1 Score', fontsize=8)
            ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
            ax.grid(True, alpha=0.2)
            ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f"<div class='section-title' style='color:#ff9500;'>🟠 Regression — {ds_name}</div>",
                unsafe_allow_html=True)

    with st.spinner(f"Training all models on {ds_name}..."):
        df_r, preds, X_sc, y = train_regression(ds_name)

    best = df_r['R² Score'].idxmax()

    # ── Top metric cards ──
    cols = st.columns(4)
    for col, m in zip(cols, ['MAE','MSE','RMSE','R² Score']):
        val = df_r.loc[best, m]
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val'>{val}</div>
          <div class='metric-name'>{m}<br><span style='color:#ff9500; font-size:0.85em;'>{best}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='best-card'>🏆 &nbsp;<b>Best Model: {best}</b> &nbsp;·&nbsp; R² = {df_r.loc[best,'R² Score']:.4f} &nbsp;·&nbsp; RMSE = {df_r.loc[best,'RMSE']:.4f}</div>",
                unsafe_allow_html=True)

    # ── Results table ──
    st.markdown("#### 📋 All Models Comparison")
    styled = df_r.style\
        .highlight_max(subset=['R² Score'], color='#1a4731')\
        .highlight_min(subset=['MAE','MSE','RMSE'], color='#1a3a1a')\
        .set_properties(**{'text-align':'center'})
    st.dataframe(styled, use_container_width=True)

    # ── Metric bar charts ──
    st.markdown("#### 📊 Metric Comparison")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.patch.set_facecolor('#0d1117')
    for ax, m in zip(axes, ['MAE','MSE','RMSE','R² Score']):
        vals = df_r[m]
        bars = ax.bar(vals.index, vals.values, color=COLORS, edgecolor='#0d1117', width=0.5)
        ax.set_title(m, fontsize=10, fontweight='bold')
        ax.set_xticklabels(vals.index, rotation=30, ha='right', fontsize=7)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, color='#e6edf3')
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Actual vs Predicted ──
    if show_avp:
        st.markdown("#### 🔍 Actual vs Predicted")
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.patch.set_facecolor('#0d1117')
        for ax, ((mname, (yp, yte)), color) in zip(axes, zip(preds.items(), COLORS)):
            r2 = r2_score(yte, yp)
            ax.scatter(yte, yp, alpha=0.3, color=color, s=10, edgecolors='none')
            mn = min(yte.min(), yp.min()); mx = max(yte.max(), yp.max())
            ax.plot([mn,mx],[mn,mx],'--', color='#e6edf3', lw=1.5)
            ax.set_title(f'{mname}\nR²={r2:.3f}', fontsize=9, fontweight='bold', color='#ff9500')
            ax.set_xlabel('Actual', fontsize=8); ax.set_ylabel('Predicted', fontsize=8)
            ax.grid(True, alpha=0.15)
            ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Learning Curves ──
    if show_lc:
        st.markdown("#### 📚 Learning Curves")
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.patch.set_facecolor('#0d1117')
        for ax, (mname, make) in zip(axes, REG_MODELS.items()):
            reg = make()
            sizes, tr_sc, va_sc = learning_curve(
                reg, X_sc, y, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 8),
                scoring='r2', n_jobs=-1)
            tr_m = tr_sc.mean(1); va_m = va_sc.mean(1)
            ax.plot(sizes, tr_m, 'o-', color='#ff9500', lw=2, label='Train')
            ax.plot(sizes, va_m, 's-', color='#ff7b72', lw=2, label='Validation')
            ax.fill_between(sizes, tr_sc.min(1), tr_sc.max(1), alpha=0.1, color='#ff9500')
            ax.fill_between(sizes, va_sc.min(1), va_sc.max(1), alpha=0.1, color='#ff7b72')
            ax.set_title(mname, fontsize=9, fontweight='bold', color='#ff9500')
            ax.set_xlabel('Training Size', fontsize=8)
            ax.set_ylabel('R² Score', fontsize=8)
            ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
            ax.grid(True, alpha=0.2)
            ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
