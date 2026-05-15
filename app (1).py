import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import (load_breast_cancer, load_iris, load_wine,
                               fetch_california_housing, load_diabetes)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
  .block-container { padding-top: 1.5rem; }
  .hero { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px; padding: 32px; text-align: center; margin-bottom: 24px; }
  .hero h1 { color: #00d2ff; font-size: 2.2em; margin: 0; letter-spacing: 2px; }
  .hero p  { color: #a0c4d8; margin: 10px 0 0 0; font-size: 1.05em; }
  .mcard { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 12px; text-align: center; }
  .mval  { font-size: 1.8em; font-weight: 700; color: #58a6ff; }
  .mname { font-size: 0.82em; color: #8b949e; margin-top: 4px; }
  .best  { background: #0d2818; border: 1px solid #238636; border-radius: 10px;
    padding: 14px 18px; margin: 10px 0; color: #3fb950; font-size: 1.05em; }
  .info  { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 14px 18px; margin: 10px 0; color: #c9d1d9; }
  .upload-hint { background: #161b22; border: 2px dashed #30363d; border-radius: 12px;
    padding: 30px; text-align: center; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

COLORS = ['#58a6ff', '#3fb950', '#ff7b72', '#d2a8ff']
plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22','axes.edgecolor':'#30363d',
    'axes.labelcolor':'#c9d1d9','xtick.color':'#8b949e','ytick.color':'#8b949e',
    'text.color':'#c9d1d9','axes.titlecolor':'#e6edf3','grid.color':'#21262d','grid.linewidth':0.8,
})

CLF_LOADERS = {'Breast Cancer':load_breast_cancer,'Iris':load_iris,'Wine':load_wine}
REG_LOADERS = {'California Housing':fetch_california_housing,'Diabetes':load_diabetes}

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

@st.cache_data(show_spinner=False)
def run_clf(ds_name):
    data = CLF_LOADERS[ds_name](as_frame=True)
    X, y = data.data.values, data.target
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    is_bin = len(np.unique(y)) == 2
    Xtr,Xte,ytr,yte = train_test_split(X_sc, y, test_size=0.2, stratify=y, random_state=42)
    rows, preds = [], {}
    for mn, mk in CLF_MODELS.items():
        clf=mk(); clf.fit(Xtr,ytr)
        yp=clf.predict(Xte); yprob=clf.predict_proba(Xte)
        avg='binary' if is_bin else 'weighted'
        try:
            roc=roc_auc_score(yte,yprob[:,1] if is_bin else yprob,
                              multi_class='raise' if is_bin else 'ovr',
                              average=None if is_bin else 'weighted')
        except: roc=0.0
        rows.append({'Model':mn,'Accuracy':round(accuracy_score(yte,yp)*100,2),
            'Precision':round(precision_score(yte,yp,average=avg,zero_division=0)*100,2),
            'Recall':round(recall_score(yte,yp,average=avg)*100,2),
            'F1 Score':round(f1_score(yte,yp,average=avg)*100,2),'ROC-AUC':round(float(roc),4)})
        preds[mn]=(yp,yprob,yte)
    return pd.DataFrame(rows).set_index('Model'), preds, data.target_names.tolist(), is_bin, X_sc, y

@st.cache_data(show_spinner=False)
def run_reg(ds_name):
    data = REG_LOADERS[ds_name](as_frame=True)
    X, y = data.data.values, data.target
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    rows, preds = [], {}
    for mn, mk in REG_MODELS.items():
        reg=mk(); reg.fit(Xtr,ytr); yp=reg.predict(Xte)
        rows.append({'Model':mn,'MAE':round(mean_absolute_error(yte,yp),4),
            'MSE':round(mean_squared_error(yte,yp),4),
            'RMSE':round(np.sqrt(mean_squared_error(yte,yp)),4),
            'R² Score':round(r2_score(yte,yp),4)})
        preds[mn]=(yp,yte)
    return pd.DataFrame(rows).set_index('Model'), preds, X_sc, y

def run_custom_clf(X, y, target_names):
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    is_bin = len(np.unique(y)) == 2
    Xtr,Xte,ytr,yte = train_test_split(X_sc, y, test_size=0.2, stratify=y, random_state=42)
    rows, preds = [], {}
    for mn, mk in CLF_MODELS.items():
        clf=mk(); clf.fit(Xtr,ytr)
        yp=clf.predict(Xte); yprob=clf.predict_proba(Xte)
        avg='binary' if is_bin else 'weighted'
        try:
            roc=roc_auc_score(yte,yprob[:,1] if is_bin else yprob,
                              multi_class='raise' if is_bin else 'ovr',
                              average=None if is_bin else 'weighted')
        except: roc=0.0
        rows.append({'Model':mn,'Accuracy':round(accuracy_score(yte,yp)*100,2),
            'Precision':round(precision_score(yte,yp,average=avg,zero_division=0)*100,2),
            'Recall':round(recall_score(yte,yp,average=avg)*100,2),
            'F1 Score':round(f1_score(yte,yp,average=avg)*100,2),'ROC-AUC':round(float(roc),4)})
        preds[mn]=(yp,yprob,yte)
    return pd.DataFrame(rows).set_index('Model'), preds, target_names, is_bin, X_sc, y

def run_custom_reg(X, y):
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    rows, preds = [], {}
    for mn, mk in REG_MODELS.items():
        reg=mk(); reg.fit(Xtr,ytr); yp=reg.predict(Xte)
        rows.append({'Model':mn,'MAE':round(mean_absolute_error(yte,yp),4),
            'MSE':round(mean_squared_error(yte,yp),4),
            'RMSE':round(np.sqrt(mean_squared_error(yte,yp)),4),
            'R² Score':round(r2_score(yte,yp),4)})
        preds[mn]=(yp,yte)
    return pd.DataFrame(rows).set_index('Model'), preds, X_sc, y

def show_eda(df, target_col, task_type):
    st.markdown("###  Exploratory Data Analysis")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='mcard'><div class='mval'>{df.shape[0]}</div><div class='mname'>Rows</div></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='mcard'><div class='mval'>{df.shape[1]-1}</div><div class='mname'>Features</div></div>",unsafe_allow_html=True)
    c3.markdown(f"<div class='mcard'><div class='mval'>{int(df.isnull().sum().sum())}</div><div class='mname'>Missing Values</div></div>",unsafe_allow_html=True)
    if task_type == "Classification":
        c4.markdown(f"<div class='mcard'><div class='mval'>{df[target_col].nunique()}</div><div class='mname'>Classes</div></div>",unsafe_allow_html=True)
    else:
        c4.markdown(f"<div class='mcard'><div class='mval'>{df[target_col].round(2).mean()}</div><div class='mname'>Target Mean</div></div>",unsafe_allow_html=True)

    with st.expander(" Dataset Preview (first 10 rows)", expanded=True):
        st.dataframe(df.head(10).style.set_properties(
            **{'background-color':'#161b22','color':'#c9d1d9','border':'1px solid #30363d'}),
            use_container_width=True)

    with st.expander(" Statistical Summary"):
        st.dataframe(df.describe().round(4).style.set_properties(
            **{'background-color':'#161b22','color':'#c9d1d9','border':'1px solid #30363d'}),
            use_container_width=True)

    if int(df.isnull().sum().sum()) == 0:
        st.markdown("<div class='info'> &nbsp;<b>No missing values — dataset is clean!</b></div>",unsafe_allow_html=True)
    else:
        with st.expander(" Missing Values per Column"):
            missing = df.isnull().sum()[df.isnull().sum()>0]
            st.dataframe(missing.rename("Missing Count").to_frame(), use_container_width=True)

 
    st.markdown("####  Target Distribution")
    fig, ax = plt.subplots(figsize=(7,3.5)); fig.patch.set_facecolor('#0d1117')
    if task_type == "Classification":
        counts = df[target_col].value_counts()
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color=COLORS[:len(counts)], edgecolor='#0d1117', width=0.5)
        for bar,v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(v), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e6edf3')
        ax.set_title(f'Class Distribution — {target_col}', fontsize=11, fontweight='bold', color='#58a6ff')
    else:
        ax.hist(df[target_col].dropna(), bins=40, color='#ff9500', edgecolor='#0d1117', alpha=0.85)
        ax.axvline(df[target_col].mean(), color='#ff7b72', lw=2, linestyle='--',
                   label=f'Mean={df[target_col].mean():.2f}')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        ax.set_title(f'Target Distribution — {target_col}', fontsize=11, fontweight='bold', color='#ff9500')
    ax.set_xlabel(target_col); ax.set_ylabel('Count')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col][:8]
    if num_cols:
        st.markdown("####  Feature Distributions")
        n = len(num_cols); ncols = min(4,n); nrows = (n+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3))
        fig.patch.set_facecolor('#0d1117')
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for ax, col in zip(axes, num_cols):
            ax.hist(df[col].dropna(), bins=30, color='#58a6ff', edgecolor='#0d1117', alpha=0.8)
            ax.set_title(col, fontsize=9, fontweight='bold', color='#58a6ff')
            ax.spines[['top','right']].set_visible(False)
        for ax in axes[n:]: ax.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    if len(num_cols) >= 2:
        st.markdown("####  Correlation Heatmap")
        corr_cols = num_cols[:10] + ([target_col] if pd.api.types.is_numeric_dtype(df[target_col]) else [])
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(max(7, len(corr_cols)), max(4, len(corr_cols)-2)))
        fig.patch.set_facecolor('#0d1117')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    linewidths=0.5, linecolor='#0d1117', annot_kws={'size':8}, vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold', color='#58a6ff')
        ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

def show_clf_results(df_r, preds, target_names, is_bin, X_sc, y, show_cm, show_roc, show_lc):
    best = df_r['F1 Score'].idxmax()
    metrics = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
    cols = st.columns(5)
    for col,m in zip(cols, metrics):
        v = df_r.loc[best,m]; sfx = '%' if m != 'ROC-AUC' else ''
        col.markdown(f"<div class='mcard'><div class='mval'>{v}{sfx}</div><div class='mname'>{m}<br><span style='color:#58a6ff;font-size:0.85em;'>{best}</span></div></div>",unsafe_allow_html=True)
    st.markdown(f"<div class='best'> <b>Best Model: {best}</b> · F1={df_r.loc[best,'F1 Score']:.2f}% · ROC-AUC={df_r.loc[best,'ROC-AUC']:.4f}</div>",unsafe_allow_html=True)

    st.markdown("####  All Models Comparison")
    st.dataframe(df_r.style
        .highlight_max(subset=metrics, color='#1a4731')
        .format({'Accuracy':'{:.2f}%','Precision':'{:.2f}%','Recall':'{:.2f}%','F1 Score':'{:.2f}%','ROC-AUC':'{:.4f}'})
        .set_properties(**{'text-align':'center'}), use_container_width=True)

    st.markdown("####  Metric Comparison")
    fig, axes = plt.subplots(1,5,figsize=(18,3.5)); fig.patch.set_facecolor('#0d1117')
    for ax,m in zip(axes, metrics):
        vals=df_r[m]; bars=ax.bar(vals.index, vals.values, color=COLORS, edgecolor='#0d1117', width=0.5)
        ax.set_title(m,fontsize=10,fontweight='bold')
        ax.set_xticklabels(vals.index,rotation=30,ha='right',fontsize=7)
        for bar,v in zip(bars,vals.values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f'{v:.1f}',ha='center',va='bottom',fontsize=7,color='#e6edf3')
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_cm:
        st.markdown("#### Confusion Matrices")
        fig, axes = plt.subplots(1,4,figsize=(20,4)); fig.patch.set_facecolor('#0d1117')
        for ax,(mn,(yp,yprob,yte)) in zip(axes, preds.items()):
            cm=confusion_matrix(yte,yp)
            sns.heatmap(cm,annot=True,fmt='d',ax=ax,cmap='Blues',xticklabels=target_names,
                        yticklabels=target_names,linewidths=0.5,linecolor='#0d1117',cbar=False,
                        annot_kws={'size':11,'weight':'bold'})
            ax.set_title(mn,fontsize=10,fontweight='bold',color='#58a6ff',pad=8)
            ax.set_xlabel('Predicted',fontsize=9); ax.set_ylabel('Actual',fontsize=9)
            ax.tick_params(axis='x',rotation=25)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_roc and is_bin:
        st.markdown("####  ROC Curves")
        fig, ax = plt.subplots(figsize=(7,5)); fig.patch.set_facecolor('#0d1117')
        for (mn,(yp,yprob,yte)),color in zip(preds.items(), COLORS):
            fpr,tpr,_=roc_curve(yte,yprob[:,1]); rv=auc(fpr,tpr)
            ax.plot(fpr,tpr,color=color,lw=2.5,label=f'{mn} (AUC={rv:.3f})')
            ax.fill_between(fpr,tpr,alpha=0.05,color=color)
        ax.plot([0,1],[0,1],'--',color='#484f58',lw=1.5)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves',fontsize=12,fontweight='bold',color='#58a6ff')
        ax.legend(facecolor='#161b22',edgecolor='#30363d',labelcolor='#c9d1d9')
        ax.grid(True,alpha=0.2); plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_lc:
        st.markdown("#### 📚 Learning Curves")
        fig, axes = plt.subplots(1,4,figsize=(20,4)); fig.patch.set_facecolor('#0d1117')
        for ax,(mn,mk) in zip(axes, CLF_MODELS.items()):
            clf=mk()
            sizes,tr_sc,va_sc=learning_curve(clf,X_sc,y,cv=5,train_sizes=np.linspace(0.1,1.0,8),scoring='f1_weighted',n_jobs=-1)
            tr_m=tr_sc.mean(1); va_m=va_sc.mean(1)
            ax.plot(sizes,tr_m,'o-',color='#58a6ff',lw=2,label='Train')
            ax.plot(sizes,va_m,'s-',color='#ff7b72',lw=2,label='Val')
            ax.fill_between(sizes,tr_sc.min(1),tr_sc.max(1),alpha=0.1,color='#58a6ff')
            ax.fill_between(sizes,va_sc.min(1),va_sc.max(1),alpha=0.1,color='#ff7b72')
            ax.set_title(mn,fontsize=9,fontweight='bold',color='#58a6ff')
            ax.set_xlabel('Training Size',fontsize=8); ax.set_ylabel('F1 Score',fontsize=8)
            ax.legend(fontsize=7,facecolor='#161b22',edgecolor='#30363d',labelcolor='#c9d1d9')
            ax.grid(True,alpha=0.2); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

def show_reg_results(df_r, preds, X_sc, y, show_avp, show_lc):
    best = df_r['R² Score'].idxmax()
    cols = st.columns(4)
    for col,m in zip(cols,['MAE','MSE','RMSE','R² Score']):
        col.markdown(f"<div class='mcard'><div class='mval'>{df_r.loc[best,m]}</div><div class='mname'>{m}<br><span style='color:#ff9500;font-size:0.85em;'>{best}</span></div></div>",unsafe_allow_html=True)
    st.markdown(f"<div class='best'> <b>Best Model: {best}</b> · R²={df_r.loc[best,'R² Score']:.4f} · RMSE={df_r.loc[best,'RMSE']:.4f}</div>",unsafe_allow_html=True)

    st.markdown("####  All Models Comparison")
    st.dataframe(df_r.style
        .highlight_max(subset=['R² Score'],color='#1a4731')
        .highlight_min(subset=['MAE','MSE','RMSE'],color='#1a3a1a')
        .set_properties(**{'text-align':'center'}), use_container_width=True)

    st.markdown("####  Metric Comparison")
    fig, axes = plt.subplots(1,4,figsize=(16,3.5)); fig.patch.set_facecolor('#0d1117')
    for ax,m in zip(axes,['MAE','MSE','RMSE','R² Score']):
        vals=df_r[m]; bars=ax.bar(vals.index,vals.values,color=COLORS,edgecolor='#0d1117',width=0.5)
        ax.set_title(m,fontsize=10,fontweight='bold')
        ax.set_xticklabels(vals.index,rotation=30,ha='right',fontsize=7)
        for bar,v in zip(bars,vals.values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.01,f'{v:.3f}',ha='center',va='bottom',fontsize=7,color='#e6edf3')
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_avp:
        st.markdown("####  Actual vs Predicted")
        fig, axes = plt.subplots(1,4,figsize=(20,4)); fig.patch.set_facecolor('#0d1117')
        for ax,((mn,(yp,yte)),color) in zip(axes, zip(preds.items(),COLORS)):
            r2=r2_score(yte,yp)
            ax.scatter(yte,yp,alpha=0.3,color=color,s=10,edgecolors='none')
            mn2=min(yte.min(),yp.min()); mx=max(yte.max(),yp.max())
            ax.plot([mn2,mx],[mn2,mx],'--',color='#e6edf3',lw=1.5)
            ax.set_title(f'{mn}\nR²={r2:.3f}',fontsize=9,fontweight='bold',color='#ff9500')
            ax.set_xlabel('Actual',fontsize=8); ax.set_ylabel('Predicted',fontsize=8)
            ax.grid(True,alpha=0.15); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_lc:
        st.markdown("####  Learning Curves")
        fig, axes = plt.subplots(1,4,figsize=(20,4)); fig.patch.set_facecolor('#0d1117')
        for ax,(mn,mk) in zip(axes,REG_MODELS.items()):
            reg=mk()
            sizes,tr_sc,va_sc=learning_curve(reg,X_sc,y,cv=5,train_sizes=np.linspace(0.1,1.0,8),scoring='r2',n_jobs=-1)
            tr_m=tr_sc.mean(1); va_m=va_sc.mean(1)
            ax.plot(sizes,tr_m,'o-',color='#ff9500',lw=2,label='Train')
            ax.plot(sizes,va_m,'s-',color='#ff7b72',lw=2,label='Val')
            ax.fill_between(sizes,tr_sc.min(1),tr_sc.max(1),alpha=0.1,color='#ff9500')
            ax.fill_between(sizes,va_sc.min(1),va_sc.max(1),alpha=0.1,color='#ff7b72')
            ax.set_title(mn,fontsize=9,fontweight='bold',color='#ff9500')
            ax.set_xlabel('Training Size',fontsize=8); ax.set_ylabel('R² Score',fontsize=8)
            ax.legend(fontsize=7,facecolor='#161b22',edgecolor='#30363d',labelcolor='#c9d1d9')
            ax.grid(True,alpha=0.2); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

with st.sidebar:
    st.markdown("##  ML Dashboard")
    st.markdown("---")
    task = st.radio("**Task Type**", [" Classification", " Regression"])
    st.markdown("---")
    ds_source = st.radio("**Dataset Source**", [" Built-in Dataset", " Upload CSV"])
    st.markdown("---")

    uploaded_file = None; target_col = None; ds_name = None

    if ds_source == " Built-in Dataset":
        if "Classification" in task:
            ds_name = st.selectbox("**Dataset**", list(CLF_LOADERS.keys()))
        else:
            ds_name = st.selectbox("**Dataset**", list(REG_LOADERS.keys()))
    else:
        uploaded_file = st.file_uploader("**Upload CSV**", type=['csv'])
        if uploaded_file:
            _tmp = pd.read_csv(uploaded_file); uploaded_file.seek(0)
            target_col = st.selectbox("**Target Column**", _tmp.columns.tolist())

    st.markdown("---")
    show_cm  = st.checkbox("Confusion Matrices",  value=True)
    show_roc = st.checkbox("ROC / AUC Curves",    value=True)
    show_lc  = st.checkbox("Learning Curves",     value=True)
    show_avp = st.checkbox("Actual vs Predicted", value=True)
    st.markdown("---")
    st.markdown("<p style='color:#8b949e;font-size:0.8em;'>B.Tech AI & DS · MBM University<br>Jodhpur, Rajasthan · 2026</p>",unsafe_allow_html=True)

st.markdown("""<div class='hero'>
  <h1> ML MODEL COMPARISON DASHBOARD</h1>
  <p>Classification &amp; Regression &nbsp;|&nbsp; Built-in + Custom CSV &nbsp;|&nbsp; 4 Algorithms &nbsp;|&nbsp; All Metrics</p>
</div>""", unsafe_allow_html=True)

if ds_source == " Upload CSV":
    if uploaded_file is None:
        st.markdown("""<div class='upload-hint'>
          <h3 style='color:#58a6ff;margin:0;'>📂 Upload Your Own CSV Dataset</h3>
          <p style='color:#8b949e;margin:10px 0 0 0;'>
            Use the sidebar → Upload CSV → Select your target column<br>
            EDA stats, all model comparisons and plots will run automatically!
          </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    df_raw = pd.read_csv(uploaded_file)
    task_type = "Classification" if "Classification" in task else "Regression"

    show_eda(df_raw, target_col, task_type)
    st.markdown("---")

    try:
        df_c = df_raw.copy()
        for col in df_c.select_dtypes(include='object').columns:
            if col != target_col:
                df_c[col] = LabelEncoder().fit_transform(df_c[col].astype(str))
        df_c = df_c.fillna(df_c.median(numeric_only=True))
        X = df_c.drop(columns=[target_col]).values

        if task_type == "Classification":
            le = LabelEncoder(); y = le.fit_transform(df_c[target_col].astype(str))
            tnames = [str(c) for c in le.classes_]
            st.markdown(f"<div style='font-size:1.3em;font-weight:700;color:#58a6ff;margin:24px 0 12px;'>🔵 Classification — {uploaded_file.name}</div>",unsafe_allow_html=True)
            with st.spinner("Training all classifiers on your dataset..."):
                df_r,preds,tnames,is_bin,X_sc,ya = run_custom_clf(X,y,tnames)
            show_clf_results(df_r,preds,tnames,is_bin,X_sc,ya,show_cm,show_roc,show_lc)
        else:
            y = pd.to_numeric(df_c[target_col],errors='coerce').fillna(0).values
            st.markdown(f"<div style='font-size:1.3em;font-weight:700;color:#ff9500;margin:24px 0 12px;'>🟠 Regression — {uploaded_file.name}</div>",unsafe_allow_html=True)
            with st.spinner("Training all regressors on your dataset..."):
                df_r,preds,X_sc,ya = run_custom_reg(X,y)
            show_reg_results(df_r,preds,X_sc,ya,show_avp,show_lc)

    except Exception as e:
        st.error(f"❌ Error: {e}\n\nTip: Make sure features are mostly numeric and the target column is correct.")

else:
    if "Classification" in task:
        data = CLF_LOADERS[ds_name](as_frame=True)
        df_b = data.frame.copy(); df_b['target'] = data.target
        show_eda(df_b, 'target', "Classification")
        st.markdown("---")
        st.markdown(f"<div style='font-size:1.3em;font-weight:700;color:#58a6ff;margin:24px 0 12px;'>🔵 Classification — {ds_name}</div>",unsafe_allow_html=True)
        with st.spinner(f"Training all models on {ds_name}..."):
            df_r,preds,tnames,is_bin,X_sc,y = run_clf(ds_name)
        show_clf_results(df_r,preds,tnames,is_bin,X_sc,y,show_cm,show_roc,show_lc)
    else:
        data = REG_LOADERS[ds_name](as_frame=True)
        df_b = data.frame.copy(); df_b['target'] = data.target
        show_eda(df_b, 'target', "Regression")
        st.markdown("---")
        st.markdown(f"<div style='font-size:1.3em;font-weight:700;color:#ff9500;margin:24px 0 12px;'>🟠 Regression — {ds_name}</div>",unsafe_allow_html=True)
        with st.spinner(f"Training all models on {ds_name}..."):
            df_r,preds,X_sc,y = run_reg(ds_name)
        show_reg_results(df_r,preds,X_sc,y,show_avp,show_lc)
