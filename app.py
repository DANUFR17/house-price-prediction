"""
House Price Predictor — Streamlit App
Notebook 05: Deployment

Cara menjalankan:
  pip install streamlit xgboost shap pandas scikit-learn
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    .main-price {
        font-size: 3rem;
        font-weight: 700;
        color: #1D9E75;
        line-height: 1;
    }
    .price-range {
        font-size: 1rem;
        color: #888;
        margin-top: 4px;
    }
    .shap-pos { color: #1D9E75; font-weight: 600; }
    .shap-neg { color: #D85A30; font-weight: 600; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load & prepare data (cached agar tidak reload setiap kali)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv')

    # Outlier removal
    df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
    df = df[df['SaleCondition'] == 'Normal'].reset_index(drop=True)

    # Feature engineering (sama persis dengan Notebook 02)
    df['TotalSF']           = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms']    = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['HouseAge']          = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    df['WasRemodeled']      = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['TotalPorchSF']      = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['HasPool']           = (df['PoolArea'] > 0).astype(int)
    df['HasGarage']         = (df['GarageArea'] > 0).astype(int)
    df['HasBasement']       = (df['TotalBsmtSF'] > 0).astype(int)

    # Ordinal encoding
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
                'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']:
        if col in df.columns:
            df[col] = df[col].fillna('None').map(quality_map)

    ordinal = {
        'BsmtExposure': {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4},
        'BsmtFinType1': {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},
        'GarageFinish': {'None':0,'Unf':1,'RFn':2,'Fin':3},
        'PavedDrive':   {'N':0,'P':1,'Y':2},
    }
    for col, mapping in ordinal.items():
        if col in df.columns:
            df[col] = df[col].fillna('None').map(mapping)

    drop_cols = ['Id','YearBuilt','YearRemodAdd','YrSold','GarageYrBlt','MoSold',
                 'SaleCondition','1stFlrSF','2ndFlrSF','TotalBsmtSF',
                 'FullBath','HalfBath','BsmtFullBath','BsmtHalfBath',
                 'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y   = np.log1p(df.pop('SalePrice'))
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=['object']).columns.tolist()
    df[num] = df[num].fillna(df[num].median())
    df[cat] = df[cat].fillna('None')
    X = pd.get_dummies(df, columns=cat, drop_first=True)

    return X, y


@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
with st.spinner("Memuat model..."):
    X, y = load_data()
    model, explainer = train_model(X, y)

baseline_usd = np.expm1(explainer.expected_value).item()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.markdown("Masukkan detail rumah di sidebar → model memprediksi harga + penjelasan SHAP")
st.divider()


# ─────────────────────────────────────────────
# Sidebar: Input user
# ─────────────────────────────────────────────
st.sidebar.header("Detail Rumah")
st.sidebar.markdown("---")

st.sidebar.subheader("Ukuran")
total_sf    = st.sidebar.slider("Total luas bangunan (sqft)", 500, 6000, 1800, step=50)
gr_liv_area = st.sidebar.slider("Luas lantai atas (sqft)",   500, 5000, 1500, step=50)
lot_area    = st.sidebar.slider("Luas lahan (sqft)",         1000, 50000, 9000, step=500)
garage_area = st.sidebar.slider("Luas garasi (sqft)",        0, 1500, 400, step=50)

st.sidebar.subheader("Kualitas")
overall_qual = st.sidebar.select_slider(
    "Kualitas keseluruhan", options=[1,2,3,4,5,6,7,8,9,10], value=6
)
kitchen_qual = st.sidebar.selectbox(
    "Kualitas dapur", options=[1,2,3,4,5],
    format_func=lambda x: {1:'Poor',2:'Fair',3:'Average',4:'Good',5:'Excellent'}[x],
    index=2
)
exter_qual = st.sidebar.selectbox(
    "Kualitas eksterior", options=[1,2,3,4,5],
    format_func=lambda x: {1:'Poor',2:'Fair',3:'Average',4:'Good',5:'Excellent'}[x],
    index=2
)

st.sidebar.subheader("Kondisi")
house_age       = st.sidebar.slider("Umur rumah (tahun)", 0, 100, 20)
total_bathrooms = st.sidebar.slider("Total kamar mandi", 1.0, 5.0, 2.0, step=0.5)
has_garage      = st.sidebar.checkbox("Punya garasi", value=True)
has_basement    = st.sidebar.checkbox("Punya basement", value=True)
was_remodeled   = st.sidebar.checkbox("Pernah direnovasi", value=False)


# ─────────────────────────────────────────────
# Buat sample input berdasarkan input user
# ─────────────────────────────────────────────
# Gunakan median dari training set sebagai default, lalu override dengan input user
sample = X.median().to_frame().T.copy()

# Override fitur yang relevan dengan input user
feature_map = {
    'TotalSF':        total_sf,
    'GrLivArea':      gr_liv_area,
    'LotArea':        lot_area,
    'GarageArea':     garage_area,
    'OverallQual':    overall_qual,
    'KitchenQual':    kitchen_qual,
    'ExterQual':      exter_qual,
    'HouseAge':       house_age,
    'TotalBathrooms': total_bathrooms,
    'HasGarage':      int(has_garage),
    'HasBasement':    int(has_basement),
    'WasRemodeled':   int(was_remodeled),
}

for feat, val in feature_map.items():
    if feat in sample.columns:
        sample[feat] = val


# ─────────────────────────────────────────────
# Prediksi & SHAP
# ─────────────────────────────────────────────
pred_log  = model.predict(sample)[0]
pred_usd  = np.expm1(pred_log)
shap_vals = explainer.shap_values(sample)[0]

# Confidence interval sederhana berdasarkan residual standar deviasi
std_log = 0.115  # dari CV RMSE notebook 03
lower   = np.expm1(pred_log - 1.96 * std_log)
upper   = np.expm1(pred_log + 1.96 * std_log)


# ─────────────────────────────────────────────
# Layout utama: 2 kolom
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # ── Harga prediksi ────────────────────────
    st.markdown("### Prediksi harga")
    st.markdown(f'<div class="main-price">${pred_usd:,.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="price-range">95% CI: ${lower:,.0f} – ${upper:,.0f}</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Metric summary ────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("vs rata-rata",  f"${pred_usd - baseline_usd:+,.0f}",
              delta_color="normal")
    m2.metric("Kualitas",       f"{overall_qual}/10")
    m3.metric("Luas total",     f"{total_sf:,} sqft")

    st.divider()

    # ── Top fitur yang mendorong prediksi ─────
    st.markdown("### Faktor pendorong harga")

    feat_names  = sample.columns.tolist()
    shap_series = pd.Series(shap_vals, index=feat_names)
    top_pos     = shap_series.nlargest(5)
    top_neg     = shap_series.nsmallest(3)
    combined    = pd.concat([top_pos, top_neg]).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ['#1D9E75' if v > 0 else '#D85A30' for v in combined.values]
    bars    = ax.barh(combined.index, combined.values,
                      color=colors, height=0.55, edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='#ccc', linewidth=0.8)

    for bar, val in zip(bars, combined.values):
        sign = '+' if val > 0 else ''
        ax.text(val + (0.001 if val > 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f'{sign}{val:.3f}', va='center',
                ha='left' if val > 0 else 'right', fontsize=8)

    ax.set_xlabel('SHAP value (kontribusi ke log-harga)')
    ax.set_title('Fitur yang menaikkan / menurunkan prediksi', fontsize=10, fontweight='500')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


with col2:
    # ── Waterfall explanation ─────────────────
    st.markdown("### Penjelasan prediksi (waterfall)")
    st.caption(f"Baseline: ${baseline_usd:,.0f} → Prediksi: ${pred_usd:,.0f}")

    top10_idx  = np.argsort(np.abs(shap_vals))[::-1][:10]
    top10_feat = [feat_names[i] for i in top10_idx]
    top10_shap = [shap_vals[i]  for i in top10_idx]
    top10_val  = [sample[f].values[0] for f in top10_feat]

    # Waterfall chart manual
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    baseline = float(explainer.expected_value)
    running   = baseline

    for i, (feat, sv, fv) in enumerate(zip(
        reversed(top10_feat), reversed(top10_shap), reversed(top10_val)
    )):
        color = '#1D9E75' if sv > 0 else '#D85A30'
        ax2.barh(i, sv, left=running, color=color,
                 height=0.55, edgecolor='white', linewidth=0.4)

        lbl = f'{feat[:20]}={fv:.0f}' if isinstance(fv, float) else f'{feat[:20]}'
        sign = '+' if sv > 0 else ''
        ax2.text(running + sv + (0.001 if sv > 0 else -0.001), i,
                 f'{sign}{sv:.3f}', va='center',
                 ha='left' if sv > 0 else 'right', fontsize=8)
        running += sv

    ax2.set_yticks(range(10))
    ax2.set_yticklabels(
        [f'{f[:18]}' for f in reversed(top10_feat)], fontsize=8
    )
    ax2.axvline(baseline, color='#aaa', linewidth=1, linestyle='--',
                label=f'baseline={baseline:.2f}')
    ax2.axvline(pred_log, color='#085041', linewidth=1.5,
                label=f'prediksi={pred_log:.2f}')
    ax2.set_xlabel('Nilai log(SalePrice)')
    ax2.set_title('Bagaimana model sampai ke prediksi ini', fontsize=10, fontweight='500')
    ax2.legend(fontsize=8, loc='lower right')
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Insight otomatis ──────────────────────
    st.markdown("### Insight otomatis")

    top1_feat = top10_feat[0]
    top1_shap = top10_shap[0]
    direction = "menaikkan" if top1_shap > 0 else "menurunkan"

    st.info(
        f"Faktor terbesar: **{top1_feat}** {direction} prediksi sebesar "
        f"**${abs(np.expm1(baseline + top1_shap) - baseline_usd):,.0f}**"
    )

    if pred_usd > baseline_usd * 1.3:
        st.success("Rumah ini dinilai signifikan di atas rata-rata pasar.")
    elif pred_usd < baseline_usd * 0.8:
        st.warning("Rumah ini dinilai di bawah rata-rata pasar.")
    else:
        st.info("Harga rumah ini berada di kisaran rata-rata pasar.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost · Dataset: Ames Housing (Kaggle) · "
    "Feature engineering + SHAP interpretability · "
    "CV RMSE ≈ 0.115 (log scale)"
)
