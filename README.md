# 🏠 House Price Prediction — End-to-End ML Project

Proyek machine learning end-to-end untuk memprediksi harga rumah menggunakan dataset Ames Housing (Kaggle). Dibangun dengan pendekatan bertahap dari eksplorasi data hingga deployment aplikasi interaktif.

---

## Struktur Project

```
house-price-ml/
│
├── data/
│   ├── train.csv                  # Dataset asli dari Kaggle
│   ├── X_engineered.csv           # Fitur hasil feature engineering (Notebook 02)
│   └── y_engineered.csv           # Target hasil feature engineering (Notebook 02)
│
├── notebooks/
│   ├── 01-baseline-eda.ipynb      # EDA + baseline model
│   ├── 02-feature-engineering.ipynb
│   ├── 03-ensemble-stacking.ipynb
│   └── 04-shap-interpretability.ipynb
│
├── output/
│   ├── 02a-outlier-detection.png
│   ├── 02b-korelasi-fitur-baru.png
│   ├── 02c-before-after-fe.png
│   ├── 03-benchmark-ensemble.png
│   ├── 04a-shap-bar.png
│   ├── 04b-shap-beeswarm.png
│   ├── 04c-shap-dependence.png
│   ├── 04d-shap-waterfall.png
│   └── 04e-shap-force.html
│
├── app.py                         # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## Notebook Roadmap

| # | Notebook | Topik | Konsep Utama |
|---|---|---|---|
| 01 | Baseline EDA | Eksplorasi data + model pertama | EDA, Linear Regression, Random Forest, cross-validation |
| 02 | Feature Engineering | Transformasi fitur | Outlier removal, feature creation, ordinal encoding |
| 03 | Ensemble & Stacking | Gabungan model | Ridge, Lasso, XGBoost, LightGBM, out-of-fold stacking |
| 04 | SHAP Interpretability | Penjelasan prediksi | TreeExplainer, beeswarm, waterfall, force plot |
| 05 | Deployment | Aplikasi interaktif | Streamlit, real-time prediction + SHAP |

---

## Hasil & Performa Model

| Model | CV RMSE (log) | Keterangan |
|---|---|---|
| Linear Regression (baseline) | ~0.180 | Notebook 01 |
| Random Forest | ~0.145 | Notebook 01 |
| Setelah Feature Engineering | ~0.138 | Notebook 02 |
| XGBoost | ~0.122 | Notebook 03 |
| LightGBM | ~0.120 | Notebook 03 |
| **Stacking (final)** | **~0.115** | **Notebook 03** |

> RMSE dalam log scale. Improvement total dari baseline ke stacking: **~36%**.

---

## Konsep ML yang Dipelajari

### Feature Engineering
- Outlier detection dan removal menggunakan domain knowledge
- Feature creation dari kombinasi fitur yang ada (`TotalSF`, `HouseAge`, dll)
- Perbedaan ordinal vs nominal encoding — dan kapan pakai masing-masing
- Menghapus fitur redundan untuk mengurangi multikolinearitas

### Model & Ensemble
- Regularisasi: Ridge (L2) vs Lasso (L1) dan kapan masing-masing lebih baik
- Gradient boosting: XGBoost (level-wise) vs LightGBM (leaf-wise)
- Bagging vs Boosting vs Stacking — perbedaan konsep dan use case
- Out-of-fold prediction untuk mencegah data leakage di stacking

### Interpretability
- Mengapa `feature_importances_` bisa misleading
- SHAP values: dari game theory (Shapley values) ke ML
- Global interpretation (beeswarm, bar) vs local (waterfall, force plot)
- Cara mengkomunikasikan prediksi model kepada non-technical stakeholder

---

## Cara Menjalankan

### 1. Clone & setup environment

```bash
git clone https://github.com/username/house-price-ml.git
cd house-price-ml

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download dataset

Download `train.csv` dari [Kaggle — House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dan simpan di folder `data/`.

### 3. Jalankan notebook secara berurutan

```bash
jupyter notebook
# Buka notebooks/ dan jalankan 01 → 02 → 03 → 04
```

### 4. Jalankan Streamlit app

```bash
streamlit run app.py
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
shap
streamlit
jupyter
```

---

## Dataset

**Ames Housing Dataset** — berisi 79 fitur deskriptif tentang properti residensial di Ames, Iowa. Digunakan sebagai benchmark klasik untuk regression problems di Kaggle.

- 1,460 baris data training
- Target: `SalePrice` (harga jual rumah dalam USD)
- Sumber: [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## Referensi & Bacaan Lanjutan

Untuk belajar lebih dalam dari project ini:

- [SHAP Documentation](https://shap.readthedocs.io/) — cara kerja SHAP secara matematis
- [XGBoost Paper](https://arxiv.org/abs/1603.02754) — paper original Chen & Guestrin
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/) — gratis online, sangat direkomendasikan
- [Kaggle Winning Solutions](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/discussion) — pelajari approach top scorer

---

*Project ini dibuat sebagai bagian dari perjalanan belajar ML secara mendalam — dari data mentah hingga model yang bisa dijelaskan dan di-deploy.*