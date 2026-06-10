# Fetal Health Risk Classification — Machine Learning for Child Safety
## XGBoost · SMOTE · SHAP Explainability · 98% Accuracy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end machine learning pipeline for classifying fetal health status (Normal / Suspect / Pathological) from cardiotocography (CTG) measurements. This project addresses class imbalance, delivers high-accuracy multi-class classification, and uses SHAP explainability to make model decisions interpretable for clinical stakeholders.

---

## Table of Contents

- [Clinical Background](#clinical-background)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Model Explainability (SHAP)](#model-explainability-shap)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Clinical Implications](#clinical-implications)
- [References](#references)

---

## Clinical Background

Cardiotocography (CTG) is a standard technique used during pregnancy to monitor fetal heart rate and uterine contractions. Obstetricians use CTG traces to classify fetal wellbeing as Normal, Suspect, or Pathological — but manual interpretation is subjective, time-consuming, and prone to inter-observer variability, particularly in resource-constrained healthcare settings.

In Bangladesh, neonatal mortality remains a critical public health challenge (23 per 1,000 live births, UNICEF 2023). Automating the classification of CTG data using machine learning can support clinicians in high-volume settings where specialist availability is limited, potentially preventing adverse neonatal outcomes through timely identification of fetal distress.

---

## Problem Statement

**Task:** Multi-class classification  
**Target:** Fetal health status — `1 = Normal`, `2 = Suspect`, `3 = Pathological`  
**Challenge:** Severe class imbalance (78% Normal, 13.5% Suspect, 8.3% Pathological)  
**Constraint:** High recall for Suspect and Pathological classes is clinically critical — false negatives are life-threatening  

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Records** | 2,126 CTG examinations |
| **Features** | 21 physiological measurements |
| **Target** | Fetal health class (1/2/3) |
| **Class distribution** | 78.1% Normal / 13.5% Suspect / 8.3% Pathological |
| **Source** | Modelled on UCI CTG dataset (Ayres de Campos et al., 2000) |
| **File** | `data/fetal_health_bangladesh.csv` |

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| `baseline_fhr` | Baseline fetal heart rate (bpm) |
| `accelerations` | Number of accelerations per second |
| `fetal_movement` | Number of fetal movements per second |
| `uterine_contractions` | Number of uterine contractions per second |
| `light_decelerations` | Number of light decelerations per second |
| `severe_decelerations` | Number of severe decelerations per second |
| `prolonged_decelerations` | Number of prolonged decelerations per second |
| `abnormal_short_term_variability` | % of time with abnormal short-term variability |
| `mean_short_term_variability` | Mean value of short-term variability |
| `percentage_long_term_variability` | % of time with long-term variability |
| `mean_long_term_variability` | Mean value of long-term variability |
| `histogram_*` | 10 histogram-derived features (width, min, max, peaks, etc.) |

---

## Methodology

### Pipeline Overview

```
Raw CTG Data
     │
     ▼
Data Preprocessing
  (type checking, validation, descriptive stats)
     │
     ▼
Exploratory Data Analysis
  (class distribution, feature distributions, correlation)
     │
     ▼
Train/Test Split (80/20, stratified)
     │
     ▼
Class Imbalance Handling — SMOTE
  (synthetic minority oversampling on training set only)
     │
     ▼
Model Training
  ├── Logistic Regression (baseline)
  ├── Random Forest (ensemble)
  └── XGBoost (gradient boosting — best performer)
     │
     ▼
Evaluation
  ├── Confusion matrix (normalized)
  ├── ROC curves (one-vs-rest, per class)
  ├── Classification report (precision, recall, F1)
  └── 5-fold cross-validation (stratified)
     │
     ▼
SHAP Explainability
  (TreeExplainer, summary plots, force plots)
     │
     ▼
Model Persistence (joblib)
```

### Class Imbalance Handling

SMOTE (Synthetic Minority Over-sampling Technique) was applied exclusively to the training set to avoid data leakage. The original class distribution (78:13.5:8.3) was balanced to 33:33:33 for training, improving minority class recall without biasing test set evaluation.

### Model Selection Rationale

**XGBoost** was selected as the primary model for the following reasons:
- Handles multiclass natively via `softprob` objective
- Robust to feature scale differences without normalization
- Built-in regularization (L1/L2) reduces overfitting
- Compatible with SHAP TreeExplainer for efficient exact Shapley values
- Gradient boosting consistently outperforms linear models on tabular clinical data

---

## Results

### Classification Performance

| Metric | Normal | Suspect | Pathological | Macro Avg |
|--------|:------:|:-------:|:------------:|:---------:|
| Precision | 0.99 | 0.93 | 0.94 | 0.95 |
| Recall | 0.99 | 0.91 | 0.94 | 0.95 |
| F1-Score | 0.99 | 0.92 | 0.94 | **0.95** |
| Support | 333 | 58 | 35 | 426 |

**Overall Accuracy: 98%**

### Cross-Validation (5-Fold Stratified)

| Metric | Score |
|--------|-------|
| CV F1-Macro (mean) | **0.997** |
| CV F1-Macro (std) | ±0.002 |

### Model Comparison

| Model | Macro F1 |
|-------|:--------:|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.93 |
| **XGBoost** | **0.952** |

### ROC-AUC (One-vs-Rest)

| Class | AUC |
|-------|:---:|
| Normal | ~0.999 |
| Suspect | ~0.990 |
| Pathological | ~0.995 |

---

## Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values are computed using the TreeExplainer, which provides exact Shapley values for tree-based models without sampling approximation.

**Top predictive features (by mean |SHAP value|):**
1. `abnormal_short_term_variability` — strongest predictor of pathological status
2. `prolonged_decelerations` — high-risk marker for fetal distress
3. `percentage_long_term_variability` — discriminates suspect from normal
4. `mean_long_term_variability` — correlated with CNS control of heart rate
5. `accelerations` — inversely associated with pathological class

SHAP plots are located in `outputs/figures/fig4_shap_summary.png`.

---

## Visualizations

| Figure | Description |
|--------|-------------|
| `fig1_confusion_matrix.png` | Normalized confusion matrix with raw counts |
| `fig2_roc_curves.png` | ROC curves for all three classes (one-vs-rest) |
| `fig3_feature_importance.png` | XGBoost gain-based feature importance (top 15) |
| `fig4_shap_summary.png` | SHAP mean absolute value summary (per class) |
| `fig5_smote_balancing.png` | Class distribution before and after SMOTE |
| `fig6_feature_distributions.png` | Top feature distributions by health class |
| `fig7_model_comparison.png` | F1-macro score comparison across three models |

---

## Project Structure

```
child-health-fetal-ml/
├── data/
│   └── fetal_health_bangladesh.csv         # CTG dataset (2,126 records)
├── notebooks/
│   └── fetal_health_classification.ipynb   # Full ML pipeline notebook
├── models/
│   └── xgboost_fetal_health.pkl            # Serialized trained model
├── outputs/
│   ├── figures/                            # All visualization outputs
│   │   ├── fig1_confusion_matrix.png
│   │   ├── fig2_roc_curves.png
│   │   ├── fig3_feature_importance.png
│   │   ├── fig4_shap_summary.png
│   │   ├── fig5_smote_balancing.png
│   │   ├── fig6_feature_distributions.png
│   │   └── fig7_model_comparison.png
│   └── reports/
│       ├── model_metrics.json              # All numeric results
│       └── model_card.md                   # Model card for transparency
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/maliksabihatasnim/child-health-fetal-ml.git
cd child-health-fetal-ml

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/fetal_health_classification.ipynb

# 5. (Optional) Load the saved model
import joblib
model = joblib.load('models/xgboost_fetal_health.pkl')
```

---

## Clinical Implications

1. **Decision support tool**: The model achieves 94% recall on Pathological cases — in clinical deployment, it could serve as a screening layer to flag high-risk CTGs for urgent specialist review.
2. **Interpretability first**: SHAP values make model decisions auditable by clinicians — a non-negotiable requirement for AI tools in healthcare.
3. **Resource-constrained settings**: Automated CTG classification could significantly reduce diagnostic delays in district hospitals without specialist obstetricians.
4. **Limitations**: This model requires prospective clinical validation on real-world CTG data before clinical deployment. It should not replace clinical judgment.

---

## References

1. Ayres de Campos, D. et al. (2000). SisPorto 2.0: A program for automated analysis of cardiotocograms. *Journal of Maternal-Fetal Medicine*, 9(5), 311–318.
2. UNICEF Bangladesh (2023). *State of the World's Children — Bangladesh Country Profile*.
3. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
4. Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
5. Chawla, N.V. et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Developed as part of a data science portfolio targeting child health and public health research applications in Bangladesh.*
