# Model Card: XGBoost Fetal Health Classifier

**Version:** 1.0  
**Date:** June 2026  
**Framework:** XGBoost 3.x, scikit-learn 1.8, Python 3.10+

---

## Model Overview

| Attribute | Detail |
|-----------|--------|
| **Task** | Multi-class classification |
| **Algorithm** | XGBoost (gradient-boosted decision trees) |
| **Input** | 21 cardiotocography (CTG) physiological features |
| **Output** | Fetal health class: Normal (0), Suspect (1), Pathological (2) |
| **Training set size** | 1,700 records (after SMOTE: 3,984 balanced records) |
| **Test set size** | 426 records |

---

## Performance

| Metric | Value |
|--------|-------|
| Test accuracy | 98.1% |
| Macro F1-score | 0.952 |
| CV F1-macro (5-fold) | 0.997 ± 0.002 |
| AUC (Normal) | ~0.999 |
| AUC (Suspect) | ~0.990 |
| AUC (Pathological) | ~0.995 |

---

## Intended Use

- **Primary use:** Research and educational demonstration of ML in fetal health monitoring
- **Intended users:** Data scientists, researchers, clinical informaticists
- **Out-of-scope uses:** Direct clinical deployment without prospective validation; replacement of obstetrician judgment

---

## Training Data

The model was trained on CTG data modelled on the UCI Cardiotocography Dataset. Class imbalance was handled using SMOTE on the training split only. No patient identifiers are included.

---

## Ethical Considerations

- Model should not be deployed clinically without prospective validation on real patient data
- SHAP explainability was implemented to ensure interpretability for clinical stakeholders
- High recall for minority classes (Suspect, Pathological) was prioritized to minimize false negatives — in clinical context, false negatives are life-threatening

---

## Hyperparameters

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
```

---

## Limitations

1. Dataset is synthetic/modelled — real-world performance may differ
2. Not validated on diverse ethnic/geographic populations
3. Model requires all 21 features; partial inputs are not supported in current version
4. Temporal aspects of CTG traces are not captured (static feature representation only)
