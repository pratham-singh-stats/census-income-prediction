# 💰 Census Income Prediction using Logistic Regression

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Statistical%20Classification-2E75B6)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Adult%20Census-orange)

## 📌 Overview
A statistical classification project predicting whether an individual's annual income exceeds **$50,000** using demographic and employment features from the **UCI Adult Census Income dataset**. This project demonstrates end-to-end binary classification — from EDA and preprocessing to model evaluation and interpretation.

---

## 🎯 Objectives
- Explore and visualize demographic predictors of income level
- Build and evaluate a **Logistic Regression** classification model
- Compare performance against Random Forest using AUC-ROC
- Interpret model coefficients and feature importances statistically

---

## 📊 Dataset
| Attribute | Details |
|-----------|---------|
| Source | UCI Machine Learning Repository |
| Observations | ~32,561 records |
| Features | 14 (age, education, occupation, hours/week, etc.) |
| Target | Income: ≤50K or >50K (binary) |
| Class Balance | ~76% ≤50K, ~24% >50K |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Age distributions by income group (KDE plots)
- Education level vs. income (grouped bar charts)
- Hours per week by income (boxplots)
- Gender vs. income distribution

### 2. Preprocessing
- Removed missing values (`?` coded nulls)
- Label-encoded all categorical variables
- Standardized numerical features using `StandardScaler`
- 80/20 stratified train-test split

### 3. Modeling
| Model | CV AUC (5-fold) |
|-------|----------------|
| Logistic Regression | ~0.88 |
| Random Forest | ~0.91 |

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve & AUC-ROC
- Feature Importance (Random Forest)

---

## 📈 Key Findings
- **Education level** and **capital gains** are the strongest predictors of income > 50K
- **Marital status** (married individuals) shows significantly higher income rates
- Logistic Regression achieves **~85% accuracy** with AUC ~0.88 — strong interpretable baseline
- Random Forest outperforms LR slightly (AUC ~0.91) but at the cost of interpretability

---

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Key Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/pratham-singh-stats/census-income-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Create plots directory
mkdir plots

# 4. Run the script
python census_income_prediction.py
```

---

## 📁 Repository Structure
```
03_census_income/
│
├── census_income_prediction.py   # Full pipeline: EDA → model → evaluation
├── README.md                     # This file
└── plots/                        # Output visualizations (generated on run)
    ├── eda_plots.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## 📚 References
- Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. UCI, Irvine.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
