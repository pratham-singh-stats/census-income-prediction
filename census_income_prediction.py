# ============================================================
# Census Income Prediction using Logistic Regression
# Binary Classification: Income >50K or ≤50K
# Author: Pratham Singh
# Dataset: UCI Adult Census Income Dataset
# ============================================================

# ── 1. Import Libraries ───────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 60)
print("Census Income Prediction — Logistic Regression")
print("=" * 60)

# ── 2. Load Data ──────────────────────────────────────────────
# Dataset: UCI Adult Census Income
# Download: https://archive.ics.uci.edu/ml/datasets/adult
# OR load directly:
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race',
           'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
           'native_country', 'income']

try:
    df = pd.read_csv(url, names=columns, sep=',\s*', engine='python', na_values='?')
    print(f"\n✓ Data loaded from UCI repository")
except Exception:
    # Fallback: simulate dataset with realistic distributions
    np.random.seed(42)
    n = 32561
    df = pd.DataFrame({
        'age': np.random.randint(18, 90, n),
        'workclass': np.random.choice(['Private', 'Self-emp', 'Gov', 'Other'], n,
                                       p=[0.70, 0.10, 0.15, 0.05]),
        'fnlwgt': np.random.randint(10000, 1500000, n),
        'education': np.random.choice(['Bachelors', 'HS-grad', 'Masters', 'Some-college',
                                        'Assoc', 'Doctorate', 'Other'], n,
                                       p=[0.16, 0.32, 0.07, 0.22, 0.07, 0.01, 0.15]),
        'education_num': np.random.randint(1, 16, n),
        'marital_status': np.random.choice(['Married', 'Single', 'Divorced'], n,
                                            p=[0.46, 0.33, 0.21]),
        'occupation': np.random.choice(['Prof-specialty', 'Craft-repair', 'Exec-managerial',
                                         'Adm-clerical', 'Sales', 'Other'], n),
        'relationship': np.random.choice(['Husband', 'Wife', 'Own-child', 'Other'], n),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n,
                                  p=[0.85, 0.10, 0.03, 0.02]),
        'sex': np.random.choice(['Male', 'Female'], n, p=[0.67, 0.33]),
        'capital_gain': np.where(np.random.random(n) > 0.92,
                                  np.random.randint(1, 99999, n), 0),
        'capital_loss': np.where(np.random.random(n) > 0.95,
                                  np.random.randint(1, 4356, n), 0),
        'hours_per_week': np.clip(np.random.normal(40, 12, n).astype(int), 1, 99),
        'native_country': np.random.choice(['United-States', 'Other'], n, p=[0.90, 0.10]),
        'income': np.random.choice(['<=50K', '>50K'], n, p=[0.76, 0.24])
    })
    print("\n✓ Simulated dataset loaded (UCI source unavailable)")

print(f"\nDataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── 3. EDA ────────────────────────────────────────────────────
print("\n=== Exploratory Data Analysis ===")
print(f"\nIncome distribution:\n{df['income'].value_counts(normalize=True).round(3)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution by income
df.groupby('income')['age'].plot(kind='kde', ax=axes[0,0], legend=True)
axes[0,0].set_title('Age Distribution by Income Group')
axes[0,0].set_xlabel('Age')

# Education vs Income
edu_income = df.groupby(['education', 'income']).size().unstack(fill_value=0)
edu_income.plot(kind='bar', ax=axes[0,1], color=['#2E75B6', '#C0392B'])
axes[0,1].set_title('Education Level vs Income')
axes[0,1].tick_params(axis='x', rotation=45)

# Hours per week
df.boxplot(column='hours_per_week', by='income', ax=axes[1,0])
axes[1,0].set_title('Weekly Hours by Income')
axes[1,0].set_xlabel('Income Group')

# Sex vs Income
sex_income = df.groupby(['sex', 'income']).size().unstack(fill_value=0)
sex_income.plot(kind='bar', ax=axes[1,1], color=['#2E75B6', '#C0392B'])
axes[1,1].set_title('Sex vs Income')
axes[1,1].tick_params(axis='x', rotation=0)

plt.suptitle('Census Income — Exploratory Data Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('plots/eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. Preprocessing ──────────────────────────────────────────
print("\n=== Preprocessing ===")

# Drop missing values
df.dropna(inplace=True)
print(f"Rows after dropping NAs: {len(df)}")

# Encode target variable
df['income_binary'] = (df['income'].str.strip() == '>50K').astype(int)

# Encode categorical features
cat_cols = ['workclass', 'education', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'native_country']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
feature_cols = ['age', 'workclass', 'education_num', 'marital_status',
                'occupation', 'relationship', 'sex', 'capital_gain',
                'capital_loss', 'hours_per_week']
X = df[feature_cols]
y = df['income_binary']

print(f"Features used: {feature_cols}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 5. Model Training ─────────────────────────────────────────
print("\n=== Model Training ===")

# Logistic Regression pipeline (scale → fit)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Random Forest (for comparison)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-validation (5-fold):")
lr_cv = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"  Logistic Regression AUC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"  Random Forest AUC:       {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# Fit on full training set
lr_pipeline.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ── 6. Evaluation ─────────────────────────────────────────────
print("\n=== Model Evaluation on Test Set ===")

for name, model in [("Logistic Regression", lr_pipeline),
                     ("Random Forest", rf_model)]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                  target_names=['<=50K', '>50K']))

# ── 7. Confusion Matrix ───────────────────────────────────────
y_pred_lr = lr_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix — Logistic Regression')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.show()

# ── 8. ROC Curve ──────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for name, model in [("Logistic Regression", lr_pipeline),
                     ("Random Forest", rf_model)]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(); plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.show()

# ── 9. Feature Importance ─────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#2E75B6')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.show()

print("\nTop 3 most important features:")
print(importance_df.tail(3)[['Feature', 'Importance']].to_string(index=False))

print("\n✓ Analysis complete. Plots saved to /plots/")
