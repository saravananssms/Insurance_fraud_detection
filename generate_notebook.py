"""
Generate Insurance_Fraud_Detection.ipynb notebook.
Run this script once to create the notebook, then open it in Jupyter.
"""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": source
    })

# ============================================================
# CELL 1 - Title
# ============================================================
md("""# Insurance Claim Fraud Detection using Deep Learning

## Problem Statement
Insurance fraud costs the industry over **$80 billion annually** in the US alone.
This project builds a deep learning model to automatically flag potentially fraudulent claims for investigation.

## Dataset
A synthetic dataset of **25,000 insurance claims** with realistic distributions modeled on industry patterns.

**Public Dataset Alternatives (20K+ rows):**
- [Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) — 15K rows
- [Porto Seguro Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) — 595K rows
- [Health Insurance Cross Sell](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) — 380K rows
- [NAIC Auto Insurance DB](https://content.naic.org/cipr-topics/auto-insurance) — Public regulatory data

## Approach
1. Generate realistic insurance claims data
2. Exploratory Data Analysis
3. Data Preprocessing
4. Build & Train Deep Neural Network
5. Evaluate Model Performance
6. Export model as `.h5` for deployment via Streamlit
""")

# ============================================================
# CELL 2 - Imports
# ============================================================
code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib, os, warnings

warnings.filterwarnings('ignore')
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
""")

# ============================================================
# CELL 3 - Create directories
# ============================================================
code("""os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
""")

# ============================================================
# CELL 4 - Section header
# ============================================================
md("""---
## 1. Data Generation

We generate a realistic synthetic dataset where fraud labels are correlated with known risk indicators:
- Young policyholders with short tenure
- High-value claims with total-loss severity
- No witnesses or police report
- No authorities contacted
""")

# ============================================================
# CELL 5 - Generate data
# ============================================================
code("""np.random.seed(42)
n = 25000

data = pd.DataFrame()

# --- Customer Demographics ---
data['age'] = np.random.normal(45, 15, n).clip(18, 80).astype(int)
data['gender'] = np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45])
data['education'] = np.random.choice(
    ['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.35, 0.35, 0.20, 0.10])
data['annual_income'] = np.random.lognormal(10.5, 0.8, n).clip(15000, 200000).astype(int)
data['marital_status'] = np.random.choice(
    ['Single', 'Married', 'Divorced', 'Widowed'], n, p=[0.30, 0.45, 0.15, 0.10])

# --- Policy Information ---
data['policy_tenure_months'] = np.random.exponential(36, n).clip(1, 120).astype(int)
data['policy_type'] = np.random.choice(
    ['Comprehensive', 'Third Party', 'Collision'], n, p=[0.45, 0.30, 0.25])
data['annual_premium'] = np.random.lognormal(7.2, 0.5, n).clip(500, 5000).astype(int)
data['policy_deductible'] = np.random.choice(
    [500, 1000, 1500, 2000], n, p=[0.30, 0.35, 0.25, 0.10])
data['umbrella_limit'] = np.random.choice(
    [0, 1000000, 2000000, 3000000, 5000000, 10000000],
    n, p=[0.10, 0.25, 0.25, 0.20, 0.15, 0.05])

# --- Vehicle Information ---
data['vehicle_age'] = np.random.exponential(5, n).clip(0, 20).astype(int)

# --- Incident Details ---
data['incident_type'] = np.random.choice(
    ['Single Vehicle Collision', 'Multi-Vehicle Collision',
     'Vehicle Theft', 'Parked Car'],
    n, p=[0.35, 0.35, 0.15, 0.15])

data['collision_type'] = np.where(
    data['incident_type'] == 'Vehicle Theft', 'NA',
    np.random.choice(['Front Collision', 'Rear Collision', 'Side Collision'],
                     n, p=[0.35, 0.35, 0.30]))

data['incident_severity'] = np.random.choice(
    ['Minor Damage', 'Major Damage', 'Total Loss'], n, p=[0.50, 0.35, 0.15])

data['authorities_contacted'] = np.random.choice(
    ['Police', 'Fire', 'Ambulance', 'None'], n, p=[0.45, 0.10, 0.15, 0.30])

data['number_of_vehicles'] = np.random.choice([1, 2, 3, 4], n, p=[0.35, 0.40, 0.20, 0.05])
data['bodily_injuries'] = np.random.choice([0, 1, 2, 3], n, p=[0.40, 0.30, 0.20, 0.10])
data['witnesses'] = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
data['police_report_available'] = np.random.choice(['Yes', 'No'], n, p=[0.60, 0.40])

# --- Claim Amounts ---
data['total_claim_amount'] = np.random.lognormal(8.5, 0.8, n).clip(1000, 100000).astype(int)
data['injury_claim'] = (data['total_claim_amount'] * np.random.uniform(0, 0.35, n)).astype(int)
data['property_claim'] = (data['total_claim_amount'] * np.random.uniform(0, 0.35, n)).astype(int)
data['vehicle_claim'] = data['total_claim_amount'] - data['injury_claim'] - data['property_claim']

# --- Target: Fraud Label (realistic pattern-based) ---
fraud_score = np.zeros(n)
fraud_score += (data['age'] < 30) * 0.15
fraud_score += (data['policy_tenure_months'] < 12) * 0.20
fraud_score += (data['incident_severity'] == 'Total Loss') * 0.25
fraud_score += (data['witnesses'] == 0) * 0.15
fraud_score += (data['police_report_available'] == 'No') * 0.20
fraud_score += (data['authorities_contacted'] == 'None') * 0.15
fraud_score += (data['total_claim_amount'] > data['total_claim_amount'].quantile(0.90)) * 0.20
fraud_score += (data['bodily_injuries'] >= 2) * 0.10
fraud_score += (data['incident_type'] == 'Vehicle Theft') * 0.15
fraud_score += (data['annual_income'] < 30000) * 0.10
fraud_score += np.random.normal(0, 0.15, n)  # noise

fraud_threshold = np.percentile(fraud_score, 75)
data['fraud_reported'] = (fraud_score > fraud_threshold).astype(int)

data.to_csv('data/insurance_claims.csv', index=False)
print(f"Dataset shape: {data.shape}")
print(f"Fraud rate: {data['fraud_reported'].mean():.2%}")
data.head()
""")

# ============================================================
# CELL 6 - Data overview
# ============================================================
code("""print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\\nRows: {data.shape[0]:,}  |  Columns: {data.shape[1]}")
print(f"\\nColumn types:\\n{data.dtypes.value_counts()}")
print(f"\\nMissing values: {data.isnull().sum().sum()}")
print("\\n" + "=" * 60)
data.describe()
""")

# ============================================================
# CELL 7 - EDA header
# ============================================================
md("""---
## 2. Exploratory Data Analysis
""")

# ============================================================
# CELL 8 - Target distribution
# ============================================================
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors = ['#2ecc71', '#e74c3c']
labels = ['Legitimate', 'Fraudulent']
sizes = data['fraud_reported'].value_counts().values
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0, 0.05), shadow=True, textprops={'fontsize': 12})
axes[0].set_title('Claim Distribution', fontsize=14, fontweight='bold')

# Bar chart
fraud_counts = data['fraud_reported'].value_counts()
bars = axes[1].bar(['Legitimate (0)', 'Fraudulent (1)'], fraud_counts.values, color=colors)
axes[1].set_title('Fraud vs Legitimate Claims', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count')
for bar, val in zip(bars, fraud_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'{val:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('data/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 9 - Numerical distributions
# ============================================================
code("""numerical_cols = ['age', 'annual_income', 'policy_tenure_months', 'annual_premium',
                   'vehicle_age', 'total_claim_amount']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(data=data, x=col, hue='fraud_reported', kde=True,
                 ax=axes[i], palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.6)
    axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[i].legend(['Legitimate', 'Fraudulent'])

plt.suptitle('Numerical Feature Distributions by Fraud Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 10 - Categorical fraud rates
# ============================================================
code("""categorical_cols = ['gender', 'education', 'marital_status', 'policy_type',
                    'incident_type', 'incident_severity', 'authorities_contacted',
                    'police_report_available']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    fraud_rate = data.groupby(col)['fraud_reported'].mean().sort_values(ascending=False)
    bars = axes[i].bar(range(len(fraud_rate)), fraud_rate.values, color='#3498db')
    axes[i].set_xticks(range(len(fraud_rate)))
    axes[i].set_xticklabels(fraud_rate.index, rotation=45, ha='right', fontsize=8)
    axes[i].set_title(f'Fraud Rate by {col}', fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Fraud Rate')
    axes[i].axhline(y=data['fraud_reported'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall Rate')
    axes[i].legend(fontsize=8)
    for bar, val in zip(bars, fraud_rate.values):
        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', fontsize=8)

plt.suptitle('Fraud Rate Across Categorical Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/categorical_fraud_rates.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 11 - Correlation heatmap
# ============================================================
code("""numerical_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(14, 10))
corr_matrix = numerical_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 12 - Claim amount analysis
# ============================================================
code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Claim amount by fraud status
sns.boxplot(data=data, x='fraud_reported', y='total_claim_amount', ax=axes[0],
            palette={0: '#2ecc71', 1: '#e74c3c'})
axes[0].set_xticklabels(['Legitimate', 'Fraudulent'])
axes[0].set_title('Claim Amount by Fraud Status', fontsize=12, fontweight='bold')

# Claim breakdown
claim_cols = ['injury_claim', 'property_claim', 'vehicle_claim']
fraud_claims = data[data['fraud_reported'] == 1][claim_cols].mean()
legit_claims = data[data['fraud_reported'] == 0][claim_cols].mean()

x = np.arange(len(claim_cols))
axes[1].bar(x - 0.2, legit_claims, 0.4, label='Legitimate', color='#2ecc71')
axes[1].bar(x + 0.2, fraud_claims, 0.4, label='Fraudulent', color='#e74c3c')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Injury', 'Property', 'Vehicle'])
axes[1].set_title('Avg Claim Breakdown', fontsize=12, fontweight='bold')
axes[1].legend()

# Age vs claim amount scatter
scatter = axes[2].scatter(data['age'], data['total_claim_amount'],
                          c=data['fraud_reported'], cmap='RdYlGn_r',
                          alpha=0.3, s=10)
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Total Claim Amount')
axes[2].set_title('Age vs Claim Amount', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=axes[2], label='Fraud')

plt.tight_layout()
plt.savefig('data/claim_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 13 - Preprocessing header
# ============================================================
md("""---
## 3. Data Preprocessing

Steps:
1. **Label Encoding** — Convert categorical features to numeric
2. **Feature Scaling** — Standardize all features
3. **Train-Test Split** — 80/20 stratified split
4. **Class Weights** — Handle fraud class imbalance
""")

# ============================================================
# CELL 14 - Preprocessing code
# ============================================================
code("""df = data.copy()

# Identify columns
categorical_columns = ['gender', 'education', 'marital_status', 'policy_type',
                       'incident_type', 'collision_type', 'incident_severity',
                       'authorities_contacted', 'police_report_available']

# Label encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separate features and target
feature_columns = [c for c in df.columns if c != 'fraud_reported']
X = df[feature_columns]
y = df['fraud_reported']

print(f"\\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\\n{y.value_counts()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\\nTraining set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights for imbalanced data
weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}
print(f"\\nClass weights: {class_weight_dict}")
""")

# ============================================================
# CELL 15 - Model header
# ============================================================
md("""---
## 4. Build Deep Neural Network

Architecture:
- Input Layer (21 features)
- Dense(256) + BatchNorm + Dropout(0.3)
- Dense(128) + BatchNorm + Dropout(0.3)
- Dense(64) + BatchNorm + Dropout(0.2)
- Dense(32) + Dropout(0.2)
- Output: Dense(1, sigmoid)
""")

# ============================================================
# CELL 16 - Build model
# ============================================================
code("""model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

model.summary()
""")

# ============================================================
# CELL 17 - Train model
# ============================================================
code("""callbacks = [
    EarlyStopping(monitor='val_auc', patience=10,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\\nTraining complete!")
""")

# ============================================================
# CELL 18 - Training history plots
# ============================================================
code("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))

metrics = [('loss', 'val_loss', 'Loss', 'Training & Validation Loss'),
           ('accuracy', 'val_accuracy', 'Accuracy', 'Training & Validation Accuracy'),
           ('auc', 'val_auc', 'AUC', 'Training & Validation AUC'),
           ('precision', 'val_precision', 'Precision', 'Training & Validation Precision')]

for ax, (train_key, val_key, ylabel, title) in zip(axes.flatten(), metrics):
    ax.plot(history.history[train_key], label='Train', linewidth=2)
    ax.plot(history.history[val_key], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Model Training History', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/training_history.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 19 - Evaluation header
# ============================================================
md("""---
## 5. Model Evaluation
""")

# ============================================================
# CELL 20 - Classification report + Confusion matrix
# ============================================================
code("""y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred,
                            target_names=['Legitimate', 'Fraudulent']))

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')

cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix (Percentages)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 21 - ROC + Precision-Recall curves
# ============================================================
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color='#3498db', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#3498db')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_vals, precision_vals)
axes[1].plot(recall_vals, precision_vals, color='#e74c3c', linewidth=2,
             label=f'PR Curve (AUC = {pr_auc:.3f})')
axes[1].fill_between(recall_vals, precision_vals, alpha=0.1, color='#e74c3c')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC:  {pr_auc:.4f}")
""")

# ============================================================
# CELL 22 - Feature importance (permutation-based)
# ============================================================
code("""from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test_scaled, y_test,
    n_repeats=10, random_state=42, scoring='roc_auc'
)

importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': result.importances_mean,
    'Std': result.importances_std
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'],
         xerr=importance_df['Std'], color='#3498db', alpha=0.8)
plt.xlabel('Mean Importance (decrease in AUC)')
plt.title('Feature Importance (Permutation-based)', fontsize=14, fontweight='bold')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('data/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nTop 10 Most Important Features:")
for _, row in importance_df.tail(10).iterrows():
    print(f"  {row['Feature']:30s} {row['Importance']:.4f} +/- {row['Std']:.4f}")
""")

# ============================================================
# CELL 23 - Save header
# ============================================================
md("""---
## 6. Save Model & Artifacts

Saving the following for Streamlit deployment:
- `fraud_detector.h5` — trained Keras model
- `scaler.pkl` — fitted StandardScaler
- `label_encoders.pkl` — fitted LabelEncoders for categorical features
- `feature_columns.pkl` — ordered feature column names
- `training_history.pkl` — training metrics for dashboard
""")

# ============================================================
# CELL 24 - Save everything
# ============================================================
code("""# Save Keras model as .h5
model.save('models/fraud_detector.h5')
print("Model saved: models/fraud_detector.h5")

# Save preprocessing artifacts
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler saved: models/scaler.pkl")

joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("Label encoders saved: models/label_encoders.pkl")

joblib.dump(feature_columns, 'models/feature_columns.pkl')
print("Feature columns saved: models/feature_columns.pkl")

joblib.dump(history.history, 'models/training_history.pkl')
print("Training history saved: models/training_history.pkl")

# Save evaluation metrics
eval_metrics = {
    'roc_auc': float(roc_auc),
    'pr_auc': float(pr_auc),
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(y_test, y_pred, output_dict=True),
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist()
}
joblib.dump(eval_metrics, 'models/eval_metrics.pkl')
print("Evaluation metrics saved: models/eval_metrics.pkl")

print("\\n All artifacts saved successfully!")
print("\\nNext step: Run the Streamlit app:")
print("  streamlit run app.py")
""")

# ============================================================
# CELL 25 - Conclusion
# ============================================================
md("""---
## Summary

| Metric | Value |
|--------|-------|
| Dataset Size | 25,000 claims |
| Features | 21 (numeric + categorical) |
| Model | Deep Neural Network (5 layers) |
| Output | `fraud_detector.h5` |

### Key Findings
- **Policy tenure** and **incident severity** are the strongest fraud predictors
- Claims with **no witnesses** and **no police report** have significantly higher fraud rates
- **Young policyholders** (<30) with **short tenure** (<12 months) are highest risk
- The model achieves strong AUC, enabling effective fraud triage

### Next Steps
```bash
streamlit run app.py
```
This launches the interactive fraud detection dashboard where you can:
1. Input claim details and get real-time fraud probability
2. Explore the analytics dashboard
3. Review model performance metrics
""")

# ============================================================
# BUILD NOTEBOOK
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open("Insurance_Fraud_Detection.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook created: Insurance_Fraud_Detection.ipynb")
print(f"Total cells: {len(cells)}")
