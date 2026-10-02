# Insurance Claim Fraud Detection

A deep learning system that detects fraudulent insurance claims using a neural network trained on 25,000+ claim records, deployed via an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Insurance fraud costs the US industry over **$80 billion annually**. This project builds an end-to-end ML pipeline that:

- Generates a realistic 25,000-row insurance claims dataset
- Trains a deep neural network to classify claims as legitimate or fraudulent
- Exports the model as `.h5` for production deployment
- Provides a 3-page Streamlit dashboard for real-time fraud prediction and analytics

## Project Structure

```
insurance_fraud_detection/
├── Insurance_Fraud_Detection.ipynb   # Full ML pipeline (EDA → Train → Export)
├── app.py                            # Streamlit dashboard (3 pages)
├── requirements.txt                  # Python dependencies
├── README.md
├── data/
│   └── insurance_claims.csv          # Generated after running notebook
└── models/
    ├── fraud_detector.h5             # Trained Keras model
    ├── scaler.pkl                    # Fitted StandardScaler
    ├── label_encoders.pkl            # Categorical encoders
    ├── feature_columns.pkl           # Feature column order
    ├── eval_metrics.pkl              # Test set evaluation metrics
    └── training_history.pkl          # Training curves data
```

## Dataset

A synthetic dataset of **25,000 insurance claims** with distributions modeled on real industry patterns.

| Category | Features |
|----------|----------|
| **Customer** | age, gender, education, annual_income, marital_status |
| **Policy** | policy_tenure_months, policy_type, annual_premium, policy_deductible, umbrella_limit |
| **Vehicle** | vehicle_age |
| **Incident** | incident_type, collision_type, incident_severity, authorities_contacted, number_of_vehicles, bodily_injuries, witnesses, police_report_available |
| **Claim** | total_claim_amount, injury_claim, property_claim, vehicle_claim |
| **Target** | fraud_reported (0 = Legitimate, 1 = Fraudulent) |

Fraud labels are correlated with known risk indicators: young policyholders, short tenure, total-loss severity, no witnesses, no police report, and high claim amounts.

**Public dataset alternatives:**
- [Porto Seguro Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) — 595K rows
- [Health Insurance Cross Sell](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) — 380K rows
- [Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) — 15K rows

## Model Architecture

```
Input (21 features)
  → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(64, ReLU)  → BatchNorm → Dropout(0.2)
  → Dense(32, ReLU)  → Dropout(0.2)
  → Dense(1, Sigmoid)
```

- **Optimizer:** Adam (lr=0.001) with ReduceLROnPlateau
- **Loss:** Binary cross-entropy with class weights for imbalance handling
- **Early stopping:** Monitors validation AUC, restores best weights

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/insurance-fraud-detection.git
cd insurance-fraud-detection
pip install -r requirements.txt
```

### 2. Train the model

Open and run all cells in the notebook:

```bash
jupyter notebook Insurance_Fraud_Detection.ipynb
```

This generates the dataset, trains the model, and saves all artifacts to `models/`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

## Streamlit Dashboard

### Fraud Prediction
Enter claim details and get a real-time fraud probability score with a gauge chart, risk level classification (High / Medium / Low), and flagged risk factors.

### Analytics Dashboard
- KPI cards: total claims, fraud rate, average claim amount, average premium
- Fraud distribution (donut chart)
- Claims by incident type (stacked bar)
- Fraud rate by severity and age group
- Claim amount distribution overlay
- Top fraud risk combinations table

### Model Performance
- Training loss and AUC curves
- Confusion matrix heatmap
- ROC curve with AUC score
- Full classification report
- Model architecture summary

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Data Generation | Creates 25K realistic insurance claims |
| 2 | EDA | 6 visualizations: distributions, fraud rates, correlations, claim analysis |
| 3 | Preprocessing | Label encoding, standard scaling, stratified split, class weights |
| 4 | Model Building | 5-layer neural network with batch normalization and dropout |
| 5 | Evaluation | Confusion matrix, ROC curve, PR curve, permutation feature importance |
| 6 | Export | Saves `.h5` model and all preprocessing artifacts |

## Tech Stack

- **ML/DL:** TensorFlow, Keras, scikit-learn
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit
- **Serialization:** Joblib (preprocessing), HDF5 (model)

## License

This project is licensed under the MIT License.
