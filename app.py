import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import joblib
import os

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Fraud Detection System",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.85; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
    .risk-high   { background: linear-gradient(135deg, #e74c3c, #c0392b); padding: 1rem; border-radius: 10px; color: white; }
    .risk-medium { background: linear-gradient(135deg, #f39c12, #e67e22); padding: 1rem; border-radius: 10px; color: white; }
    .risk-low    { background: linear-gradient(135deg, #2ecc71, #27ae60); padding: 1rem; border-radius: 10px; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Artifacts
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "fraud_detector.h5")
DATA_PATH = os.path.join(BASE, "data", "insurance_claims.csv")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Run the Jupyter notebook first to train and export the model.")
    st.info("Open **Insurance_Fraud_Detection.ipynb** and run all cells, then restart this app.")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(BASE, "models", "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(BASE, "models", "feature_columns.pkl"))
    eval_metrics = joblib.load(os.path.join(BASE, "models", "eval_metrics.pkl"))
    history = joblib.load(os.path.join(BASE, "models", "training_history.pkl"))
    return scaler, label_encoders, feature_columns, eval_metrics, history


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
scaler, label_encoders, feature_columns, eval_metrics, training_history = load_artifacts()
df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/000000/shield.png", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Fraud Prediction", "Analytics Dashboard", "Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Insurance Fraud Detection**")
st.sidebar.caption(f"Dataset: {len(df):,} claims")
st.sidebar.caption(f"Model: Deep Neural Network (.h5)")

# ===================================================================
# PAGE 1: FRAUD PREDICTION
# ===================================================================
if page == "Fraud Prediction":
    st.markdown('<p class="main-header">Fraud Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown("Enter claim details below and click **Predict** to assess fraud probability.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer Details")
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        annual_income = st.number_input("Annual Income ($)", 15000, 200000, 55000, step=5000)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

    with col2:
        st.subheader("Policy Details")
        policy_tenure = st.slider("Policy Tenure (months)", 1, 120, 24)
        policy_type = st.selectbox("Policy Type", ["Comprehensive", "Third Party", "Collision"])
        annual_premium = st.number_input("Annual Premium ($)", 500, 5000, 1500, step=100)
        policy_deductible = st.selectbox("Deductible ($)", [500, 1000, 1500, 2000])
        umbrella_limit = st.selectbox(
            "Umbrella Limit ($)",
            [0, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000],
            format_func=lambda x: f"${x:,}",
        )

    with col3:
        st.subheader("Incident Details")
        vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
        incident_type = st.selectbox(
            "Incident Type",
            ["Single Vehicle Collision", "Multi-Vehicle Collision", "Vehicle Theft", "Parked Car"],
        )
        collision_type = st.selectbox(
            "Collision Type", ["Front Collision", "Rear Collision", "Side Collision", "NA"]
        )
        incident_severity = st.selectbox("Severity", ["Minor Damage", "Major Damage", "Total Loss"])
        authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "None"])
        number_of_vehicles = st.selectbox("Vehicles Involved", [1, 2, 3, 4])
        bodily_injuries = st.selectbox("Bodily Injuries", [0, 1, 2, 3])
        witnesses = st.selectbox("Witnesses", [0, 1, 2, 3, 4, 5])
        police_report_available = st.selectbox("Police Report Available", ["Yes", "No"])

    st.markdown("---")
    st.subheader("Claim Amounts")
    ca, cb, cc, cd = st.columns(4)
    total_claim_amount = ca.number_input("Total Claim ($)", 1000, 100000, 15000, step=1000)
    injury_claim = cb.number_input("Injury Claim ($)", 0, 50000, 3000, step=500)
    property_claim = cc.number_input("Property Claim ($)", 0, 50000, 3000, step=500)
    vehicle_claim = total_claim_amount - injury_claim - property_claim
    cd.metric("Vehicle Claim ($)", f"${vehicle_claim:,}")

    st.markdown("---")

    if st.button("Predict Fraud Probability", type="primary", use_container_width=True):
        input_dict = {
            "age": age, "gender": gender, "education": education,
            "annual_income": annual_income, "marital_status": marital_status,
            "policy_tenure_months": policy_tenure, "policy_type": policy_type,
            "annual_premium": annual_premium, "policy_deductible": policy_deductible,
            "umbrella_limit": umbrella_limit, "vehicle_age": vehicle_age,
            "incident_type": incident_type, "collision_type": collision_type,
            "incident_severity": incident_severity,
            "authorities_contacted": authorities_contacted,
            "number_of_vehicles": number_of_vehicles,
            "bodily_injuries": bodily_injuries, "witnesses": witnesses,
            "police_report_available": police_report_available,
            "total_claim_amount": total_claim_amount,
            "injury_claim": injury_claim, "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
        }
        input_df = pd.DataFrame([input_dict])

        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        input_scaled = scaler.transform(input_df[feature_columns])
        fraud_prob = float(model.predict(input_scaled, verbose=0).flatten()[0])

        res1, res2 = st.columns([1, 1])

        with res1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_prob * 100,
                number={"suffix": "%", "font": {"size": 48}},
                title={"text": "Fraud Probability", "font": {"size": 20}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 2},
                    "bar": {"color": "#2c3e50"},
                    "steps": [
                        {"range": [0, 30], "color": "#2ecc71"},
                        {"range": [30, 60], "color": "#f39c12"},
                        {"range": [60, 100], "color": "#e74c3c"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.8,
                        "value": fraud_prob * 100,
                    },
                },
            ))
            fig.update_layout(height=350, margin=dict(t=80, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with res2:
            if fraud_prob >= 0.7:
                st.markdown(
                    '<div class="risk-high"><h2>HIGH RISK</h2>'
                    "<p>This claim has a high probability of being fraudulent. "
                    "Recommend immediate investigation by the SIU team.</p></div>",
                    unsafe_allow_html=True,
                )
            elif fraud_prob >= 0.4:
                st.markdown(
                    '<div class="risk-medium"><h2>MEDIUM RISK</h2>'
                    "<p>This claim shows some fraud indicators. "
                    "Recommend additional documentation and verification.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="risk-low"><h2>LOW RISK</h2>'
                    "<p>This claim appears legitimate based on the model analysis. "
                    "Standard processing recommended.</p></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Key Risk Factors Checked")
            risk_factors = []
            if age < 30:
                risk_factors.append(("Young policyholder (< 30)", True))
            if policy_tenure < 12:
                risk_factors.append(("Short policy tenure (< 12 months)", True))
            if incident_severity == "Total Loss":
                risk_factors.append(("Total loss claim", True))
            if witnesses == 0:
                risk_factors.append(("No witnesses present", True))
            if police_report_available == "No":
                risk_factors.append(("No police report filed", True))
            if authorities_contacted == "None":
                risk_factors.append(("No authorities contacted", True))
            if total_claim_amount > df["total_claim_amount"].quantile(0.90):
                risk_factors.append(("Claim amount in top 10%", True))
            if bodily_injuries >= 2:
                risk_factors.append(("Multiple bodily injuries", True))

            if risk_factors:
                for factor, flagged in risk_factors:
                    st.markdown(f"- :red[{factor}]")
            else:
                st.markdown("- :green[No major risk factors detected]")


# ===================================================================
# PAGE 2: ANALYTICS DASHBOARD
# ===================================================================
elif page == "Analytics Dashboard":
    st.markdown('<p class="main-header">Insurance Claims Analytics</p>', unsafe_allow_html=True)
    st.markdown("---")

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="metric-card"><h3>Total Claims</h3><h1>{len(df):,}</h1></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="metric-card" style="background:linear-gradient(135deg,#e74c3c,#c0392b)">'
            f'<h3>Fraud Rate</h3><h1>{df["fraud_reported"].mean():.1%}</h1></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="metric-card" style="background:linear-gradient(135deg,#2ecc71,#27ae60)">'
            f'<h3>Avg Claim Amount</h3><h1>${df["total_claim_amount"].mean():,.0f}</h1></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="metric-card" style="background:linear-gradient(135deg,#3498db,#2980b9)">'
            f'<h3>Avg Premium</h3><h1>${df["annual_premium"].mean():,.0f}</h1></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Fraud distribution + Incident type
    c1, c2 = st.columns(2)
    with c1:
        fraud_counts = df["fraud_reported"].value_counts().reset_index()
        fraud_counts.columns = ["Status", "Count"]
        fraud_counts["Status"] = fraud_counts["Status"].map({0: "Legitimate", 1: "Fraudulent"})
        fig = px.pie(
            fraud_counts, values="Count", names="Status",
            color="Status",
            color_discrete_map={"Legitimate": "#2ecc71", "Fraudulent": "#e74c3c"},
            title="Claims Distribution",
            hole=0.4,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        incident_fraud = (
            df.groupby("incident_type")["fraud_reported"]
            .agg(["sum", "count"])
            .reset_index()
        )
        incident_fraud.columns = ["Incident Type", "Fraudulent", "Total"]
        incident_fraud["Legitimate"] = incident_fraud["Total"] - incident_fraud["Fraudulent"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=incident_fraud["Incident Type"], y=incident_fraud["Legitimate"],
            name="Legitimate", marker_color="#2ecc71",
        ))
        fig.add_trace(go.Bar(
            x=incident_fraud["Incident Type"], y=incident_fraud["Fraudulent"],
            name="Fraudulent", marker_color="#e74c3c",
        ))
        fig.update_layout(
            barmode="stack", title="Claims by Incident Type",
            height=400, xaxis_tickangle=-25,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Fraud rate by feature + Age distribution
    c3, c4 = st.columns(2)
    with c3:
        severity_fraud = df.groupby("incident_severity")["fraud_reported"].mean().reset_index()
        severity_fraud.columns = ["Severity", "Fraud Rate"]
        fig = px.bar(
            severity_fraud, x="Severity", y="Fraud Rate",
            color="Fraud Rate",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            title="Fraud Rate by Incident Severity",
            text_auto=".1%",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        df["age_group"] = pd.cut(df["age"], bins=[17, 25, 35, 45, 55, 65, 81],
                                 labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
        age_fraud = df.groupby("age_group")["fraud_reported"].mean().reset_index()
        age_fraud.columns = ["Age Group", "Fraud Rate"]
        fig = px.line(
            age_fraud, x="Age Group", y="Fraud Rate",
            markers=True, title="Fraud Rate by Age Group",
            line_shape="spline",
        )
        fig.update_traces(line=dict(width=3, color="#e74c3c"), marker=dict(size=10))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Claim amount distribution + Feature heatmap
    c5, c6 = st.columns(2)
    with c5:
        fig = px.histogram(
            df, x="total_claim_amount", color=df["fraud_reported"].map({0: "Legitimate", 1: "Fraudulent"}),
            nbins=50, title="Claim Amount Distribution",
            color_discrete_map={"Legitimate": "#2ecc71", "Fraudulent": "#e74c3c"},
            opacity=0.7, barmode="overlay",
        )
        fig.update_layout(height=400, xaxis_title="Total Claim Amount ($)", legend_title="Status")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        policy_fraud = df.groupby("policy_type")["fraud_reported"].mean().reset_index()
        policy_fraud.columns = ["Policy Type", "Fraud Rate"]
        auth_fraud = df.groupby("authorities_contacted")["fraud_reported"].mean().reset_index()
        auth_fraud.columns = ["Authority", "Fraud Rate"]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("By Policy Type", "By Authority Contacted"))
        fig.add_trace(
            go.Bar(x=policy_fraud["Policy Type"], y=policy_fraud["Fraud Rate"],
                   marker_color="#3498db", text=policy_fraud["Fraud Rate"].apply(lambda x: f"{x:.1%}"),
                   textposition="outside", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=auth_fraud["Authority"], y=auth_fraud["Fraud Rate"],
                   marker_color="#9b59b6", text=auth_fraud["Fraud Rate"].apply(lambda x: f"{x:.1%}"),
                   textposition="outside", showlegend=False),
            row=1, col=2,
        )
        fig.update_layout(title="Fraud Rate Breakdown", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Top risk combinations
    st.subheader("Top Fraud Risk Combinations")
    risk_combos = (
        df.groupby(["incident_severity", "police_report_available", "authorities_contacted"])
        .agg(total_claims=("fraud_reported", "count"), fraudulent=("fraud_reported", "sum"))
        .reset_index()
    )
    risk_combos["fraud_rate"] = risk_combos["fraudulent"] / risk_combos["total_claims"]
    risk_combos = risk_combos.sort_values("fraud_rate", ascending=False).head(10)
    risk_combos["fraud_rate"] = risk_combos["fraud_rate"].apply(lambda x: f"{x:.1%}")

    st.dataframe(
        risk_combos.rename(columns={
            "incident_severity": "Severity",
            "police_report_available": "Police Report",
            "authorities_contacted": "Authority",
            "total_claims": "Total Claims",
            "fraudulent": "Fraudulent",
            "fraud_rate": "Fraud Rate",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ===================================================================
# PAGE 3: MODEL PERFORMANCE
# ===================================================================
elif page == "Model Performance":
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Metrics cards
    report = eval_metrics["classification_report"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{report['accuracy']:.1%}")
    m2.metric("ROC AUC", f"{eval_metrics['roc_auc']:.3f}")
    m3.metric("Precision (Fraud)", f"{report['Fraudulent']['precision']:.1%}")
    m4.metric("Recall (Fraud)", f"{report['Fraudulent']['recall']:.1%}")

    st.markdown("---")

    # Training history
    st.subheader("Training History")
    tc1, tc2 = st.columns(2)

    with tc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=training_history["loss"], name="Train Loss",
                                 line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(y=training_history["val_loss"], name="Val Loss",
                                 line=dict(color="#e74c3c", width=2, dash="dash")))
        fig.update_layout(title="Loss over Epochs", xaxis_title="Epoch",
                          yaxis_title="Loss", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tc2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=training_history["auc"], name="Train AUC",
                                 line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(y=training_history["val_auc"], name="Val AUC",
                                 line=dict(color="#e74c3c", width=2, dash="dash")))
        fig.update_layout(title="AUC over Epochs", xaxis_title="Epoch",
                          yaxis_title="AUC", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Confusion matrix + ROC curve
    st.subheader("Evaluation on Test Set")
    ec1, ec2 = st.columns(2)

    with ec1:
        cm = np.array(eval_metrics["confusion_matrix"])
        labels = ["Legitimate", "Fraudulent"]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted", yaxis_title="Actual",
            height=400, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ec2:
        fpr = eval_metrics["fpr"]
        tpr = eval_metrics["tpr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC (AUC = {eval_metrics['roc_auc']:.3f})",
            line=dict(color="#3498db", width=3),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random", line=dict(color="grey", dash="dash"),
        ))
        fig.update_layout(
            title="ROC Curve", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed classification report
    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    report_df = report_df.round(3)
    st.dataframe(report_df, use_container_width=True)

    # Model architecture summary
    with st.expander("Model Architecture"):
        model_layers = []
        for layer in model.layers:
            config = layer.get_config()
            model_layers.append({
                "Layer": layer.__class__.__name__,
                "Output Shape": str(layer.output_shape) if hasattr(layer, "output_shape") else "N/A",
                "Parameters": layer.count_params(),
            })
        st.dataframe(pd.DataFrame(model_layers), use_container_width=True, hide_index=True)
        st.caption(f"Total parameters: {model.count_params():,}")
