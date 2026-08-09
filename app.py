import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Bank Churn Risk Predictor", layout="wide")


@st.cache_resource
def load_model_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "models", "churn_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))
    model_columns = joblib.load(os.path.join(base_dir, "models", "model_columns.pkl"))
    numeric_cols = joblib.load(os.path.join(base_dir, "models", "numeric_cols.pkl"))
    return model, scaler, model_columns, numeric_cols


model, scaler, model_columns, numeric_cols = load_model_assets()

st.title("🏦 Bank Customer Churn Risk Predictor")
st.caption("Predicts churn probability using an XGBoost model trained on 10,000 European bank customers")
st.write(f"Model loaded successfully. Expects {len(model_columns)} features.")


def build_customer_input():
    with st.sidebar:
        st.header("Customer Profile")
        credit_score = st.slider("Credit Score", 350, 850, 650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_credit_card = st.checkbox("Has credit card", value=True)
        is_active_member = st.checkbox("Is active member", value=True)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=60000.0, step=1000.0)

    return {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
    }


def make_feature_frame(raw_features):
    row = {
        'CreditScore': raw_features['CreditScore'],
        'Age': raw_features['Age'],
        'Tenure': raw_features['Tenure'],
        'Balance': raw_features['Balance'],
        'NumOfProducts': raw_features['NumOfProducts'],
        'HasCrCard': raw_features['HasCrCard'],
        'IsActiveMember': raw_features['IsActiveMember'],
        'EstimatedSalary': raw_features['EstimatedSalary'],
        # One-hot encoding matching training exactly (drop_first=True: France/Female are baseline)
        'Geography_Germany': 1 if raw_features['Geography'] == "Germany" else 0,
        'Geography_Spain': 1 if raw_features['Geography'] == "Spain" else 0,
        'Gender_Male': 1 if raw_features['Gender'] == "Male" else 0,
    }

    # Engineered features - MUST match notebook Cell 21 exactly
    row['BalanceSalaryRatio'] = row['Balance'] / (row['EstimatedSalary'] + 1)
    row['ProductDensity'] = row['NumOfProducts'] / (row['Tenure'] + 1)
    row['EngagementProductInteraction'] = row['IsActiveMember'] * row['NumOfProducts']
    row['AgeTenureInteraction'] = row['Age'] * row['Tenure']

    df = pd.DataFrame([row])
    df = df.reindex(columns=model_columns, fill_value=0)

    scaled_df = df.copy()
    scaled_df[numeric_cols] = scaler.transform(df[numeric_cols])
    return scaled_df


def predict_risk(raw_features):
    feature_df = make_feature_frame(raw_features)
    probability = model.predict_proba(feature_df)[:, 1][0]
    return float(probability)


st.sidebar.markdown("---")
raw_features = build_customer_input()

st.divider()
if st.button("Predict Churn Risk", type="primary"):
    churn_probability = predict_risk(raw_features)
    churn_percent = churn_probability * 100

    if churn_probability >= 0.5:
        risk_label = "High risk"
        risk_color = "#FF4B4B"
    elif churn_probability >= 0.3:
        risk_label = "Medium risk"
        risk_color = "#F9C74F"
    else:
        risk_label = "Low risk"
        risk_color = "#2ECC71"

    st.subheader("Prediction Result")
    st.markdown(
        f"<div style='background-color:{risk_color};padding:20px;border-radius:12px;color:white;font-size:28px;font-weight:bold;text-align:center'>"
        f"{risk_label}: {churn_percent:.1f}% churn probability"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{churn_percent:.1f}%")
    with col2:
        st.metric("Risk Level", risk_label)

    st.info(
        "Interpretation: values above 50% suggest a customer is highly likely to churn; "
        "30-50% indicates elevated risk and may warrant retention outreach."
    )

    st.session_state['last_raw_features'] = raw_features
    st.session_state['last_probability'] = churn_probability
else:
    st.info("Complete the customer profile in the sidebar and click 'Predict Churn Risk' to generate an assessment.")

st.divider()
st.subheader("What Drives Churn Predictions")
importance_data = pd.Series(model.feature_importances_, index=model_columns).sort_values(ascending=False)

fig = px.bar(
    x=importance_data.head(10).index,
    y=importance_data.head(10).values,
    labels={'x': 'Feature', 'y': 'Importance Score'}
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Top drivers: NumOfProducts, IsActiveMember, and Age consistently rank highest across all 5 models tested during development.")

st.divider()
st.subheader("What-If: Number of Products")
st.caption("See how churn probability changes for the last-predicted customer across different product counts, holding everything else constant.")

if 'last_raw_features' in st.session_state:
    base_features = st.session_state['last_raw_features']
    products_range = [1, 2, 3, 4]
    probs = []
    for p in products_range:
        temp_features = base_features.copy()
        temp_features['NumOfProducts'] = p
        probs.append(predict_risk(temp_features) * 100)

    whatif_df = pd.DataFrame({'NumOfProducts': products_range, 'ChurnProbability (%)': probs})
    st.line_chart(whatif_df.set_index('NumOfProducts'))
    st.dataframe(whatif_df.style.format({'ChurnProbability (%)': '{:.1f}'}), use_container_width=True)
else:
    st.info("Click 'Predict Churn Risk' above first to see the what-if simulation for that customer.")