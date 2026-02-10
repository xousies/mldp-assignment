import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Term Deposit Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, FEATURE_COLS = load_artifacts()

# These match your notebook
NUM_COLS = ["age", "balance", "duration", "campaign", "pdays", "previous"]

st.title("Term Deposit Predictor")
st.write("Enter customer details to predict whether they will subscribe to a term deposit.")

# ---- Input UI (bank-full standard columns) ----
# NOTE: bank-full has 'day' (1-31), not day_of_week.
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Profile")
    age = st.slider("Age", 18, 95, 35)
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ])
    marital = st.selectbox("Marital", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Default", ["no", "yes", "unknown"])
    balance = st.number_input("Balance", value=0)
    housing = st.selectbox("Housing loan", ["no", "yes", "unknown"])
    loan = st.selectbox("Personal loan", ["no", "yes", "unknown"])

with col2:
    st.subheader("Campaign Details")
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
    day = st.slider("Day of month (last contact)", 1, 31, 15)
    month = st.selectbox("Month (last contact)", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    duration = st.slider("Call duration (seconds)", 0, 5000, 180)
    campaign = st.slider("Contacts in this campaign", 1, 60, 2)
    pdays = st.slider("Days since last contact (pdays)", -1, 999, 999)
    previous = st.slider("Contacts before this campaign", 0, 50, 0)
    poutcome = st.selectbox("Previous outcome", ["failure", "success", "other", "unknown"])

# Build raw input row exactly like your dataset columns
raw_input = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    # Match your notebook: strip + lowercase object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Match your notebook: one-hot encode with drop_first=True
    df_enc = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Align to training columns (fill missing with 0)
    df_ready = df_enc.reindex(columns=FEATURE_COLS, fill_value=0)

    # Scale numeric columns (if present)
    present_num = [c for c in NUM_COLS if c in df_ready.columns]
    df_ready[present_num] = scaler.transform(df_ready[present_num])

    return df_ready

st.divider()

if st.button("Predict"):
    X_input = preprocess_for_model(raw_input.copy())

    pred = model.predict(X_input)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_input)[0, 1]

    # Your y in dataset was "yes"/"no" then you binarized.
    # Depending on training, pred might be 0/1.
    if str(pred).lower() in ["1", "yes", "true"]:
        st.success("Prediction: YES — likely to subscribe")
    else:
        st.info("Prediction: NO — unlikely to subscribe")

    if prob is not None:
        st.write(f"Probability of YES: **{prob:.2%}**")
