# app.py
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Term Deposit Predictor (Tuned Model)", layout="centered")

# ---------------------------------
# Load trained model + expected columns (LABSHEET: model.pkl)
# ---------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    expected_cols = joblib.load("columns.pkl")
    return model, expected_cols

model, expected_cols = load_artifacts()

# ---------------------------------
# User-friendly "unknown" label mapping
# ---------------------------------
UNKNOWN_LABEL = "Not sure / Prefer not to say"

def map_unknown(value: str) -> str:
    return "unknown" if value == UNKNOWN_LABEL else value

# ---------------------------------
# Helper: preprocess to match trained columns
# ---------------------------------
def preprocess_to_expected_columns(input_raw: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(input_raw)
    encoded = encoded.reindex(columns=expected_cols, fill_value=0)
    return encoded

# ---------------------------------
# Header
# ---------------------------------
st.title("Bank Term Deposit Subscription Prediction")
st.write(
    "Fill in the customer details below. The app will predict whether the customer is likely to "
    "subscribe to a term deposit using a **tuned Random Forest model**."
)

with st.expander("Show loaded tuned model (for verification)", expanded=False):
    st.write(model)

st.divider()

# =========================================================
# 1) CUSTOMER PROFILE
# =========================================================
st.header("Customer Profile")
st.write(
    "This section captures the customer's **background and financial profile**. "
    "These details help the model understand the customer's general likelihood of subscribing."
)

cp_col1, cp_col2 = st.columns(2)

with cp_col1:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30,
        help="Customer's age in years."
    )

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", UNKNOWN_LABEL
        ],
        help="Customer's occupation category. Select 'Not sure' if unknown."
    )

    marital = st.selectbox(
        "Marital Status",
        ["single", "married", "divorced"],
        help="Customer's marital status."
    )

    education = st.selectbox(
        "Education Level",
        ["primary", "secondary", "tertiary", UNKNOWN_LABEL],
        help="Highest education attained by the customer."
    )

with cp_col2:
    default = st.selectbox(
        "Credit Default?",
        ["no", "yes", UNKNOWN_LABEL],
        help="Whether the customer has credit in default."
    )

    balance = st.number_input(
        "Account Balance",
        value=1000,
        help="Average yearly balance (bank account balance)."
    )

    housing = st.selectbox(
        "Housing Loan",
        ["no", "yes"],
        help="Whether the customer has a housing loan."
    )

    loan = st.selectbox(
        "Personal Loan",
        ["no", "yes"],
        help="Whether the customer has a personal loan."
    )

st.divider()

# =========================================================
# 2) CAMPAIGN BEHAVIOUR
# =========================================================
st.header("Campaign Behaviour")
st.write(
    "This section captures **how the bank contacted the customer** and how the customer responded "
    "in the current and previous marketing campaigns."
)

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", UNKNOWN_LABEL],
        help="Type of communication used to contact the customer (e.g., mobile phone vs telephone)."
    )

    day = st.number_input(
        "Last Contact Day (day of month)",
        min_value=1,
        max_value=31,
        value=5,
        help="Day of the month when the customer was last contacted."
    )

    month = st.selectbox(
        "Last Contact Month",
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        help="Month when the customer was last contacted."
    )

    duration = st.number_input(
        "Last Contact Duration (seconds)",
        min_value=0,
        value=180,
        help="Duration of the last contact/call in seconds."
    )

with cb_col2:
    campaign = st.number_input(
        "Number of Contacts in Current Campaign",
        min_value=0,
        value=1,
        help="How many times the customer was contacted during the current campaign."
    )

    pdays = st.number_input(
        "Days Since Last Contact",
        value=-1,
        help="Number of days since the customer was last contacted from a previous campaign (-1 means never contacted before)."
    )

    previous = st.number_input(
        "Number of Previous Contacts",
        min_value=0,
        value=0,
        help="Number of contacts performed before this campaign."
    )

    poutcome = st.selectbox(
        "Outcome of Previous Campaign",
        ["failure", "success", "other", UNKNOWN_LABEL],
        help="Result of the previous marketing campaign for this customer."
    )

st.divider()

# =========================================================
# 3) PREDICTION BUTTON
# =========================================================
st.header("Prediction")

st.write(
    "Click **Predict** to generate the model's prediction based on the inputs above."
)

# Build raw input (map "Not sure" back to 'unknown' for model compatibility)
input_raw = pd.DataFrame([{
    "age": age,
    "job": map_unknown(job),
    "marital": marital,
    "education": map_unknown(education),
    "default": map_unknown(default),
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": map_unknown(contact),
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": map_unknown(poutcome)
}])

input_ready = preprocess_to_expected_columns(input_raw)

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    pred = int(model.predict(input_ready)[0])

    if pred == 1:
        st.success("Prediction: **YES** — customer is likely to subscribe ✅")
    else:
        st.warning("Prediction: **NO** — customer is unlikely to subscribe ❌")

    if hasattr(model, "predict_proba"):
        prob_yes = float(model.predict_proba(input_ready)[0][1])
        st.write(f"Probability of YES: **{prob_yes:.2%}**")

    with st.expander("Show processed input used by the model"):
        st.dataframe(input_ready)

# ---------------------------------
# Optional: batch prediction (kept, but not in your main layout)
# ---------------------------------
with st.expander("Optional: Upload CSV for batch prediction"):
    st.write(
        "Upload a CSV containing the raw input columns (age, job, marital, education, default, balance, "
        "housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome). "
        "The app will preprocess and align the data automatically."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")

    sample = pd.DataFrame([{
        "age": 30,
        "job": "technician",
        "marital": "single",
        "education": "tertiary",
        "default": "no",
        "balance": 1000,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 5,
        "month": "may",
        "duration": 180,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }])

    st.caption("Sample CSV format:")
    st.dataframe(sample)

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)

            # Drop target if user accidentally included it
            for possible_target in ["y", "deposit", "subscribed", "target"]:
                if possible_target in df_raw.columns:
                    df_raw = df_raw.drop(columns=[possible_target])

            df_ready = preprocess_to_expected_columns(df_raw)
            preds = model.predict(df_ready)

            out = df_raw.copy()
            out["prediction"] = [int(x) for x in preds]
            out["prediction_label"] = ["YES" if int(x) == 1 else "NO" for x in preds]

            if hasattr(model, "predict_proba"):
                out["prob_yes"] = model.predict_proba(df_ready)[:, 1]

            st.success("Batch prediction completed.")
            st.dataframe(out)

        except Exception as e:
            st.error("Could not process the CSV. Please check your column names and values.")
            st.exception(e)
