import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Term Deposit Predictor",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, FEATURE_COLS = load_artifacts()

NUM_COLS = ["age", "balance", "duration", "campaign", "pdays", "previous"]

DISPLAY_UNKNOWN = "Not Sure"
MODEL_UNKNOWN = "unknown"

def ui_to_model(v: str) -> str:
    return MODEL_UNKNOWN if v == DISPLAY_UNKNOWN else v

def model_to_ui(v: str) -> str:
    return DISPLAY_UNKNOWN if v == MODEL_UNKNOWN else v

st.markdown(
    """
    <div style="padding: 1.2rem 1.2rem; border-radius: 14px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);">
      <div style="font-size: 2.2rem; font-weight: 800; line-height: 1.1;">Term Deposit Predictor</div>
      <div style="margin-top: .55rem; font-size: 1rem; opacity: 0.92;">
        This application estimates whether a customer is likely to subscribe to a term deposit based on customer profile information and marketing campaign details.
      </div>
      <div style="margin-top: .55rem; font-size: .95rem; opacity: 0.78;">
        Use the help icons next to each field if you are unsure what it means. If the information is not available, choose <b>Not Sure</b>.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

st.subheader("Input Details")
st.caption("Fill in the fields below. The help icons explain what each field means and why it matters to the prediction.")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Customer Profile")
    st.caption("Information about the customer and their financial background.")

    age = st.slider(
        "Age (years)",
        18, 95, 35,
        help="Customer age in years. Age can affect the likelihood of subscribing due to life stage and financial goals."
    )

    job = st.selectbox(
        "Job type",
        [model_to_ui(x) for x in [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"
        ]],
        help="Customer’s occupation group. Different job types often correlate with different income stability and banking needs."
    )

    marital = st.selectbox(
        "Marital status",
        [model_to_ui(x) for x in ["married", "single", "divorced", "unknown"]],
        help="Customer’s marital status. This can be associated with household financial responsibilities and saving behavior."
    )

    education = st.selectbox(
        "Education level",
        [model_to_ui(x) for x in ["primary", "secondary", "tertiary", "unknown"]],
        help="Highest education level (grouped). This can be linked to income potential and financial literacy patterns."
    )

    default = st.selectbox(
        "Credit in default",
        [model_to_ui(x) for x in ["no", "yes", "unknown"]],
        help="Whether the customer has credit in default. Customers in default may be less likely to take up financial products."
    )

    balance = st.number_input(
        "Account balance",
        value=0,
        step=50,
        help="Customer’s account balance in euros (can be negative). Balance may reflect saving capacity and financial health."
    )

    housing = st.selectbox(
        "Housing loan",
        [model_to_ui(x) for x in ["no", "yes", "unknown"]],
        help="Whether the customer has a housing loan. Existing commitments may influence willingness to subscribe."
    )

    loan = st.selectbox(
        "Personal loan",
        [model_to_ui(x) for x in ["no", "yes", "unknown"]],
        help="Whether the customer has a personal loan. This may reflect credit usage and monthly repayment obligations."
    )

with col_right:
    st.subheader("Campaign Behaviour")
    st.caption("Information about how the customer was contacted and their interaction with the campaign.")

    contact = st.selectbox(
        "Contact method",
        [model_to_ui(x) for x in ["cellular", "telephone", "unknown"]],
        help="How the customer was contacted. Contact method may affect response rates (e.g., reachable via mobile)."
    )

    month = st.selectbox(
        "Month of last contact",
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        help="Month when the last contact happened. Campaign timing can impact outcomes due to seasonal patterns."
    )

    day = st.slider(
        "Day of month (last contact)",
        1, 31, 15,
        help="Day of the month when the last contact happened. This can sometimes capture payday cycles or end-of-month behavior."
    )

    duration = st.slider(
        "Call duration (seconds)",
        0, 5000, 180,
        help="Length of the last call in seconds. Longer calls often indicate higher engagement, which can increase likelihood of subscription."
    )

    campaign = st.slider(
        "Contacts in this campaign",
        1, 60, 2,
        help="Number of contacts made during the current marketing campaign. Too many contacts may reduce success due to customer fatigue."
    )

    pdays = st.slider(
        "Days since previous contact (pdays)",
        -1, 999, 999,
        help="Days since the customer was last contacted in a previous campaign. A value of 999 usually means the customer was not previously contacted."
    )

    previous = st.slider(
        "Contacts before this campaign",
        0, 50, 0,
        help="Number of contacts performed before this campaign. Higher values indicate the customer has been approached before."
    )

    poutcome = st.selectbox(
        "Previous campaign outcome",
        [model_to_ui(x) for x in ["failure", "success", "other", "unknown"]],
        help="Result of the previous marketing campaign (if any). A previous success can be a strong indicator of future subscription."
    )

raw_input = pd.DataFrame([{
    "age": age,
    "job": ui_to_model(job),
    "marital": ui_to_model(marital),
    "education": ui_to_model(education),
    "default": ui_to_model(default),
    "balance": balance,
    "housing": ui_to_model(housing),
    "loan": ui_to_model(loan),
    "contact": ui_to_model(contact),
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": ui_to_model(poutcome),
}])

def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df_enc = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    df_ready = df_enc.reindex(columns=FEATURE_COLS, fill_value=0)

    present_num = [c for c in NUM_COLS if c in df_ready.columns]
    df_ready[present_num] = scaler.transform(df_ready[present_num])

    return df_ready

st.write("")
st.divider()

c1, c2 = st.columns([1, 3])
with c1:
    predict = st.button("Predict", use_container_width=True)

with c2:
    st.caption("Click Predict to generate the model’s estimated likelihood of subscription based on the provided inputs.")

if predict:
    X_input = preprocess_for_model(raw_input.copy())

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0, 1] if hasattr(model, "predict_proba") else None

    if str(pred).lower() in ["1", "yes", "true"]:
        st.success("Prediction: YES — customer is likely to subscribe.")
    else:
        st.info("Prediction: NO — customer is unlikely to subscribe.")

    if prob is not None:
        st.metric("Estimated probability of YES", f"{prob:.2%}")
