import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

        .stApp {
            background: linear-gradient(135deg, #0f172a, #1e293b);
        }

        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0;
        }

        .subtitle {
            text-align: center;
            color: #cbd5e1;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        .section-title {
            color: white;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .result-card {
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: bold;
            margin-top: 20px;
        }

        .churn-card {
            background-color: rgba(239, 68, 68, 0.15);
            border: 2px solid #ef4444;
            color: #fca5a5;
        }

        .safe-card {
            background-color: rgba(34, 197, 94, 0.15);
            border: 2px solid #22c55e;
            color: #86efac;
        }

        .stButton > button {
            width: 100%;
            height: 50px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ==================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model("model.h5")

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return (
        model,
        label_encoder_gender,
        onehot_encoder_geo,
        scaler
    )


model, label_encoder_gender, onehot_encoder_geo, scaler = load_resources()


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<h1 class="main-title">📊 Customer Churn Predictor</h1>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    Enter customer information to predict the probability of customer churn
    using an Artificial Neural Network.
    </p>
    """,
    unsafe_allow_html=True
)


# ==================================================
# CUSTOMER INPUT SECTION
# ==================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    geography = st.selectbox(
        "🌍 Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "👤 Gender",
        label_encoder_gender.classes_
    )

    age = st.slider(
        "🎂 Age",
        min_value=18,
        max_value=92,
        value=35
    )


with col2:

    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=0,
        value=650
    )

    balance = st.number_input(
        "💰 Account Balance",
        min_value=0.0,
        value=50000.0
    )

    estimated_salary = st.number_input(
        "💵 Estimated Salary",
        min_value=0.0,
        value=50000.0
    )


with col3:

    tenure = st.slider(
        "📅 Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    num_of_products = st.slider(
        "📦 Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    has_cr_card = st.selectbox(
        "💳 Has Credit Card",
        ["No", "Yes"]
    )

    is_active_member = st.selectbox(
        "🟢 Active Member",
        ["No", "Yes"]
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.markdown("---")

predict_button = st.button("🔮 Predict Customer Churn")


# ==================================================
# PREDICTION LOGIC
# ==================================================

if predict_button:

    with st.spinner("🤖 AI is analyzing customer behavior..."):

        # Convert Yes / No into 0 / 1
        has_cr_card_value = 1 if has_cr_card == "Yes" else 0
        is_active_member_value = (
            1 if is_active_member == "Yes" else 0
        )


        # ----------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------

        input_data = pd.DataFrame(
            {
                "CreditScore": [credit_score],
                "Gender": [
                    label_encoder_gender.transform([gender])[0]
                ],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [num_of_products],
                "HasCrCard": [has_cr_card_value],
                "IsActiveMember": [
                    is_active_member_value
                ],
                "EstimatedSalary": [
                    estimated_salary
                ],
            }
        )


        # ----------------------------------------------
        # ONE-HOT ENCODE GEOGRAPHY
        # ----------------------------------------------

        geo_encoded = onehot_encoder_geo.transform(
            [[geography]]
        ).toarray()

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(
                ["Geography"]
            )
        )


        # ----------------------------------------------
        # COMBINE FEATURES
        # ----------------------------------------------

        input_data = pd.concat(
            [
                input_data.reset_index(drop=True),
                geo_encoded_df
            ],
            axis=1
        )


        # ----------------------------------------------
        # SCALE INPUT DATA
        # ----------------------------------------------

        input_data_scaled = scaler.transform(input_data)


        # ----------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------

        prediction = model.predict(
            input_data_scaled,
            verbose=0
        )

        prediction_proba = float(prediction[0][0])


    # ==================================================
    # RESULT SECTION
    # ==================================================

    st.markdown("## 🤖 Prediction Result")

    churn_probability = prediction_proba
    retention_probability = 1 - prediction_proba


    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            label="🔴 Churn Probability",
            value=f"{churn_probability * 100:.2f}%"
        )


    with metric_col2:

        st.metric(
            label="🟢 Retention Probability",
            value=f"{retention_probability * 100:.2f}%"
        )


    # ----------------------------------------------
    # PROBABILITY BAR
    # ----------------------------------------------

    st.write("### 📊 Churn Risk Score")

    st.progress(int(churn_probability * 100))


    # ----------------------------------------------
    # RESULT CARD
    # ----------------------------------------------

    if prediction_proba > 0.5:

        st.markdown(
            f"""
            <div class="result-card churn-card">
                ⚠️ HIGH CHURN RISK<br><br>
                This customer is likely to churn.<br>
                Churn probability: {churn_probability * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-card safe-card">
                ✅ LOW CHURN RISK<br><br>
                This customer is likely to remain with the bank.<br>
                Retention probability: {retention_probability * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )


    # ----------------------------------------------
    # DETAILS
    # ----------------------------------------------

    with st.expander("🔬 View Prediction Details"):

        st.write("### Model Output")

        st.code(f"{prediction_proba:.6f}")

        st.write(
            """
            The ANN produces a probability between **0 and 1**.

            - **0.00 → Low churn probability**
            - **0.50 → Decision threshold**
            - **1.00 → High churn probability**
            """
        )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <p style="text-align:center; color:#94a3b8;">
        Built with TensorFlow • Streamlit • Artificial Neural Networks
    </p>
    """,
    unsafe_allow_html=True
)