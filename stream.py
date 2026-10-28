
import hmac
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Claim Severity Triage",
    page_icon="📋",
    layout="wide",
)


# --------------------------------------------------
# 2. LOGIN
# --------------------------------------------------
# Replace these placeholders with your own credentials.
# This simple login is intended for a local demonstration,
# not a publicly deployed application.

EMAIL = "wanjiru456@gmail.com"
PASSWORD = "niggasaintshit"


def verify_login(email, password):
    return (
        hmac.compare_digest(
            email.strip().lower(),
            EMAIL.strip().lower(),
        )
        and hmac.compare_digest(password, PASSWORD)
    )


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


if not st.session_state.authenticated:
    st.title("📋 Insurance Claim Severity Triage")
    st.write("Please log in to access the application.")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button(
            "Log In",
            use_container_width=True,
        )

    if login_submitted:
        if verify_login(email, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid email or password.")

    st.stop()


# --------------------------------------------------
# 3. LOAD SAVED MODELS AND PREPROCESSING OBJECTS
# --------------------------------------------------

@st.cache_resource
def load_objects():
    model_dir = Path("saved_models")

    model = joblib.load(model_dir / "mlp_model.pkl")
    tfidf = joblib.load(model_dir / "tfidf_vectorizer.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    numeric_cols = joblib.load(model_dir / "numeric_cols.pkl")
    structured_columns = joblib.load(
        model_dir / "structured_columns.pkl"
    )

    # Exclude InitialCaseEstimate, which was not used as a model input.
    numeric_cols = [
        column for column in numeric_cols
        if column != "InitialCaseEstimate"
    ]
    return model, tfidf, scaler, numeric_cols, structured_columns


try:
    (
        model,
        tfidf,
        scaler,
        numeric_cols,
        structured_columns,
    ) = load_objects()

except Exception as error:
    st.error(
        "The saved model files could not be loaded. "
        "Check that all five files are inside the saved_models folder."
    )
    st.exception(error)
    st.stop()


# --------------------------------------------------
# 4. CLAIM DESCRIPTION PREPROCESSING
# --------------------------------------------------
# IMPORTANT:
# This must match the exact text preprocessing used
# when the TF-IDF vectorizer was fitted in your notebook.

stemmer = PorterStemmer()

word_map = {
    "fell": "fall",
    "fallen": "fall",
    "struck": "strike",
}


def stem_description(description):
    words = re.findall(
        r"\b[a-zA-Z]+\b",
        str(description).lower(),
    )

    words = [word_map.get(word, word) for word in words]

    return " ".join(stemmer.stem(word) for word in words)


# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------

st.title("📋 Insurance Claim Severity Triage")

st.write(
    "This application uses machine learning to estimate "
    "whether an insurance claim is Low, Medium, or High severity."
)

with st.expander("About This App"):
    st.markdown(
        """
        **What does this app do?**

        It estimates the severity category of an insurance claim
        using the information entered and the claim description.

        **How does it work?**

        1. Enter the claimant and accident details.
        2. Describe the accident or injury.
        3. The application processes the information.
        4. The trained machine learning model predicts a severity category.

        **Important:** The prediction is an estimate, not a final
        claims decision. An authorised claims assessor should review
        the claim before making decisions.
        """
    )
    


if st.button("Log Out"):
    st.session_state.authenticated = False
    st.rerun()


st.divider()


# --------------------------------------------------
# 6. CLAIM INPUT FORM
# --------------------------------------------------

st.subheader("Enter Claim Details")

with st.form("claim_form"):

    st.markdown("#### Claimant Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Claimant Age",
            min_value=18,
            max_value=100,
            value=35,
        )

        gender = st.selectbox(
            "Gender",
            ["F", "M"],
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["M", "S", "Unknown"],
        )

        weekly_rate = st.number_input(
            "Weekly Rate",
            min_value=0.0,
            value=500.0,
        )

    with col2:
        employment_type = st.selectbox(
            "Employment Type",
            ["F", "P"],
        )

        hours_worked = st.number_input(
            "Hours Worked Per Week",
            min_value=0.0,
            max_value=168.0,
            value=40.0,
        )

        days_worked = st.number_input(
            "Days Worked Per Week",
            min_value=0,
            max_value=7,
            value=5,
        )

        dependent_children = st.number_input(
            "Dependent Children",
            min_value=0,
            value=0,
        )

        dependents_other = st.number_input(
            "Other Dependents",
            min_value=0,
            value=0,
        )

    st.markdown("#### Accident Information")

    col3, col4 = st.columns(2)

    with col3:
        accident_year = st.number_input(
            "Accident Year",
            min_value=1988,
            max_value=2100,
            value=2000,
        )

        accident_month = st.selectbox(
            "Accident Month",
            list(range(1, 13)),
        )

        accident_day = st.number_input(
            "Accident Day of Week (0 = Monday)",
            min_value=0,
            max_value=6,
            value=0,
        )

    with col4:
        accident_hour = st.number_input(
            "Accident Hour",
            min_value=0,
            max_value=24,
            value=12,
        )

        reporting_delay = st.number_input(
            "Reporting Delay",
            min_value=0,
            value=1,
        )

        reported_day = st.selectbox(
            "Reported Day",
            ["0Mon", "1Tues", "2Wed", "3Thurs", "4Fri"],
        )

    st.markdown("#### Claim Description")

    claim_description = st.text_area(
        "Describe the accident or injury",
        placeholder=(
            "For example: The claimant fell and injured "
            "their right shoulder."
        ),
        height=150,
    )

    submitted = st.form_submit_button(
        "Submit Claim Details",
        use_container_width=True,
    )


# --------------------------------------------------
# 7. VALIDATE AND PROCESS THE CLAIM
# --------------------------------------------------

if submitted:

    if not claim_description.strip():
        st.warning("Please enter a claim description.")

    elif len(claim_description.strip()) < 10:
        st.warning(
            "Please provide a more detailed claim description."
        )

    else:
        try:
            # ------------------------------------------
            # 7A. CREATE STRUCTURED FEATURES
            # ------------------------------------------

            claim_data = pd.DataFrame(
                0.0,
                index=[0],
                columns=structured_columns,
            )

            numeric_values = {
                "ReportingDelay": reporting_delay,
                "AccidentHour": accident_hour,
                "Age": age,
                "WeeklyRate": weekly_rate,
                "HoursWorkedPerWeek": hours_worked,
                "DependentChildren": dependent_children,
                "DependentsOther": dependents_other,
                "DaysWorkedPerWeek": days_worked,
                "AccidentYear": accident_year,
                "AccidentMonth": accident_month,
                "AccidentDayOfWeek": accident_day,
            }

            for column, value in numeric_values.items():
                if column in claim_data.columns:
                    claim_data.loc[0, column] = value

            categorical_values = {
                "Gender": gender,
                "MaritalStatus": marital_status,
                "PartTimeFullTime": employment_type,
                "ReportedDay": reported_day,
            }

            for column, value in categorical_values.items():
                encoded_column = f"{column}_{value}"

                if encoded_column in claim_data.columns:
                    claim_data.loc[0, encoded_column] = 1.0

            # Check that every required numeric feature exists.
            missing_numeric = [
                column
                for column in numeric_cols
                if column not in claim_data.columns
            ]

            if missing_numeric:
                st.error(
                    "Some required numeric features are missing: "
                    + ", ".join(missing_numeric)
                )
                st.stop()

            
            # Scale only the columns the saved scaler was trained on.
            scaler_columns = list(scaler.feature_names_in_)

            missing_scaler_columns = [
                col for col in scaler_columns
                if col not in claim_data.columns
            ]

            if missing_scaler_columns:
                st.error(
                    f"Missing features required by the scaler: "
                    f"{missing_scaler_columns}"
                )
                st.stop()

            claim_data.loc[:, scaler_columns] = scaler.transform(
                claim_data[scaler_columns]
            )


            # Restore the exact training column order.
            claim_data = claim_data[structured_columns]

            if claim_data.columns.tolist() != list(structured_columns):
                st.error(
                    "Structured feature order does not match training."
                )
                st.stop()

            if claim_data.shape[1] != 23:
                st.error(
                    f"Expected 23 structured features, but received "
                    f"{claim_data.shape[1]}."
                )
                st.stop()

            # ------------------------------------------
            # 7B. PROCESS CLAIM DESCRIPTION
            # ------------------------------------------

            claim_text = stem_description(claim_description)

            claim_text_features = tfidf.transform([claim_text])

            # ------------------------------------------
            # 7C. COMBINE STRUCTURED AND TEXT FEATURES
            # ------------------------------------------

            claim_structured_features = csr_matrix(
                claim_data.to_numpy(dtype=float)
            )

            claim_features = hstack(
                [
                    claim_structured_features,
                    claim_text_features,
                ],
                format="csr",
            )

            if claim_features.shape[1] != 488:
                st.error(
                    f"Expected 488 combined features, but received "
                    f"{claim_features.shape[1]}. "
                    "Check the saved model and preprocessing objects."
                )
                st.stop()

            if claim_features.shape[1] != model.n_features_in_:
                st.error(
                    f"The model expects {model.n_features_in_} features, "
                    f"but received {claim_features.shape[1]}."
                )
                st.stop()

            # ------------------------------------------
            # 7D. PREDICT SEVERITY
            # ------------------------------------------

            predicted_severity = model.predict(claim_features)[0]

            probabilities = model.predict_proba(claim_features)[0]

            # ------------------------------------------
            # 8. DISPLAY CLAIM SUMMARY
            # ------------------------------------------

            st.divider()
            st.subheader("Submitted Claim Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.write("**Claimant age:**", age)
                st.write("**Gender:**", gender)
                st.write("**Marital status:**", marital_status)
                st.write("**Accident year:**", accident_year)
                st.write("**Accident month:**", accident_month)
                st.write("**Accident hour:**", accident_hour)

            with summary_col2:
                st.write("**Reporting delay:**", reporting_delay)
                st.write("**Weekly rate:**", weekly_rate)
                st.write("**Employment type:**", employment_type)
                st.write("**Hours worked per week:**", hours_worked)
                st.write("**Days worked per week:**", days_worked)
                st.write("**Reported day:**", reported_day)

            with st.expander("View submitted claim description"):
                st.write(claim_description)

            # ------------------------------------------
            # 9. DISPLAY PREDICTION
            # ------------------------------------------

            st.divider()
            st.subheader("Assessment Results")

            result_col1, result_col2 = st.columns([2, 1])

            with result_col1:
                st.markdown("### Predicted Claim Severity")

                if predicted_severity == "High":
                    st.error("HIGH SEVERITY")

                elif predicted_severity == "Medium":
                    st.warning("MEDIUM SEVERITY")

                else:
                    st.success("LOW SEVERITY")

            with result_col2:
                st.metric(
                    "Highest estimated class probability",
                    f"{max(probabilities):.1%}",
                )

            st.caption(
                "These probabilities are model estimates, not certainties. "
                "They do not guarantee the actual outcome of a claim."
            )

            # ------------------------------------------
            # 10. DISPLAY CLASS PROBABILITIES
            # ------------------------------------------

            st.subheader("Estimated Probability by Severity")

            probability_results = pd.DataFrame(
                {
                    "Severity": model.classes_,
                    "Estimated Probability": [
                        f"{probability:.1%}"
                        for probability in probabilities
                    ],
                }
            )

            st.dataframe(
                probability_results,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------
            # 11. ASSESSMENT GUIDANCE
            # ------------------------------------------

            st.subheader("Assessment Guidance")

            if predicted_severity == "High":
                st.warning(
                    "This claim was classified as High severity. "
                    "Prioritise it for prompt review by an authorised "
                    "claims assessor."
                )

            elif predicted_severity == "Medium":
                st.info(
                    "This claim was classified as Medium severity. "
                    "Review the supporting information and assess whether "
                    "further investigation is needed."
                )

            else:
                st.info(
                    "This claim was classified as Low severity. "
                    "Continue with the standard claims assessment process."
                )

            st.caption(
                "This guidance is illustrative. Follow your organisation's "
                "official claims-handling procedures when making decisions."
            )

        except Exception as error:
            st.error(
                "The claim could not be processed. "
                "Check the model files, feature preparation, and "
                "preprocessing steps."
            )
            st.exception(error)


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Insurance Claim Severity Triage | Machine learning decision support. "
    "All predictions require appropriate human review."
)
