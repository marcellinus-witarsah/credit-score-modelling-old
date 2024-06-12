import pandas as pd
import streamlit as st
import requests
import pickle


def main():
    # Load model:
    with open("models/scorecard_model_optbinning.pkl", "rb") as f:
        scorecard_model = pickle.load(f)

    # App User Interfaces:
    st.title("Loan Application Form")

    st.header("Personal Information")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
    person_income = st.number_input(
        "Annual Income (USD)", min_value=0, value=50000, step=1000
    )
    person_home_ownership = st.selectbox(
        "Home Ownership", ["Rent", "Own", "Mortgage", "Other"]
    )
    person_emp_length = st.number_input(
        "Employment Length (Years)", min_value=0.0, max_value=50.0, value=5.0, step=1.0
    )

    st.header("Loan Information")
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "Personal",
            "Education",
            "Home Improvement",
            "Debt Consolidation",
            "Business",
            "Other",
        ],
    )
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount (USD)", min_value=0, value=10000, step=500)
    loan_int_rate = st.number_input(
        "Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1
    )
    loan_percent_income = round(loan_amnt / person_income, 2)

    st.header("Credit Information")
    cb_person_default_on_file = st.selectbox("Default on File", ["Yes", "No"])
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)", min_value=0, value=10, step=1
    )

    # Credit Score calculations:
    if st.button("Submit"):
        st.header("Result")
        input = pd.DataFrame(
            {
                "person_age": [person_age],
                "person_income": [person_income],
                "person_home_ownership": [person_home_ownership],
                "person_emp_length": [person_emp_length],
                "loan_intent": [loan_intent],
                "loan_grade": [loan_grade],
                "loan_amnt": [loan_amnt],
                "loan_int_rate": [loan_int_rate],
                "loan_percent_income": [loan_percent_income],
                "cb_person_default_on_file": [cb_person_default_on_file],
                "cb_person_cred_hist_length": [cb_person_cred_hist_length],
            }
        )
        score = int(scorecard_model.score(input)[0])
        st.write("Credit Score: ", score)


if __name__ == "__main__":
    main()
