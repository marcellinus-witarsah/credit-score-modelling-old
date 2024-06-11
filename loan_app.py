import streamlit as st
import requests


def main():
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
    # loan_percent_income = st.number_input(
    #     "Loan Amount as Percentage of Income",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=0.0,
    #     step=0.1,
    # )
    loan_percent_income = round(loan_amnt / person_income, 2)

    st.header("Credit Information")
    cb_person_default_on_file = st.selectbox("Default on File", ["Yes", "No"])
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)", min_value=0, value=10, step=1
    )

    if st.button("Submit"):
        st.header("Credit Score")
        ENDPOINT = "http://127.0.0.1:8000/credit-score"
        input = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }
        prediction = requests.post(
            url=ENDPOINT, json=input, headers={"Content-Type": "application/json"}
        )
        result = prediction.json()
        st.write(result["credit_score"])


if __name__ == "__main__":
    main()
