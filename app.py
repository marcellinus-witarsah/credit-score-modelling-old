import mlflow
import os
import pandas as pd
from src.utils.common import logger
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load model:
model_name = "credit-score-model"
version = 1
try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
except Exception as e:
    logger.error(e)


# Create class containing input data schema for validation:
class LoanApplicant(BaseModel):
    """Schema for data validation"""

    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


app = FastAPI()


# Default endpoint:
@app.get("/")
def index():
    return {"message": "Credit Scorecard Model API"}


# Get credit score:
@app.post("/credit-score")
def calculate_credit_score(data: LoanApplicant):
    data = data.dict()
    input = pd.DataFrame(
        {
            "person_age": [data["person_age"]],
            "person_income": [data["person_income"]],
            "person_home_ownership": [data["person_home_ownership"]],
            "person_emp_length": [data["person_emp_length"]],
            "loan_intent": [data["loan_intent"]],
            "loan_grade": [data["loan_grade"]],
            "loan_amnt": [data["loan_amnt"]],
            "loan_int_rate": [data["loan_int_rate"]],
            "loan_percent_income": [data["loan_percent_income"]],
            "cb_person_default_on_file": [data["cb_person_default_on_file"]],
            "cb_person_cred_hist_length": [data["cb_person_cred_hist_length"]],
        }
    )

    credit_score = model.score(input)
    return {"credit_score": credit_score[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
