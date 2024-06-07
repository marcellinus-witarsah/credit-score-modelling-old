import mlflow
import os
import pandas as pd
from sklearn.base import BaseEstimator
from src.utils.common import logger
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


def load_model(model_name: str, version: int) -> BaseEstimator:
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        return model
    except Exception as e:
        logger.error(e)


model = load_model("WeightOfEvidence+LogisticRegression", 3)


@app.route("/", methods=["POST"])
def predict():
    result = None
    json_data = request.get_json()
    data = pd.DataFrame(json_data["input"])
    result = {"credit_score": model.score(data).tolist()}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
