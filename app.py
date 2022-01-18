from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import (
    model_predictions, dataframe_summary, missing_data, execution_time,
    outdated_packages_list
)
from scoring import score_model
import json
import os


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Prediction Endpoint.

    call the prediction function you created in Step 3
    """
    datapath = request.args.get("datapath")
    pred_li = model_predictions(datapath)
    return jsonify(pred_li)


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    Scoring Endpoint.

    check the score of the deployed model
    and add return value (a single F1 score number)
    """
    scores = score_model()
    return jsonify(scores)


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Summary Statistics Endpoint.

    check means, medians, and modes for each column
    and return a list of all calculated summary statistics
    """
    summary = dataframe_summary()
    return jsonify(summary)


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnose():
    """
    Diagnostics Endpoint.

    check timing and percent NA values
    and add return value for all diagnostics
    """
    res = [execution_time(), missing_data(), outdated_packages_list()]
    return jsonify(res)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
