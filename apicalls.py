import json
import os
import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


def run_api_calls():
    pred_req_str = URL + "/prediction?datapath=testdata/testdata.csv"
    response1 = requests.get(pred_req_str).content

    score_req_str = URL + "/scoring"
    response2 = requests.get(score_req_str).content

    stats_req_str = URL + "/summarystats"
    response3 = requests.get(stats_req_str).content

    diagnose_req_str = URL + "/diagnostics"
    response4 = requests.get(diagnose_req_str).content

    # combine all API responses
    responses = "\n".join([response1, response2, response3, response4])

    # write the responses to your workspace
    # write the combined outputs to a file call apireturns.txt.

    with open("config.json", "r") as f:
        config = json.load(f)
    api_return_path = os.path.join(config["output_model_path"],
                                   'apireturns.txt')
    with open(api_return_path, "w") as file:
        file.write(responses)


if __name__ == "__main__":
    run_api_calls()
