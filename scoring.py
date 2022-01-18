import pandas as pd
import pickle
import os
from sklearn import metrics
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


# Function for model scoring
def score_model():
    """
    this function should take a trained model, load test data,
    and calculate an F1 score for the model relative to the test data
    """
    with open(model_path, 'rb') as modelf:
        model = pickle.load(modelf)

    test_df = pd.read_csv(test_data_path, index_col=0)
    X_test = test_df[[
        'lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_df['exited']

    y_preds = model.predict(X_test)
    # it should write the result to the latestscore.txt file
    f1_model = metrics.f1_score(y_test, y_preds)

    score_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    with open(score_path, 'w') as scoref:
        scoref.write(f'{f1_model}\n')

    return f1_model


if __name__ == '__main__':
    score_model()
