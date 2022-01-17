import pickle
import pandas as pd
import timeit
import os
import json
import subprocess

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
test_df = pd.read_csv(test_data_path, index_col=0)
FEATURES = [
    'lastmonth_activity', 'lastyear_activity', 'number_of_employees'
]


def model_predictions(dataset_df=test_df):
    """
    Function to get model predictions
    Read the deployed model and a test dataset, calculate predictions
    """
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    with open(model_path, 'rb') as modelf:
        model = pickle.load(modelf)
    X_test = dataset_df[FEATURES]
    # y_test = dataset_df['exited']
    y_preds = model.predict(X_test)
    return y_preds.tolist()


def dataframe_summary():
    """
    Function to get summary statistics
    calculate summary statistics here
    """
    X_test = test_df[FEATURES]
    X_test_desc = X_test.describe()
    res_li = []
    for col in FEATURES:
        for row in ['mean', '50%', 'std']:
            res_li.append([col, row, X_test_desc.loc[row, col]])
    # return value should be a list containing all summary statistics
    return res_li


def missing_data():
    total_count = len(test_df)
    res_li = []
    for col in FEATURES:
        na_count = test_df[col].isna().sum()
        res_li.append([col, na_count/total_count])
    return res_li


def execution_time():
    """
    Function to get timings
    calculate timing of training.py and ingestion.py
    """
    result_li = []
    for script_name in ["training.py", "ingestion.py"]:
        start_time = timeit.default_timer()
        os.system(f'python {script_name}')
        exec_time = timeit.default_timer() - start_time
        result_li.append([script_name, str(round(exec_time, 2))])
    return  result_li


def outdated_packages_list():
    """
    Function to check dependencies
    get a list of
    """
    cmd = subprocess.Popen(
        'pip list --outdated', shell=True, stdout=subprocess.PIPE)
    for line in cmd.stdout:
        yield line


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
