import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
FEATURES = [
    'lastmonth_activity', 'lastyear_activity', 'number_of_employees'
]


def score_model(plot_fname='confusionmatrix.png'):
    '''
    Function for reporting.
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    '''
    plot_path = os.path.join(config['output_model_path'], plot_fname)
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    test_df = pd.read_csv(test_data_path, index_col=0)
    with open(model_path, 'rb') as modelf:
        model = pickle.load(modelf)
    X_test = test_df[FEATURES]
    y_test = test_df['exited']

    metrics.plot_confusion_matrix(model, X_test, y_test)
    plt.title('Logistic Regression Confusion Matrix')
    plt.savefig(plot_path)


if __name__ == '__main__':
    score_model()
