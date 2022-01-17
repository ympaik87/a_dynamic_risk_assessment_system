import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def train_model():
    '''
    Function for training the model
    '''
    data_df = pd.read_csv(dataset_csv_path, index_col=0)
    X_df = data_df[[
        'lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_df = data_df['exited']
    # use this logistic regression for training
    clf = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )

    # fit the logistic regression to your data
    clf.fit(X_df, y_df)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(model_path, 'wb') as modelf:
        pickle.dump(clf, modelf)


if __name__ == '__main__':
    train_model()
