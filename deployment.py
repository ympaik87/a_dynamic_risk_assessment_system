import shutil
import os
import json


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])
ingested_path = os.path.join(
    config['output_folder_path'], 'ingestedfiles.txt')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
score_path = os.path.join(config['output_model_path'], 'latestscore.txt')


# function for deployment
def store_model_into_pickle():
    """
    copy the latest pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    """
    file_p_li = [ingested_path, model_path, score_path]
    for file_p in file_p_li:
        shutil.copy2(file_p, prod_deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()
