import pathlib
import json
import sys
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicalls


with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = pathlib.Path(config['output_folder_path'])
input_folder_path = pathlib.Path(config["input_folder_path"])
prod_deployment_path = pathlib.Path(config['prod_deployment_path'])
output_model_path = pathlib.Path(config['output_model_path'])

# Check and read new data
# first, read ingestedfiles.txt
ingested_fpath = output_folder_path/'ingestedfiles.txt'

ingested_f_li = []
with open(ingested_fpath, 'r') as file:
    for line in file:
        ingested_f_li.append(line.strip())


# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_file_li = sorted(input_folder_path.glob('*.csv'))
new_file_paths = []
for filepath in input_file_li:
    if filepath.name not in ingested_f_li:
        new_file_paths.append(filepath)

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_file_paths) == 0:
    print('No new files to ingest. Exiting..')
    sys.exit(0)

ingestion.merge_multiple_dataframe()

# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

scoring.score_model()

with open(prod_deployment_path/'latestscore.txt', 'r') as file:
    prev_score = float(file.read())

with open(output_folder_path/'latestscore.txt', 'r') as file:
    current_score = float(file.read())

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here

if current_score >= prev_score:
    sys.exit(0)

training.train_model()


# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

deployment.store_model_into_pickle()

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model

diagnostics.model_predictions()
diagnostics.dataframe_summary()
diagnostics.execution_time()
diagnostics.outdated_packages_list()
reporting.score_model(plot_fname='confusionmatrix2.png')

apicalls.run_api_calls(apireturns_fname='apireturns2.txt')