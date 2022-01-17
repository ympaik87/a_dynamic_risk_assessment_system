import os
import pathlib
import json
import pandas as pd


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    file_li = sorted(pathlib.Path(input_folder_path).glob('*.csv'))

    df_li = []
    for fpath in file_li:
        df_li.append(pd.read_csv(fpath))

    dataset = pd.concat(df_li, ignore_index=True)
    dataset.drop_duplicates(inplace=True)

    output_fpath = os.path.join(output_folder_path, 'finaldata.csv')
    dataset.to_csv(output_fpath, index=False)

    output_record_fpath = os.path.join(output_folder_path, 'ingestedfiles.txt')
    file_str = ','.join([str(fp.name) for fp in file_li])
    with open(output_record_fpath, 'w') as file:
        file.write(file_str)


if __name__ == '__main__':
    merge_multiple_dataframe()
