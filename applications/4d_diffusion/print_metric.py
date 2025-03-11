import os
import numpy as np
import pandas as pd
import pickle
from glob import glob
import argparse

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

parser = argparse.ArgumentParser(description="Process PKL files and compute metrics.")
parser.add_argument('--metric_path', type=str, required=True, help="Path to the directory containing PKL files.")
args = parser.parse_args()

pkl_path_list = glob(os.path.join(args.metric_path, "*.pkl"))

protein_metric_list = []

for pkl_path in pkl_path_list:
    pdb_name = os.path.basename(pkl_path).replace('.pkl', '')
    data_list = read_pickle(pkl_path)
    rmse_all = []
    rmse_ca = []

    for data in data_list:
        rmse_all.append(data['rmse_all'])
        rmse_ca.append(data['rmse_ca'])

    protein_metric = {
        'name': pdb_name,
        'rmse_all_mean': np.mean(rmse_all),
        'rmse_all_std': np.std(rmse_all),
        'rmse_all_median': np.median(rmse_all),
        'rmse_ca_mean': np.mean(rmse_ca),
        'rmse_ca_std': np.std(rmse_ca),
        'rmse_ca_median': np.median(rmse_ca)
    }
    protein_metric_list.append(protein_metric)

df = pd.DataFrame(protein_metric_list)
print('=' * 10)
print(df)
print('=' * 10)
print(df.mean(numeric_only=True))
