import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='src/toolbox/processing_atlas/data_csv/atlas.csv')
parser.add_argument('--processed_npz', type=str, default='./atlas/processed_npz')
parser.add_argument('--embeddings', type=str, default='./atlas/embeddings/OmegaFold')
parser.add_argument('--atlas_dir', type=str, default='./atlas/atlas_unzip')
parser.add_argument('--save_path', type=str, default='merged.csv')
args, _ = parser.parse_known_args()

atlas_csv = pd.read_csv(args.csv)
root = args.processed_npz
embed_root = args.embeddings

atlas_csv = atlas_csv.set_index('name')

npz_list = []
embed_path_list = []
seq_list = []
pdb_list_list = []
for pdb_name, row in tqdm(atlas_csv.iterrows()):
    npz_path = os.path.join(root,pdb_name+'.npz')
    embed_path = os.path.join(embed_root,pdb_name+'.npz')
    pdb_path = os.path.join(args.atlas_dir,pdb_name,pdb_name+'.pdb')
    seq_list.append(len(row.seqres))
    try:
        npz_list.append(npz_path)
        embed_path_list.append(embed_path)
        pdb_list_list.append(pdb_path)
    except RuntimeError as e:
        print(e)

atlas_csv['atlas_npz'] = npz_list
atlas_csv['embed_path'] = embed_path_list
atlas_csv['seq_len'] = seq_list
atlas_csv['pdb_path'] = pdb_list_list
atlas_csv.to_csv(args.save_path)