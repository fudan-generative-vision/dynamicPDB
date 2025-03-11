import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse

def check_file_exists(file_path):
    return os.path.isfile(file_path)

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='applications/physics_condition/examples/dynamicPDB.csv')
parser.add_argument('--processed_npz', type=str, default='./examples/data_dynamic/processed_npz')
parser.add_argument('--embeddings', type=str, default='./examples/data_dynamic/embeddings')
parser.add_argument('--dynamicPDB_dir', type=str, default='./examples/samples')
parser.add_argument('--save_path', type=str, default='merged.csv')
parser.add_argument('--simulation_suffix',type=str,default='npt1000000.0_ts0.001')
args, _ = parser.parse_known_args()

atlas_csv = pd.read_csv(args.csv)
root = args.processed_npz
embed_root = args.embeddings

atlas_csv = atlas_csv.set_index('name')

npz_list = []
embed_path_list = []
seq_list = []
pdb_list_list = []
vel_path_list = []
force_path_list = []
for pdb_name, row in tqdm(atlas_csv.iterrows()):
    traj_npz_path = os.path.join(root,pdb_name+'.npz')
    embed_path = os.path.join(embed_root,pdb_name+'.npz')
    pdb_path = os.path.join(args.dynamicPDB_dir,f'{pdb_name}_{args.simulation_suffix}',pdb_name+'.pdb')
    force_path = os.path.join(args.dynamicPDB_dir,f'{pdb_name}_{args.simulation_suffix}',pdb_name+'_F_Ca.pkl')
    vel_path = os.path.join(args.dynamicPDB_dir,f'{pdb_name}_{args.simulation_suffix}',pdb_name+'_V_Ca.pkl')

    # check_list = [traj_npz_path,embed_path,pdb_path,force_path,vel_path]
    # for path in check_list:
    #     if not check_file_exists(path):
    #         print(f"file not exist:{path}")
    vel_path_list.append(vel_path)
    force_path_list.append(force_path)
    seq_list.append(len(row.seqres))
    try:
        npz_list.append(traj_npz_path)
        embed_path_list.append(embed_path)
        pdb_list_list.append(pdb_path)
    except RuntimeError as e:
        print(e)

atlas_csv['dynamic_npz'] = npz_list
atlas_csv['embed_path'] = embed_path_list
atlas_csv['seq_len'] = seq_list
atlas_csv['pdb_path'] = pdb_list_list
atlas_csv['vel_path'] = vel_path_list
atlas_csv['force_path'] = force_path_list
atlas_csv.to_csv(args.save_path)