import argparse
import sys
#sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/')
import numpy as np
import mdtraj, os, tempfile, tqdm
# from betafold.utils import protein
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
import pandas as pd 
from multiprocessing import Pool,cpu_count
import numpy as np
from glob import glob
import torch
import pickle
import time
#sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/')


dir_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/liuce/data_10w/data_10w'

#for root, dirs, files in os.walk(dir_path):
#    print(dirs)
dirs = os.listdir(dir_path)
#print(dirs)
dirs.sort()

for d in dirs:
    print(d, time.strftime("%Y-%M-%D %H:%M:%S", time.localtime(time.time())))
    protein_name = d

    protein_path = os.path.join(dir_path, protein_name)
    #traj_path = os.path.join(protein_path, protein_name+'_post.dcd') 
    pdb_path = os.path.join(protein_path, protein_name+'.pdb')
    #save_path = os.path.join(protein_path, protein_name+'_new_w_pp.npz')
    force_path = os.path.join(protein_path, protein_name+'_F.pkl')
    vel_path = os.path.join(protein_path, protein_name+'_V.pkl')
    force_new_path = os.path.join(protein_path, protein_name+'_F_Ca.pkl')
    vel_new_path = os.path.join(protein_path, protein_name+'_V_ca.pkl')

    traj = mdtraj.load(pdb_path)
    ca_indices = traj.topology.select('name CA')
    ca_indices = np.array(ca_indices)[None, :, None]

    with open(force_path, 'rb') as f:
        force_feats = pickle.load(f)
    with open(vel_path, 'rb') as f:
        vel_feats = pickle.load(f)

    ca_indices = torch.from_numpy(ca_indices).expand(force_feats.shape[0], -1, force_feats.shape[2])
    force_feats = torch.gather(input=torch.from_numpy(force_feats), dim=1, index=ca_indices).numpy()
    vel_feats = torch.gather(input=torch.from_numpy(vel_feats), dim=1, index=ca_indices).numpy()

    with open(force_new_path, 'wb') as f: 
        pickle.dump(force_feats, f)
    with open(vel_new_path, 'wb') as f:
        pickle.dump(vel_feats, f)
    print(d, time.strftime("%Y-%M-%D %H:%M:%S", time.localtime(time.time())))


       
