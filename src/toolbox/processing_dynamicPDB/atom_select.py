
import numpy as np
import mdtraj
import numpy as np
import torch
import pickle
import time
import os
import argparse
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--dynamic_dir', type=str, required=True)
args = parser.parse_args()
dir_path = args.dynamic_dir


dirs = os.listdir(dir_path)

dirs.sort()

for d in dirs:

    print(d, time.strftime("%Y-%M-%D %H:%M:%S", time.localtime(time.time())))
    protein_name = "_".join(d.split("_")[:2])


    protein_path = os.path.join(dir_path, d)
    pdb_path = os.path.join(protein_path, protein_name+'.pdb')
    force_path = os.path.join(protein_path, protein_name+'_F.pkl')
    vel_path = os.path.join(protein_path, protein_name+'_V.pkl')
    force_new_path = os.path.join(protein_path, protein_name+'_F_Ca.pkl')
    vel_new_path = os.path.join(protein_path, protein_name+'_V_Ca.pkl')

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


       
