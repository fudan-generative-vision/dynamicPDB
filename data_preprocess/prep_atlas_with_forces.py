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


#sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/')

#traj_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/jwang/share/caizhiqiang/dynamic_pdb/simulate/raw/4ue8_B_npt10000.0_ts0.001_2024-07-24-13-20-51/4ue8_B_T.dcd'
#traj_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/jwang/share/caizhiqiang/dynamic_pdb/simulate_0.2ps/raw/2erl_A_npt2000.0_ts0.001_2024-07-26-07-22-21/2erl_A_T.dcd'
traj_path = './2erl_A_post.dcd'

#pdb_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/jwang/share/caizhiqiang/dynamic_pdb/simulate/raw/4ue8_B_npt10000.0_ts0.001_2024-07-24-13-20-51/4ue8_B_minimized.pdb'
pdb_path = './2erl_A.pdb'
#samples = glob(os.path.join(simulation_path,'*.dcd'))
#sample_names = [os.path.basename(sample).split('_npt')[0] for sample in samples]

traj = mdtraj.load(traj_path,top=pdb_path)
save_path = os.path.join('./2erl_A_new_w_pp.npz')
f, temp_path = tempfile.mkstemp(); os.close(f)
positions_stacked = []
print(save_path)
for i in tqdm.trange(0, len(traj), 1):
    traj[i].save_pdb(temp_path)
    with open(temp_path) as f:  
        prot = protein.from_pdb_string(f.read())
        pdb_feats = make_protein_features(prot, '2erl_A')
        positions_stacked.append(pdb_feats['all_atom_positions'])
pdb_feats['all_atom_positions'] = np.stack(positions_stacked)
np.savez(save_path, **pdb_feats)
