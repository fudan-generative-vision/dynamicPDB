import argparse
import sys
sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/')
import numpy as np
import mdtraj, os, tempfile, tqdm
# from betafold.utils import protein
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
import pandas as pd 
from multiprocessing import Pool,cpu_count
import numpy as np
from glob import glob


sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/')
# parser = argparse.ArgumentParser()
# parser.add_argument('--split', type=str, default='data_csv/atlas.csv')
# parser.add_argument('--atlas_dir', type=str, required=True)
# parser.add_argument('--outdir', type=str, default='./data_atlas')
# parser.add_argument('--num_workers', type=int, default=1)
# parser.add_argument("--rank_idx",type=int,help="Batch index to process, you may pass machine rank as it like $RANK")
# parser.add_argument("--world_size", type=int, help="num of batches to create like $WORLD_SIZE", required=False)

# args = parser.parse_args()
# # python prep_atlas.py --split=./splits/atlas.csv --atlas_dir=/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/atlas_unzip 
# # --outdir=/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/processed_npz --num_workers=16
# # /cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/atlas_unzip

# os.makedirs(args.outdir, exist_ok=True)
# df = pd.read_csv(args.split, index_col='name')
# df = df[args.rank_idx::args.world_size]


def main():
    jobs = []
    for name in df.index:
        jobs.append(name)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

def do_job(name):
    traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb')
    # ref = mdtraj.load(f'{args.atlas_dir}/{name}/{name}.pdb')
    # traj = ref + traj
    f, temp_path = tempfile.mkstemp(); os.close(f)
    positions_stacked = []
    # for i in tqdm.trange(0, len(traj), 3000):
    for i in tqdm.trange(0, len(traj), 1):
        traj[i].save_pdb(temp_path)
    
        with open(temp_path) as f:
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, name)
            # pdb_feats.update({'b_factors':prot.b_factors})
            positions_stacked.append(pdb_feats['all_atom_positions'])
            
    
    pdb_feats['all_atom_positions'] = np.stack(positions_stacked)
    print({key: pdb_feats[key].shape for key in pdb_feats})
    np.savez(f"{args.outdir}/{name}.npz", **pdb_feats)
    os.unlink(temp_path)
    



simulation_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/ef_simulation/ckh_4d_data/raw/4DFOLD_data/'
# simulation_traj='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/ef_simulation/simulate_protein/raw/4ue8_B_npt40_ts0.002_2024-07-10-03-41-52/npt.dcd'
# out_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/datasets/simulation_data/processed_npz/'
out_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/datasets/simulation_data/processed_npz/AAAI/'
atlas_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/atlas_unzip'
name_list = ['4ue8_B']

samples = glob(os.path.join(simulation_path,'*.dcd'))
sample_names = [os.path.basename(sample).split('_npt')[0] for sample in samples]

for name in sample_names:
    pdb_dir = f'{atlas_dir}/{name}/{name}.pdb'
    os.makedirs(f'{out_dir}/{name}_40ps_new',exist_ok=True)
    traj = mdtraj.load(os.path.join(simulation_path,f"{name}_npt.dcd"),top=pdb_dir)
    save_path = os.path.join(f'{out_dir}/{name}_40ps_new',f'{name}.npz')
    f, temp_path = tempfile.mkstemp(); os.close(f)
    positions_stacked = []
    print(save_path)
    for i in tqdm.trange(0, len(traj), 1):
        traj[i].save_pdb(temp_path)
        with open(temp_path) as f:  
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, name)
            positions_stacked.append(pdb_feats['all_atom_positions'])
    pdb_feats['all_atom_positions'] = np.stack(positions_stacked)
    np.savez(save_path, **pdb_feats)
