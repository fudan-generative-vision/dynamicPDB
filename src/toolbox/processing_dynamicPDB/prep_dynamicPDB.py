import argparse
import sys
import mdtraj, os, tempfile, tqdm
# for openfold dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np

# from betafold.utils import protein
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
import pandas as pd 
from multiprocessing import Pool,cpu_count
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='applications/physics_condition/examples/dynamicPDB.csv')
parser.add_argument('--dynamic_dir', type=str, required=True)
parser.add_argument('--outdir', type=str, default='applications/physics_condition/data_dynamic/processed_npz')
parser.add_argument('--num_workers', type=int, default=cpu_count())
parser.add_argument('--simulation_suffix',type=str,default='npt1000000.0_ts0.001')

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='name')
# df = df[rank_idx::world_size]


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
    traj = mdtraj.load(os.path.join(args.dynamic_dir,f'{name}_{args.simulation_suffix}',f"{name}_T.dcd"),top=f'{args.dynamic_dir}/{name}_{args.simulation_suffix}/{name}.pdb')
    f, temp_path = tempfile.mkstemp(); os.close(f)
    positions_stacked = []
    # for i in tqdm.trange(0, len(traj), 3000): 
    # # process all trajectory is time-consuming maybe try some skip or cutoff
    for i in tqdm.trange(0, len(traj)//1, 1):
        traj[i].save_pdb(temp_path)
    
        with open(temp_path) as f:
            prot = protein.from_pdb_string(f.read())
            pdb_feats = make_protein_features(prot, name)
            positions_stacked.append(pdb_feats['all_atom_positions'])
            
    
    pdb_feats['all_atom_positions'] = np.stack(positions_stacked)
    np.savez(os.path.join(args.outdir,f"{name}.npz"), **pdb_feats)
    
    os.unlink(temp_path)
    
main()