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
parser.add_argument('--split', type=str, default='applications/4d_diffusion/examples/atlas_visual_se3_filter.csv')
parser.add_argument('--atlas_dir', type=str, required=True)
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=cpu_count())
parser.add_argument('--traj_id', type=int, default=1)

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='name')
# df = df[rank_idx::world_size]


def main():
    jobs = []
    for name in df.index:
        #if os.path.exists(f'{args.outdir}/{name}.npz'): continue
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
    if args.traj_id ==1:
        traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') 
    elif args.traj_id ==2:
        traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') 
    elif args.traj_id ==3:
        traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') 
    else:
        traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb')
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
    np.savez(os.path.join(args.outdir,f"/{name}.npz"), **pdb_feats)
    os.unlink(temp_path)
    
main()