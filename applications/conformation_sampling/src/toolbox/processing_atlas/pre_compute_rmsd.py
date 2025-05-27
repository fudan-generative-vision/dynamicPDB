import numpy as np
from scipy.spatial.transform import Rotation as R
import mdtraj
import argparse
import os,tqdm
from multiprocessing import Pool,cpu_count

def calculate_aligned_rmsd(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    A_rotated = A_centered @ R
    rmsd = np.sqrt(np.mean(np.linalg.norm(A_rotated - B_centered, axis=1)**2))
    return rmsd

def farthest_point_sampling(proteins, M):
    F, N, _ = proteins.shape
    selected_indices = [np.random.randint(F)]
    selected_proteins = [proteins[selected_indices[-1]]]
    
    for _ in range(M - 1):
        max_dist = -np.inf
        farthest_idx = None
        for i in range(F):
            if i in selected_indices:
                continue
            min_dist = np.inf
            for sp in selected_proteins:
                dist = calculate_aligned_rmsd(proteins[i], sp)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > max_dist:
                max_dist = min_dist
                farthest_idx = i
        selected_indices.append(farthest_idx)
        selected_proteins.append(proteins[farthest_idx])
    
    return np.array(selected_indices).reshape(-1, 1)



def load_traj(pdb_name,atlas_dir,traj_id=1):
    # load traj
    print(f'{atlas_dir}/{pdb_name}/{pdb_name}_prod_R{traj_id}_fit.xtc')
    traj = mdtraj.load(f'{atlas_dir}/{pdb_name}/{pdb_name}_prod_R{traj_id}_fit.xtc', top=f'{atlas_dir}/{pdb_name}/{pdb_name}.pdb') 
    # get ca index
    ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
    # get ca_position
    ca_positions = traj.atom_slice(ca_indices).xyz
    # convert nm to angs
    return ca_positions*10



def do_job(pdb_name):
    ca_positions = load_traj(pdb_name,args.atlas_dir)
    ca_nums = len(ca_positions)
    rmsd_dist_matrix = np.full((ca_nums, ca_nums), -np.inf)
    for i in range(ca_nums):
        for j in range(ca_nums):
            rmsd_dist = calculate_aligned_rmsd(ca_positions[i], ca_positions[j])
            rmsd_dist_matrix[i,j] = rmsd_dist

    np.save(f"{args.outdir}/{pdb_name}.npz", rmsd_dist_matrix)
    return rmsd_dist_matrix

# multi-processors
def main():
    jobs = []
    
        #if os.path.exists(f'{args.outdir}/{name}.npz'): continue
    for pdb_name in os.listdir(args.atlas_dir):
        jobs.append(pdb_name)

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

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='data_csv/atlas.csv')
parser.add_argument('--atlas_dir', type=str, default='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/atlas/atlas_unzip/')
parser.add_argument('--outdir', type=str, default='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/atlas/ca_rmsd_matrix')
parser.add_argument('--num_workers', type=int, default=cpu_count())
parser.add_argument("--rank_idx",type=int,help="Batch index to process, you may pass machine rank as it like $RANK")
parser.add_argument("--world_size", type=int, help="num of batches to create like $WORLD_SIZE", required=False)

args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)


pdb_name = '1k5n_A'
do_job('1k5n_A')
# print(ca_positions.shape)
# print(calculate_aligned_rmsd(ca_positions[0], ca_positions[1]))
