import argparse
import sys
sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/DFOLDv2/')
import numpy as np
import mdtraj, os, tempfile, tqdm
# from betafold.utils import protein
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features
import pandas as pd 
from multiprocessing import Pool,cpu_count
import MDAnalysis as mda
from MDAnalysis.analysis import rms,align,rdf,contacts
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def calculate_rmsd(u, reference_select="protein and name CA"):
    reference = u.select_atoms(reference_select)
    rmsd_analysis = rms.RMSD(u, reference, select=reference_select)
    rmsd_analysis.run()
    return rmsd_analysis.rmsd

def calculate_rmsf(u, reference_select="protein and name CA"):
    atoms = u.select_atoms(reference_select)
    rmsf_analysis = rms.RMSF(atoms).run()
    return rmsf_analysis.rmsf

def calculate_rdf(u, select1="name O", select2="name H"):
    group1 = u.select_atoms(select1)
    group2 = u.select_atoms(select2)
    rdf_analysis = rdf.InterRDF(group1, group2)
    rdf_analysis.run()
    return rdf_analysis.bins, rdf_analysis.rdf

def align_trajectories(u, reference_select="protein and name CA"):
    reference = u.select_atoms(reference_select)
    aligner = align.AlignTraj(u, reference, select=reference_select, in_memory=True)
    aligner.run()
    return u

def calculate_hydrogen_bonds(u, selection="protein"):
    hbonds_analysis = contacts.HydrogenBondAnalysis(u, selection, selection)
    hbonds_analysis.run()
    return hbonds_analysis.timeseries, hbonds_analysis.count_by_time()

def process_sequence(sequence):
    return len(sequence)

# 计算两两之间的差异
def calculate_differences(rmsf_list):
    num_elements = len(rmsf_list)
    differences = []

    for i in range(num_elements):
        for j in range(i + 1, num_elements):
            diff = rmsf_list[i] - rmsf_list[j]
            differences.append(np.mean(diff**2))
    
    return differences

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='src/toolbox/processing_atlas/atlas.csv')
parser.add_argument('--atlas_dir', type=str, default='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/atlas_unzip')
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument("--rank_idx",type=int,help="Batch index to process, you may pass machine rank as it like $RANK")
parser.add_argument("--world_size", type=int, help="num of batches to create like $WORLD_SIZE", required=False)

args = parser.parse_args()
# python prep_atlas.py --split=./splits/atlas.csv --atlas_dir=/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/atlas_unzip --outdir=/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/processed_npz --num_workers=16

# 
# os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='name')
df = df[args.rank_idx::args.world_size]
df['seq_len'] = df['seqres'].apply(process_sequence)
# df = df[df.seq_len <= 256]

jobs = []
for name in df.index:
    jobs.append(name)

rmsf_pearson_correlation = []
rmsd_pearson_correlation = []

rmsf_mse_dist = []
rmsd_mse_dist = []

for name in tqdm.tqdm(jobs,total=len(jobs)):
    data = dict(np.load(f'./atlas_analysis/{name}.npz',allow_pickle=True))
    rmsf = data['rmsf']
    rmsd = data['rmsd'][:,1:,:]
    


    rmsf_mse = calculate_differences(list(rmsf))
    rmsd_mse = calculate_differences(list(rmsd[...,2]))

    rmsf_mse_dist.append(sum(rmsf_mse) / len(rmsf_mse))
    rmsd_mse_dist.append(sum(rmsd_mse) / len(rmsd_mse))

    # pearsonr on rmsf
    corr12, _ = pearsonr(rmsf[0], rmsf[1])
    corr13, _ = pearsonr(rmsf[0], rmsf[2])
    corr23, _ = pearsonr(rmsf[1], rmsf[2])

    mean_corr_rmsf = (corr12+corr13+corr23)/3.
    rmsf_pearson_correlation.append(mean_corr_rmsf)
    # pearsonr on rmsf
    corr12, _ = pearsonr(rmsd[0,:,2], rmsd[1,:,2])
    corr13, _ = pearsonr(rmsd[0,:,2], rmsd[2,:,2])
    corr23, _ = pearsonr(rmsd[1,:,2], rmsd[2,:,2])

    mean_corr_rmsd = (corr12+corr13+corr23)/3.
    rmsd_pearson_correlation.append(mean_corr_rmsd)


# 计算均值和方差
rmsd_mean = np.mean(np.array(rmsd_pearson_correlation))
rmsd_var = np.var(np.array(rmsd_pearson_correlation))
rmsf_mean = np.mean(np.array(rmsf_pearson_correlation))
rmsf_var = np.var(np.array(rmsf_pearson_correlation))

# 创建绘图
plt.figure(dpi=300, figsize=(10, 6))

# 子图1: RMSD Pearson Correlation
plt.subplot(221)
plt.plot(rmsd_pearson_correlation, label='RMSD Pearson Correlation')
plt.title(f"RMSD: {round(rmsd_mean, 3)} ± {round(rmsd_var, 3)}")
plt.xlabel('Index')
plt.ylabel('Correlation')
plt.legend(loc='upper left')
plt.grid(True)
plt.text(0.6, 0.9, f"Mean: {round(rmsd_mean, 3)}\nVariance: {round(rmsd_var, 3)}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white'))

# 子图2: RMSF Pearson Correlation
plt.subplot(222)
plt.plot(rmsf_pearson_correlation, label='RMSF Pearson Correlation', color='orange')
plt.title(f"RMSF: {round(rmsf_mean, 3)} ± {round(rmsf_var, 3)}")
plt.xlabel('Index')
plt.ylabel('Correlation')
plt.legend(loc='upper left')
plt.grid(True)
plt.text(0.6, 0.9, f"Mean: {round(rmsf_mean, 3)}\nVariance: {round(rmsf_var, 3)}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white'))


# 计算均值和方差
rmsd_mse_mean = np.mean(np.array(rmsd_mse_dist))
rmsd_mse_var = np.var(np.array(rmsd_mse_dist))
rmsf_mse_mean = np.mean(np.array(rmsf_mse_dist))
rmsf_mse_var = np.var(np.array(rmsf_mse_dist))

# 子图1: RMSD MSE
plt.subplot(223)
plt.plot(rmsd_mse_dist, label='RMSD MSE')
plt.title(f"RMSD: {round(rmsd_mse_mean, 3)} ± {round(rmsd_mse_var, 3)}")
plt.xlabel('Index')
plt.ylabel('MSE')
plt.legend(loc='upper left')
plt.grid(True)
plt.text(0.6, 0.9, f"Mean: {round(rmsd_mse_mean, 3)}\nVariance: {round(rmsd_mse_var, 3)}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white'))

# 子图2: RMSF MSE
plt.subplot(224)
plt.plot(rmsf_mse_dist, label='RMSF MSE', color='orange')
plt.title(f"RMSF: {round(rmsf_mse_mean, 3)} ± {round(rmsf_mse_var, 3)}")
plt.xlabel('Index')
plt.ylabel('MSE')
plt.legend(loc='upper left')
plt.grid(True)
plt.text(0.6, 0.9, f"Mean: {round(rmsf_mse_mean, 3)}\nVariance: {round(rmsf_mse_var, 3)}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white'))


# 保存图像
plt.tight_layout()
plt.savefig('rmsf_rmsd_pearson_correlation_and_mse.png')
exit()







all_ave_rmsf_diff = []
all_ave_rmsd_diff = []
for name in tqdm.tqdm(jobs,total=len(jobs)):
    pdf_file_path = f'{args.atlas_dir}/{name}/{name}.pdb'
    first_traj_path = f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc'
    second_traj_path = f'{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc'
    third_traj_path = f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc'
    # 加载轨迹
    u1 = mda.Universe(pdf_file_path, first_traj_path)
    u2 = mda.Universe(pdf_file_path, second_traj_path)
    u3 = mda.Universe(pdf_file_path, third_traj_path)
    # 计算轨迹的RMSF
    rmsf1_data = calculate_rmsf(u1,reference_select="protein and name CA")
    rmsf2_data = calculate_rmsf(u2,reference_select="protein and name CA")
    rmsf3_data = calculate_rmsf(u3,reference_select="protein and name CA")
    # (N,)
    rmsf_list = [rmsf1_data,rmsf2_data,rmsf3_data]
    diff_list = calculate_differences(rmsf_list)



    all_ave_rmsf_diff.append(sum(diff_list) / len(diff_list))
    # 计算轨迹的RMSD
    rmsd1_data = calculate_rmsd(u1,reference_select="protein and name CA")
    rmsd2_data = calculate_rmsd(u2,reference_select="protein and name CA")
    rmsd3_data = calculate_rmsd(u3,reference_select="protein and name CA")
    rmsd_list = [rmsd1_data,rmsd2_data,rmsd3_data]

    np.savez(f'./atlas_analysis/{name}.npz', rmsf=np.array(rmsf_list), rmsd=np.array(rmsd_list))
    diff_list = calculate_differences(rmsd_list)
    rmsd_list = [ diff[1:].mean(-1) for diff in diff_list] # 1: out the ref
    all_ave_rmsd_diff.append(sum(rmsd_list) / len(rmsd_list))


plt.figure(dpi=300)
plt.plot(all_ave_rmsf_diff)
plt.savefig('ave_rmsf.png')
plt.figure(dpi=300)
plt.plot(all_ave_rmsd_diff)
plt.savefig('ave_rmsd.png')

exit()





# print(u1.coord._pos.shape,u2.coord._pos.shape)
# 选择要提取坐标的原子（例如，蛋白质所有原子）

# selection = u1.select_atoms("protein and name CA")
# 初始化一个数组来存储所有帧的坐标
# n_atoms = len(selection)
# n_frames = len(u1.trajectory)
# 遍历轨迹并输出坐标
# for ts in u1.trajectory:
#     print(f"Frame {ts.frame}:{selection.positions[:10]}")
#     print('='*10)
#     if ts.frame ==2:
#         break
    

# print(n_atoms,n_frames)
# print(u1.trajectory[0])
# print(selection.positions[:10])
# selection = u2.select_atoms("protein and name CA")
# for ts in u2.trajectory:
#     print(f"Frame {ts.frame}:{selection.positions[:10]}")
#     print('='*10)
#     if ts.frame ==2:
#         break
# exit()
# print('='*10)
# print(selection.positions[:10])
# exit()
# all_coords = np.zeros((n_frames, n_atoms, 3))

# # 遍历轨迹并提取坐标
# for i, ts in enumerate(u.trajectory):
#     all_coords[i] = selection.positions

# # 输出所有帧的坐标
# print(all_coords)
# 
# exit()

rmsf1_data = calculate_rmsf(u1,reference_select='protein and name CA')
rmsf2_data = calculate_rmsf(u2,reference_select='protein and name CA')
rmsf3_data = calculate_rmsf(u3,reference_select='protein and name CA')
rmsf_list = [rmsf1_data,rmsf2_data,rmsf3_data]
plt.figure(figsize=(10, 6))
for i in range(len(rmsf_list)):
    rmsf_data = rmsf_list[i]
    plt.plot(rmsf_data, label=f"Trajectory {i+1} RMSF")
plt.xlabel("Residue index")
plt.ylabel("RMSF (Å)")
plt.legend()
plt.savefig('rmsf_test.jpg')


plt.figure(figsize=(10, 6))
# 计算轨迹1的RMSD
rmsd1_data = calculate_rmsd(u1,reference_select="protein and name CA")
# 计算轨迹2的RMSD
rmsd2_data = calculate_rmsd(u2,reference_select="protein and name CA")
# 计算轨迹3的RMSD
rmsd3_data = calculate_rmsd(u3,reference_select="protein and name CA")
rmsd_list = [rmsd1_data,rmsd2_data,rmsd3_data]

# 可视化RMSD比较


plt.figure(figsize=(10, 6))
for i in range(len(rmsd_list)):
    rmsd_data = rmsd_list[i]
    plt.plot(rmsd_data[:, 1], rmsd_data[:, 2], label=f"Trajectory {i+1} RMSD")
plt.xlabel("Time (ns)")
plt.ylabel("RMSD (Å)")
plt.legend()
plt.savefig('rmsd_test.jpg')
exit()


do_job('1k5n_A')
exit()


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
    traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') \
        # + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') \
        # + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb')
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



main()