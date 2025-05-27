import os
import numpy as np
import mdtraj as md
from openfold.np import residue_constants
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features

pdb_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/atlas/atlas_unzip/2erl_A/2erl_A.pdb'
dcd_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/jwang/share/caizhiqiang/dynamic_pdb/simulate_test/raw/2erl_A_npt10000.0_ts0.001_2024-07-25-10-14-41/2erl_A_T.dcd'

npz_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/liuce/code/kaihui_force/2erl_A_new.npz'

read_data = dict(np.load(npz_path,allow_pickle=True))



traj = md.load(dcd_path,top = pdb_path)



all_position = np.zeros((traj.n_frames,traj.topology.n_residues,residue_constants.atom_type_num, 3))

# 遍历每个残基
for res_idx,residue in enumerate(traj.topology.residues): # 遍历每个residue

    # print(f"Residue {residue.name} (index {residue.index}):")
    # 提取残基的原子索引
    atom_indices = [atom.index for atom in residue.atoms]
    # 提取残基的轨迹
    residue_traj = traj.atom_slice(atom_indices)
    # 将 residue.atoms 转换为列表
    atoms = list(residue.atoms)
    # 获取residue的缩写名称
    res_shortname = residue_constants.restype_3to1.get(residue.name)
    # 获取residue在openfold的index 
    restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
    for time_idx, frame in enumerate(residue_traj.xyz): # 遍历每个时间 
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for j, atom_coord in enumerate(frame): #遍历每个时间点上氨基酸的atom

            atom_name = atoms[j].name

            if atom_name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom_name]] =  np.round(atom_coord*10, 3) # 
            mask[residue_constants.atom_order[atom_name]] = 1.0

            res_b_factors[
                residue_constants.atom_order[atom_name]
            ] = 1.00
        all_position[time_idx,res_idx]=pos

topology=md.load(pdb_path)
topology.save('tmp.pdb')
with open('tmp.pdb') as f:  
    prot = protein.from_pdb_string(f.read())
    pdb_feats = make_protein_features(prot, 'name')
# print(pdb_feats)
# print(np.mean(all_position[0,:,1]*10-read_data['all_atom_positions'][0,:,1]))
pdb_feats['all_atom_positions'] = all_position

