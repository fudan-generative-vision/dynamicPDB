"""PDB dataset loader."""
import sys
sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/StateFold')
import math
from typing import Optional
from src.experiments import utils as eu
import torch
import torch.distributed as dist
import os
import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn
from src.data import se3_diffuser
from torch.utils import data
from src.data import utils as du
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils
# from data import pdb_data_loader
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import mdtraj
import time
# def parse_dynamics_chain_feats_norm(chain_feats, scale_factor=1.):
#     ca_idx = residue_constants.atom_order['CA']
#     chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
#     bb_pos = chain_feats['all_atom_positions'][0,:, ca_idx] # [F,N,37,3]->[N,3] select first protein as anchor
#     bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [3]
#     mn = np.mean(chain_feats['all_atom_positions'], axis=0, keepdims=True)
#     sd = np.std(chain_feats['all_atom_positions'], axis=0, keepdims=True)
#     chain_feats['all_atom_positions'] = (chain_feats['all_atom_positions'] - mn) / (sd+1e-6)

#     centered_pos = chain_feats['all_atom_positions'] - 0.0 * bb_center[None, None, None, :] # [F,N,37,3]
#     chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]
#     return chain_feats

def parse_dynamics_chain_feats_with_ref(chain_feats, first_frame, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']

    first_frame['bb_mask'] = first_frame['all_atom_mask'][:, ca_idx] # [37]
    ref_bb_pos = first_frame['all_atom_positions'][:, ca_idx] # [N,37,3]->[N,3] select C-alpha as anchor
    ref_bb_center = np.sum(ref_bb_pos, axis=0) / (np.sum(first_frame['bb_mask']) + 1e-5) # [3]

    # normalize the datasets
    centered_pos = chain_feats['all_atom_positions'] - ref_bb_center[None, None, None, :] # [F,N,37,3]
    scaled_pos = centered_pos / scale_factor
    chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx]

    return chain_feats

def parse_dynamics_chain_feats(chain_feats, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
    bb_pos = chain_feats['all_atom_positions'][0,:, ca_idx] # [F,N,37,3]->[N,3] select first protein as anchor
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [3]
    centered_pos = chain_feats['all_atom_positions'] - bb_center[None, None, None, :] # [F,N,37,3]
    scaled_pos = centered_pos / scale_factor
    chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]
    return chain_feats


def calculate_rotation(A_centered, B_centered):
    #centroid_A = np.mean(A, axis=0, keepdims=True)
    #centroid_B = np.mean(B, axis=0, keepdims=True)
    #A_centered = A - centroid_A
    #B_centered = B - centroid_B
    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    #A_rotated = A_centered @ R
    return R.T

def parse_dynamics_chain_feats_with_align(chain_feats, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
    #print(chain_feats['bb_mask'][np.newaxis,:,np.newaxis].shape)
    #exit()
    bb_pos = chain_feats['all_atom_positions'][:, :, ca_idx] #[F,N,3]
    bb_center = np.sum(bb_pos, axis=1, keepdims=True) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [3]
    bb_pos = bb_pos - bb_center
    #print(bb_center.shape)
    #exit()
    rs = []
    for i in range(bb_pos.shape[0]):
        ri = calculate_rotation(bb_pos[i], bb_pos[0])
        rs.append(ri)


    centered_pos = chain_feats['all_atom_positions'] - bb_center[:, :, None, :] # [F,N,37,3]
    scaled_pos = centered_pos / scale_factor
    for i in range(scaled_pos.shape[0]):
        for j in range(scaled_pos.shape[1]):
            for k in range(37):
                scaled_pos[i,j,k] = scaled_pos[i,j,k] @ rs[i]
                #print(scaled_pos[i,j,k].shape)
                #exit()

    chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]
    return chain_feats

def parse_dynamics_chain_feats_no_norm(chain_feats, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
    chain_feats['all_atom_positions'] = chain_feats['all_atom_positions'] * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]

    return chain_feats

def parse_dynamics_chain_feats_split(chain_feats, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,]
    bb_pos = chain_feats['all_atom_positions'][:,:, ca_idx] # [F,N,37,3]->[F,N,3] select the CA for each protein
    bb_center = np.sum(bb_pos, axis=1) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [F,3]
    # bb_center = np.mean(bb_center,axis=0)
    centered_pos = chain_feats['all_atom_positions'] - bb_center[:, None, None, :] # [F,N,37,3]
    scaled_pos = centered_pos / scale_factor
    chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]
    return chain_feats

class PdbDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
            is_testing=False,
            is_random_test=False
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_test = is_testing
        self._is_random_test=is_random_test
        self._data_conf = data_conf
        self._init_metadata()
        self._diffuser = diffuser
        self.offset =  {idx: 0 for idx in range(len(self.csv))}

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self):
        """Initialize metadata."""
        filter_conf = self.data_conf.filtering
        if self._is_training:
            pdb_csv = pd.read_csv(self.data_conf.csv_path) # 读取CS件
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
        elif self._is_test:
            pdb_csv = pd.read_csv(self.data_conf.test_csv_path) # 读取CSV文件
            #pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            print(pdb_csv)
            #pdb_csv = pdb_csv.head(6)
        else:
            pdb_csv = pd.read_csv(self.data_conf.val_csv_path)
            #pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            pdb_csv = pdb_csv[pdb_csv.seq_len <= 256]
            # pdb_csv = pdb_csv.sort_values('seq_len', ascending=False)
            #pdb_csv = pdb_csv.head(6)# 保留10个用于val
            # print(pdb_csv)
            # pdb_csv = pdb_csv.sort_values('seqlen', ascending=False)

        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv#[pdb_csv.split == 'train']
            self._log.info(f'Training: {len(self.csv)} examples')
        else:
            self.csv = pdb_csv#[pdb_csv.split == 'val']
            self._log.info(f'Validation: {len(self.csv)} examples')

    def select_random_samples(self,arr, t, k):
        n = arr.shape[0]  # Obtain the size of the first dimension, the number of samples
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")
        start_index = np.random.randint(0, n - (t)*k + 1)  # randomly select the start indexnp.random.randint(0, n - t*(k-1))
        # print('=========>>>>',start_index)
        end_index = start_index + (t)*k # the end index
        selected_samples = arr[start_index:end_index:k]  # select with step k
        return selected_samples,start_index


    def select_first_samples(self,arr, t, k):
        n = arr.shape[0]  # 获取第一个维度的大小，即样本数量
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")

        start_index = 0 #np.random.randint(0, n - (t)*k + 1)  # 随机选择起始索引
        end_index = start_index + (t)*k # 计算结束索引
        selected_samples = arr[start_index:end_index:k]  # 选择连续的t个样本，考虑步长k
        return selected_samples

    def select_last_samples(self,arr, t, k):
        n = arr.shape[0]  # 获取第一个维度的大小，即样本数量
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")
        # print('hello last')

        start_index =  n - (t)*k  #np.random.randint(0, n - (t)*k + 1)  # 随机选择起始索引
        end_index =  n # 计算结束索引
        selected_samples = arr[start_index:end_index:k]  # 选择连续的t个样本，考虑步长k
        return selected_samples

    def stage1_sampler(self,arr):
        n = arr.shape[0]  # Obtain the size of the first dimension, the number of samples
        index1 = np.random.randint(0, n-2)
        # 在 index1 之后随机选择一个 index2
        index2 = np.random.randint(index1 + 1, n)
        selected_samples = arr[[index1, index2]]  # select with step k
        return selected_samples


    def select_with_motion(self, arr, t, k):
        # 步骤1: 从数组中随机选择一个元素
        selected_index = np.random.randint(0, len(arr))
        first_element = arr[selected_index:selected_index+1]

        # 步骤2: 从剩下的数组中排除刚才选中的元素
        remaining_arr = np.delete(arr, selected_index,axis=0)

        # 步骤3: 从剩下的数组中随机选择连续的 t 个元素，采样间隔为 s
        max_start_index = len(remaining_arr) - (t - 1) * k - 1
        if max_start_index < 0:
            raise ValueError("The array is too small to select t elements with the given interval s.")

        start_index = np.random.randint(0, max_start_index + 1)
        t = t-1
        sampled_elements = remaining_arr[start_index:start_index + t * k:k]

        # 拼接 first_element 和 sampled_elements
        # print(first_element.shape, sampled_elements.shape,selected_index,start_index)
        # exit()
        result = np.concatenate((first_element, sampled_elements))

        return result

    def select_first_samples_with_motion(self,arr, t, k):
        # 步骤1: 从数组中随机选择一个元素
        selected_index = 0
        first_element = arr[selected_index:selected_index+1]

        # 步骤2: 从剩下的数组中排除刚才选中的元素
        remaining_arr = np.delete(arr, selected_index,axis=0)

        # 步骤3: 从剩下的数组中随机选择连续的 t 个元素，采样间隔为 s
        max_start_index = len(remaining_arr) - (t - 1) * k - 1
        if max_start_index < 0:
            raise ValueError("The array is too small to select t elements with the given interval s.")

        start_index = np.random.randint(0, max_start_index + 1)
        t = t-1
        sampled_elements = remaining_arr[start_index:start_index + t * k:k]

        # 拼接 first_element 和 sampled_elements
        # print(first_element.shape, sampled_elements.shape,selected_index,start_index)
        # exit()
        result = np.concatenate((first_element, sampled_elements))

        return result

    def farthest_point_sampling_with_dist_matrix(self,arr,distance_matrix, M):
        F = distance_matrix.shape[0]
        ####
        #selected_indices = np.random.permutation(F)[:M]
        #sampled_elements = arr[selected_indices]
        #return sampled_elements
        ####

        # 初始化：随机选择一个蛋白质作为第一个点
        #selected_indices = [np.random.randint(F)]
        selected_indices = [0]
        # 初始化每个蛋白质到当前选中子集的最小距离
        min_distances = distance_matrix[selected_indices[-1]].copy()

        for _ in range(M - 1):
            # 找到当前最远的蛋白质，即与现有子集最远的蛋白质
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(farthest_idx)

            # 更新每个蛋白质到已选蛋白质集合的最小距离
            min_distances = np.minimum(min_distances, distance_matrix[farthest_idx])

        selected_indices[0] = F // 2
        #print(selected_indices)
        sampled_elements = arr[selected_indices]
        #print(sampled_elements.shape)
        #exit()
        return sampled_elements

    def select_with_motion_continue(self, arr, t, k):

        # 步骤3: 从剩下的数组中随机选择连续的 t 个元素，采样间隔为 s
        max_start_index = len(arr) - t* k - 1
        if max_start_index < 0:
            raise ValueError("The array is too small to select t elements with the given interval s.")

        start_index = np.random.randint(0, max_start_index + 1)
        sampled_elements = arr[start_index:start_index + t * k:k]
        return sampled_elements

    # @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path,dist_matrix):
        processed_feats = dict(np.load(processed_file_path,allow_pickle=True))

        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        # frame_time_ref_motion = ref_frame + motion_frame + frame_time
        # frame_time_ref_motion = frame_time

        #atlas_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/atlas/atlas_unzip/'
        #pdb_name = processed_file_path.split('/')[-1].split('.')[0]
        #print(pdb_path)

        #data_path = processed_file_path.split('/')[:-2]
        #pdb_name = processed_file_path.split('/')[-1]
        #processed_file_path2 = os.path.join('/', *data_path, 'processed_npz_traj2_ce', pdb_name)
        #processed_file_path3 = os.path.join('/', *data_path, 'processed_npz_traj3_ce', pdb_name)

        processed_file_path2 = processed_file_path.replace('atlas_process', 'atlas_process2')
        processed_file_path3 = processed_file_path.replace('atlas_process', 'atlas_process3')

        #start_time = time.perf_counter()
        processed_feats2 = dict(np.load(processed_file_path2, allow_pickle=True))
        processed_feats3 = dict(np.load(processed_file_path3, allow_pickle=True))
        #start_time = time.perf_counter()
        #print(end_time-start_time, '======')
        processed_feats['all_atom_positions'] = np.concatenate((processed_feats['all_atom_positions'], processed_feats2['all_atom_positions'], processed_feats3['all_atom_positions']), axis=0)
        #end_time = time.perf_counter()
        #print(end_time-start_time, '======')
        #print(processed_feats['all_atom_mask'].shape, processed_feats['all_atom_positions'].shape) #(163, 37) (10001, 163, 37, 3)
        #print(processed_file_path)
        #exit()
        #traj1 = mdtraj.load(f'{atlas_dir}/{pdb_name}/{pdb_name}_prod_R1_fit.xtc', top=f'{atlas_dir}/{pdb_name}/{pdb_name}.pdb')
        #positions1 = traj1.xyz*10.0

        #traj2 = mdtraj.load(f'{atlas_dir}/{pdb_name}/{pdb_name}_prod_R2_fit.xtc', top=f'{atlas_dir}/{pdb_name}/{pdb_name}.pdb')
        #positions2 = traj2.xyz*10.0

        #traj3 = mdtraj.load(f'{atlas_dir}/{pdb_name}/{pdb_name}_prod_R3_fit.xtc', top=f'{atlas_dir}/{pdb_name}/{pdb_name}.pdb')
        #positions3 = traj3.xyz*10.0

        #print(processed_feats['all_atom_positions'].shape)
        #print(positions1)
        #print(positions1.shape)
        #exit()
        if self._is_training:
            #if self.data_conf.keep_first is not None:
            #    #print(self.data_conf.keep_first, 'is not None')
            #    first_data = processed_feats['all_atom_positions'][:self.data_conf.keep_first]
            #    dist_matrix= dist_matrix[:self.data_conf.keep_first,:self.data_conf.keep_first]
            #    tmp = self.farthest_point_sampling_with_dist_matrix(first_data,dist_matrix,ref_frame+frame_time)
            #else:
                #print('none')
            first_data = processed_feats['all_atom_positions']
            tmp = self.farthest_point_sampling_with_dist_matrix(first_data,dist_matrix,ref_frame+frame_time)
            start_index = 0
        else:
            first_data = processed_feats['all_atom_positions']
            tmp = self.farthest_point_sampling_with_dist_matrix(first_data,dist_matrix,ref_frame+frame_time)
            start_index = 0
            #print("evaluate dataloader")
            #exit()
            #if self._is_random_test:
            #    tmp,start_index = self.select_random_samples(processed_feats['all_atom_positions'],t=frame_time,k=self.data_conf.frame_sample_step)
            #else:
            #    tmp = self.select_first_samples(processed_feats['all_atom_positions'],t=frame_time,k=self.data_conf.frame_sample_step)
            #    start_index = -1

            #if self.data_conf.fix_sample_start is not None:
            #    first_data = processed_feats['all_atom_positions'][self.data_conf.fix_sample_start:]
            #    tmp = self.select_first_samples(first_data,t=frame_time,k=self.data_conf.frame_sample_step)
            #    start_index = -1
            # tmp[:ref_frame] = processed_feats['all_atom_positions'][:ref_frame]
            # print(tmp.shape)
        # tmp[-1] = tmp[0]
        # print(tmp.shape)
        # tmp = tmp[:1].repeat(np.shape(tmp)[0],axis=0)
        # print(tmp[:,0,0])
        # print(tmp.shape)
        # exit()
        processed_feats['all_atom_positions'] = tmp
        processed_feats = parse_dynamics_chain_feats_with_align(processed_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(processed_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time+ref_frame, -1),
            'all_atom_positions': torch.tensor(processed_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time+ref_frame, -1, -1)
        }
        # for k,v in chain_feats.items():
        #     print(k,v.shape)
        # exit()
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        # motion ref frame

        # motion_feats = {
        #     'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:motion_frame])[:,:, 0].to_tensor_7(),
        #     'motion_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
        #     'motion_atom37_pos': chain_feats['all_atom_positions'][:motion_frame] ,
        # }

        #print(processed_feats['all_atom_positions'].shape)
        #exit()
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][motion_frame:ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][motion_frame:ref_frame+motion_frame] ,
        }

        final_feats = {
            'aatype': chain_feats['aatype'][ref_frame+motion_frame:],
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][ref_frame+motion_frame:],
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][ref_frame+motion_frame:],
            'atom37_mask': chain_feats['all_atom_mask'][ref_frame+motion_frame:],
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][ref_frame+motion_frame:],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][ref_frame+motion_frame:],
        }

        final_feats.update(ref_feats)
        # final_feats.update(motion_feats)

        if not self._is_training:
            final_feats.update({'start_index':start_index})

        return final_feats

    def _create_diffused_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order['CA']]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min,
                high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(
                    f'Unable to generate diffusion mask for {row}')
        return diff_mask

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if 'name' in csv_row:
            pdb_name = csv_row['name']
        else:
            raise ValueError('Need chain identifier.')
        # matrix_path = os.path.join(
        #     '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/datasets/atlas/ca_rmsd_matrix6_compress',
        #     f"{pdb_name}.npz.npy"
        # )
        matrix_path = os.path.join('data/dist_matrix', f"{pdb_name}.npz.npy")

        #start_time = time.perf_counter()
        dist_matrix = np.load(matrix_path)
        #end_time = time.perf_counter()
        #print(end_time-start_time, '===')
        #print(matrix_path, dist_matrix.shape)
        #exit()
        processed_file_path = csv_row['atlas_npz']
        chain_feats = self._process_csv_row(processed_file_path, dist_matrix)
        # info = ''
        # for k in chain_feats.keys():
        #     info+=f' {k}:{chain_feats[k].shape}:{type(chain_feats[k])}||'
        # print(info)
        # exit()
        # with embedding features.
        # Note

        frame_time = chain_feats['aatype'].shape[0]
        node_edge_feature_path = csv_row['embed_path']  # here
        #attr_dict = dict(np.load(node_edge_feature_path.replace("zsy_43187", "public")))
        #print('aaaa', attr_dict['node_repr'].shape,attr_dict['edge_repr'].shape)
        # chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr']).unsqueeze(0).expand(frame_time, -1,-1)})
        # chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr']).unsqueeze(0).expand(frame_time, -1,-1,-1)})
        #chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr'])})
        #chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr'])})
        # node_edge_feature_path = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/liuce/code/openfold-main/alphafold_features/predictions/' + pdb_name + '.npz'
        #print(node_edge_feature_path)
        #exit()
        attr_dict = dict(np.load(node_edge_feature_path))
        #print('bbbb', attr_dict['single'].shape, attr_dict['pair'].shape)
        #exit()
        chain_feats.update({'node_repr':torch.tensor(attr_dict['single'])})
        chain_feats.update({'edge_repr':torch.tensor(attr_dict['pair'])})

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:,:, 0] # 为什么要取0
        # print(gt_bb_rigid.shape,chain_feats['rigidgroups_0'].shape)
        # exit()
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())
        # print(gt_bb_rigid.shape)#torch.Size([6, N_res])
        # exit()
        # Sample t and diffuse.
        if self.is_training:
            # TODO here add different t
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            #print(t)
            #exit()
            ref_diff_feats = self._diffuser.forward_marginal(
                rigids_0=rigid_utils.Rigid.from_tensor_7(chain_feats['ref_rigids_0']),
                t=t,
                diffuse_mask=None
            )
            chain_feats['ref_rot_score'] = ref_diff_feats['rot_score']
        else:
            t = 1.0
            # todo here
            if self._data_conf.dynamics:
                diff_feats_t = self.diffuser.sample_ref(
                    n_samples=gt_bb_rigid.shape[0]*gt_bb_rigid.shape[1],
                    diffuse_mask=None,
                    as_tensor_7=True,
                )
            else:
                diff_feats_t = self.diffuser.sample_ref(
                    n_samples=gt_bb_rigid.shape[0],
                    impute=gt_bb_rigid,
                    diffuse_mask=None,
                    as_tensor_7=True,
                )

        chain_feats.update(diff_feats_t)

        chain_feats['t'] = t
        if not self.is_training:
            start_index = chain_feats.pop('start_index')
            # print(start_index)
        # Convert all features to tensors.
        final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)

        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name,start_index



class PdbEvalDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            traj_npz_path,
            omega_embed_path,
            sample_numbers,
            pdb_name,
            is_training,
            is_testing=False,
            is_random_test=False,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_test = is_testing
        self._is_random_test=is_random_test
        self._data_conf = data_conf
        self._diffuser = diffuser
        # load traj npz
        self.traj_npz_path = traj_npz_path
        self.omega_embed_path = omega_embed_path
        self.processed_feats = dict(np.load(self.traj_npz_path,allow_pickle=True))
        self.sample_numbers = sample_numbers
        self.start_sample_index = self.data_conf.fix_sample_start
        # self.start_sample_index+10000 #
        upper = len(self.processed_feats['all_atom_positions']) - (self.data_conf.motion_number+self.data_conf.ref_number+self.data_conf.frame_time)*self.data_conf.frame_sample_step
        # if upper > self.start_sample_index+2000:
        #     upper=self.start_sample_index+200
        self.sample_index = random.sample(range(self.start_sample_index,upper),sample_numbers)
        print(f'====================>>>>>>>>>>>>>> Sample From Time Step {self.start_sample_index} to {upper}')
        self.pdb_name =pdb_name

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def select_first_samples(self,arr,start_idx, t, k):
        n = arr.shape[0]  # 获取第一个维度的大小，即样本数量
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")
        start_index = start_idx  - self.start_sample_index  #np.random.randint(0, n - (t)*k + 1) #np.random.randint(0, n - (t)*k + 1)  # 随机选择起始索引
        end_index = start_index + (t)*k # 计算结束索引
        print('===================>>>>>>>>>>>>>>>>>>',start_idx)
        selected_samples = arr[start_index:end_index:k]  # 选择连续的t个样本，考虑步长k
        return selected_samples


    # @fn.lru_cache(maxsize=100)
    def _process_one_sample(self,idx):
        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        frame_time_ref_motion = ref_frame + motion_frame + frame_time

        if self._is_training:
            raise RuntimeError("Error: Training Mode, PdbEvalDataset is for evalation")

        else:
            first_data = self.processed_feats['all_atom_positions'][self.data_conf.fix_sample_start:]
            tmp = self.select_first_samples(first_data,t=frame_time_ref_motion,start_idx = idx,k=self.data_conf.frame_sample_step)


        sample_feats = {k: v for k, v in self.processed_feats.items() if k != 'all_atom_positions'}

        sample_feats['all_atom_positions'] = tmp

        sample_feats = parse_dynamics_chain_feats(sample_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(sample_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(sample_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(sample_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }
        # for k,v in chain_feats.items():
        #     print(k,v.shape)
        # exit()
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        motion_feats = {
            'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:motion_frame])[:,:, 0].to_tensor_7(),
            'motion_node_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
            'motion_atom37_pos': chain_feats['all_atom_positions'][:motion_frame] ,
        }
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][motion_frame:ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][motion_frame:ref_frame+motion_frame] ,
        }

        final_feats = {
            'aatype': chain_feats['aatype'][ref_frame+motion_frame:],
            'seq_idx':  torch.tensor(sample_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][ref_frame+motion_frame:],
            'residue_index': torch.tensor(sample_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][ref_frame+motion_frame:],
            'atom37_mask': chain_feats['all_atom_mask'][ref_frame+motion_frame:],
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][ref_frame+motion_frame:],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][ref_frame+motion_frame:],
        }

        final_feats.update(ref_feats)
        final_feats.update(motion_feats)

        return final_feats

    def __len__(self):
        return self.sample_numbers

    def __getitem__(self, idx):
        sample_idx = self.sample_index[idx]
        # Sample data example.
        chain_feats = self._process_one_sample(sample_idx)

        frame_time = chain_feats['aatype'].shape[0]
        attr_dict = dict(np.load(self.omega_embed_path))

        chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr'])})
        chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr'])})
        # Use a fixed seed for evaluation.
        rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:,:, 0] # sample 0 for backbone
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())
        if self.is_training:
            raise RuntimeError("Error: Training Mode, PdbEvalDataset is for evalation")
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0]*gt_bb_rigid.shape[1],
                diffuse_mask=None,
                as_tensor_7=True,
            )

        chain_feats.update(diff_feats_t)

        chain_feats['t'] = t
        # Convert all features to tensors.
        final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)

        if self.is_training:
            raise RuntimeError("Error: Training Mode, PdbEvalDataset is for evalation")
        else:
            return final_feats, self.pdb_name



class PdbPersonalDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
            is_testing=False,
            is_random_test=False,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_test = is_testing
        self._is_random_test=is_random_test
        self._data_conf = data_conf
        self._diffuser = diffuser
        # load traj npz
        if self._is_training:
            pdb_csv = pd.read_csv(self.data_conf.csv_path) # 读取CSV文件
            print(f'Loading Train Data From {self.data_conf.csv_path}')
        elif self._is_test:
            pdb_csv = pd.read_csv(self.data_conf.test_csv_path)
        else:
            pdb_csv = pd.read_csv(self.data_conf.val_csv_path)
            print(f'Loading Val Data  From {self.data_conf.csv_path}')
        csv_row = pdb_csv.iloc[0]
        processed_file_path = csv_row['atlas_npz']
        self.omega_embed_path = csv_row['embed_path']

        self.processed_feats = dict(np.load(processed_file_path,allow_pickle=True))
        upper = len(self.processed_feats['all_atom_positions']) - (self.data_conf.motion_number+self.data_conf.ref_number+self.data_conf.frame_time)*self.data_conf.frame_sample_step
        self.data_len = upper
        self.pdb_name = csv_row['name']

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def select_with_motion_continue(self, arr, t, k):

        # 步骤3: 从剩下的数组中随机选择连续的 t 个元素，采样间隔为 s
        max_start_index = len(arr) - t* k - 1
        if max_start_index < 0:
            raise ValueError("The array is too small to select t elements with the given interval s.")

        start_index = np.random.randint(0, max_start_index + 1)
        sampled_elements = arr[start_index:start_index + t * k:k]
        return sampled_elements

    def select_first_samples(self,arr, t, k):
        n = arr.shape[0]  # 获取第一个维度的大小，即样本数量
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")

        start_index = 0 #np.random.randint(0, n - (t)*k + 1)  # 随机选择起始索引
        end_index = start_index + (t)*k # 计算结束索引
        selected_samples = arr[start_index:end_index:k]  # 选择连续的t个样本，考虑步长k
        return selected_samples


    # @fn.lru_cache(maxsize=100)
    def _process_one_sample(self,idx):
        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        frame_time_ref_motion = ref_frame + motion_frame + frame_time

        if self._is_training:
            first_data = self.processed_feats['all_atom_positions'][:self.data_conf.fix_sample_start]
            # print('=====================Train:',len(first_data))
            tmp = self.select_with_motion_continue(first_data,t=frame_time_ref_motion,k=self.data_conf.frame_sample_step)

        else:
            first_data = self.processed_feats['all_atom_positions'][self.data_conf.fix_sample_start:]
            # print('=====================Test:',len(first_data))
            tmp = self.select_first_samples(first_data,t=frame_time_ref_motion,k=self.data_conf.frame_sample_step)


        sample_feats = {k: v for k, v in self.processed_feats.items() if k != 'all_atom_positions'}

        sample_feats['all_atom_positions'] = tmp

        sample_feats = parse_dynamics_chain_feats(sample_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(sample_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(sample_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(sample_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }
        # for k,v in chain_feats.items():
        #     print(k,v.shape)
        # exit()
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        motion_feats = {
            'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:motion_frame])[:,:, 0].to_tensor_7(),
            'motion_node_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
            'motion_atom37_pos': chain_feats['all_atom_positions'][:motion_frame] ,
        }
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][motion_frame:ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][motion_frame:ref_frame+motion_frame] ,
        }

        final_feats = {
            'aatype': chain_feats['aatype'][ref_frame+motion_frame:],
            'seq_idx':  torch.tensor(sample_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][ref_frame+motion_frame:],
            'residue_index': torch.tensor(sample_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][ref_frame+motion_frame:],
            'atom37_mask': chain_feats['all_atom_mask'][ref_frame+motion_frame:],
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][ref_frame+motion_frame:],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][ref_frame+motion_frame:],
        }

        final_feats.update(ref_feats)
        final_feats.update(motion_feats)

        return final_feats

    def __len__(self):
        if self._is_training:
            return self.data_len
        else:
            return 1 # just for validation

    def __getitem__(self, idx):
        sample_idx = idx
        # Sample data example.
        chain_feats = self._process_one_sample(sample_idx)

        frame_time = chain_feats['aatype'].shape[0]
        attr_dict = dict(np.load(self.omega_embed_path))

        chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr'])})
        chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr'])})
        # Use a fixed seed for evaluation.
        rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:,:, 0] # sample 0 for backbone
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())
        if self.is_training:
            # TODO here add different t
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            ref_diff_feats = self._diffuser.forward_marginal(
                rigids_0=rigid_utils.Rigid.from_tensor_7(chain_feats['ref_rigids_0']),
                t=t,
                diffuse_mask=None
            )
            chain_feats['ref_rot_score'] = ref_diff_feats['rot_score']
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0]*gt_bb_rigid.shape[1],
                diffuse_mask=None,
                as_tensor_7=True,
            )

        chain_feats.update(diff_feats_t)

        chain_feats['t'] = t
        # Convert all features to tensors.
        final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)

        if self.is_training:
            return final_feats
        else:
            return final_feats, self.pdb_name



class FFPersonalDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
            is_testing=False,
            is_random_test=False,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_test = is_testing
        self._is_random_test=is_random_test
        self._data_conf = data_conf
        self._diffuser = diffuser
        self._split_percent = data_conf.split_percent
        print(self._split_percent)

        # load traj npz
        if self._is_training:
            pdb_csv = pd.read_csv(self.data_conf.csv_path) # 读取CSV文件
            print(f'Loading Train Data From {self.data_conf.csv_path}')
        elif self._is_test:
            pdb_csv = pd.read_csv(self.data_conf.test_csv_path)
        else:
            pdb_csv = pd.read_csv(self.data_conf.val_csv_path)
            print(f'Loading Val Data  From {self.data_conf.csv_path}')

        csv_row = pdb_csv.iloc[0]
        processed_file_path = csv_row['atlas_npz']
        self.omega_embed_path = csv_row['embed_path']

        self.processed_feats = dict(np.load(processed_file_path,allow_pickle=True))
        ori_len = len(self.processed_feats['all_atom_positions'])

        self.start_border = int(ori_len*(self._split_percent-0.1))
        self.train_data_border = int(ori_len*self._split_percent)
        upper = self.train_data_border - (self.data_conf.motion_number+self.data_conf.ref_number+self.data_conf.frame_time)*self.data_conf.frame_sample_step
        self.data_len = upper
        self.pdb_name = csv_row['name']
        print(f'{self.pdb_name} with lenght {ori_len} DATA border:{self.train_data_border}')

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def select_with_motion_continue(self, arr, t, k):

        # 步骤3: 从剩下的数组中随机选择连续的 t 个元素，采样间隔为 s
        max_start_index = len(arr) - t* k - 1
        if max_start_index < 0:
            raise ValueError("The array is too small to select t elements with the given interval s.")

        start_index = np.random.randint(0, max_start_index + 1)
        sampled_elements = arr[start_index:start_index + t * k:k]
        return sampled_elements

    def select_first_samples(self,arr, t, k):
        n = arr.shape[0]  # 获取第一个维度的大小，即样本数量
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")

        start_index = 0 #np.random.randint(0, n - (t)*k + 1)  # 随机选择起始索引
        end_index = start_index + (t)*k # 计算结束索引
        selected_samples = arr[start_index:end_index:k]  # 选择连续的t个样本，考虑步长k
        return selected_samples


    # @fn.lru_cache(maxsize=100)
    def _process_one_sample(self,idx):
        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        frame_time_ref_motion = ref_frame + motion_frame + frame_time
        if self._is_training:
            first_data = self.processed_feats['all_atom_positions'][self.start_border:self.train_data_border]
            # print('=====================Train:',len(self.processed_feats['all_atom_positions']),len(first_data),self.train_data_border)
            tmp = self.select_with_motion_continue(first_data,t=frame_time_ref_motion,k=self.data_conf.frame_sample_step)

        else:
            first_data = self.processed_feats['all_atom_positions'][self.train_data_border:]
            # print('=====================Test:',len(self.processed_feats['all_atom_positions']),len(first_data),self.test_data_border)
            tmp = self.select_first_samples(first_data,t=frame_time_ref_motion,k=self.data_conf.frame_sample_step)


        sample_feats = {k: v for k, v in self.processed_feats.items() if k != 'all_atom_positions'}

        sample_feats['all_atom_positions'] = tmp

        sample_feats = parse_dynamics_chain_feats(sample_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(sample_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(sample_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(sample_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }
        # for k,v in chain_feats.items():
        #     print(k,v.shape)
        # exit()
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        motion_feats = {
            'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:motion_frame])[:,:, 0].to_tensor_7(),
            'motion_node_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
            'motion_atom37_pos': chain_feats['all_atom_positions'][:motion_frame] ,
        }
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][motion_frame:ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][motion_frame:ref_frame+motion_frame] ,
        }

        final_feats = {
            'aatype': chain_feats['aatype'][ref_frame+motion_frame:],
            'seq_idx':  torch.tensor(sample_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][ref_frame+motion_frame:],
            'residue_index': torch.tensor(sample_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(sample_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][ref_frame+motion_frame:],
            'atom37_mask': chain_feats['all_atom_mask'][ref_frame+motion_frame:],
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][ref_frame+motion_frame:],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][ref_frame+motion_frame:],
        }

        final_feats.update(ref_feats)
        final_feats.update(motion_feats)

        return final_feats

    def __len__(self):
        if self._is_training:
            return self.data_len
        else:
            return 1 # just for validation

    def __getitem__(self, idx):
        sample_idx = idx
        # Sample data example.
        chain_feats = self._process_one_sample(sample_idx)

        frame_time = chain_feats['aatype'].shape[0]
        attr_dict = dict(np.load(self.omega_embed_path))

        chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr'])})
        chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr'])})
        # Use a fixed seed for evaluation.
        rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:,:, 0] # sample 0 for backbone
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())
        if self.is_training:
            # TODO here add different t
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            ref_diff_feats = self._diffuser.forward_marginal(
                rigids_0=rigid_utils.Rigid.from_tensor_7(chain_feats['ref_rigids_0']),
                t=t,
                diffuse_mask=None
            )
            chain_feats['ref_rot_score'] = ref_diff_feats['rot_score']
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0]*gt_bb_rigid.shape[1],
                diffuse_mask=None,
                as_tensor_7=True,
            )

        chain_feats.update(diff_feats_t)

        chain_feats['t'] = t
        # Convert all features to tensors.
        final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)

        if self.is_training:
            return final_feats
        else:
            return final_feats, self.pdb_name


class FFDatasetExtrapolation(PdbDataset):
    def _process_csv_row(self, processed_file_path):
        processed_feats = dict(np.load(processed_file_path,allow_pickle=True))
        all_data_numbers = len(processed_feats['all_atom_positions'])
        split_idx = int(all_data_numbers*self._data_conf.split_percent)
        motion_frame = self.data_conf.motion_number
        ref_frame = self.data_conf.ref_number
        frame_time = self.data_conf.frame_time
        # here to sample frame_time continuous positions.
        frame_time_ref_motion = ref_frame + motion_frame + frame_time

        if self._is_training:
            first_data = processed_feats['all_atom_positions'][:split_idx]
            print(split_idx)
            tmp = self.select_with_motion_continue(first_data,
                                            frame_time_ref_motion,
                                            self.data_conf.frame_sample_step)
        else:
            first_data = processed_feats['all_atom_positions'][split_idx:]
            print(split_idx)
            tmp = self.select_first_samples(first_data,t=frame_time_ref_motion,k=self.data_conf.frame_sample_step)
            # tmp[:ref_frame] = processed_feats['all_atom_positions'][:ref_frame]
            # print(tmp.shape)
        # tmp[-1] = tmp[0]
        # print(tmp.shape)
        # tmp = tmp[:1].repeat(np.shape(tmp)[0],axis=0)
        # print(tmp[:,0,0])
        # print(tmp.shape)
        # exit()
        processed_feats['all_atom_positions'] = tmp
        processed_feats = parse_dynamics_chain_feats(processed_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(processed_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time_ref_motion, -1),
            'all_atom_positions': torch.tensor(processed_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time_ref_motion, -1, -1)
        }
        # for k,v in chain_feats.items():
        #     print(k,v.shape)
        # exit()
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        # motion ref frame

        motion_feats = {
            'motion_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:motion_frame])[:,:, 0].to_tensor_7(),
            'motion_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(motion_frame, -1),
            'motion_atom37_pos': chain_feats['all_atom_positions'][:motion_frame] ,
        }
        ref_feats = {
            'ref_rigids_0': rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][motion_frame:ref_frame+motion_frame])[:,:, 0].to_tensor_7(),
            'ref_node_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(ref_frame, -1),
            'ref_atom37_pos': chain_feats['all_atom_positions'][motion_frame:ref_frame+motion_frame] ,
        }



        final_feats = {
            'aatype': chain_feats['aatype'][ref_frame+motion_frame:],
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'][ref_frame+motion_frame:],
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(frame_time, -1),
            'res_mask': torch.tensor(processed_feats['bb_mask']).unsqueeze(0).expand(frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'][ref_frame+motion_frame:],
            'atom37_mask': chain_feats['all_atom_mask'][ref_frame+motion_frame:],
            # 'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'][ref_frame+motion_frame:],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'][ref_frame+motion_frame:],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'][ref_frame+motion_frame:],
        }

        final_feats.update(ref_feats)
        final_feats.update(motion_feats)

        return final_feats


class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
        ):
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices) * self._batch_size

        if self._sample_mode in ['cluster_length_batch', 'cluster_time_batch']:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._log.info(f'Read {self._max_cluster} clusters.')
            self._missing_pdbs = 0
            def cluster_lookup(pdb):
                pdb = pdb.upper()
                pdb = pdb.split('.')[0]
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]
            self._data_csv['cluster'] = self._data_csv['name'].map(cluster_lookup)
            num_clusters = len(set(self._data_csv['cluster']))
            self.sampler_len = num_clusters * self._batch_size
            self._log.info(
                f'Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}'
            )

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._data_conf.cluster_path, "r") as f:
            for i,line in enumerate(f):
                for chain in line.split(' '):
                    pdb = chain.split('_')[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def __iter__(self):
        if self._sample_mode == 'length_batch':
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'time_batch':
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'cluster_time_batch':
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            dataset_indices = sampled_clusters['index'].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len


# modified from torch.utils.data.distributed.DistributedSampler
# key points: shuffle of each __iter__ is determined by epoch num to ensure the same shuffle result for each proccessor
class DistributedTrainSampler(data.Sampler):
    def __init__(self,
                *,data_conf,dataset,batch_size,
                num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"" [0, {}]".format(rank, num_replicas - 1))
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._data_csv)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil((self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) :
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._batch_size)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate((indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        #
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


if __name__=='__main__':
    from omegaconf import DictConfig
    import hydra
    import os

    @hydra.main(version_base=None, config_path="../config", config_name="debug_dynamics_extrapolation")
    def run(conf: DictConfig) -> None:

        _diff_conf = conf.diffuser
        _model_conf = conf.model
        _data_conf = conf.data
        _exp_conf = conf.experiment
        _diffuser = se3_diffuser.SE3Diffuser(_diff_conf)
        _use_ddp = _exp_conf.use_ddp
        if _use_ddp :
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl')
            ddp_info = eu.get_ddp_info()
            if ddp_info['rank'] not in [0,-1]:
                _use_wandb = False
                _exp_conf.ckpt_dir = None


        train_dataset = PdbDataset(
                data_conf=_data_conf,
                diffuser=_diffuser,
        is_training=False
            )
        a = train_dataset.__getitem__(0)
        print(a.keys())
        for k in a.keys():
            print(f'key:{k} value:{a[k].shape}')
        exit()

        test_dataset = PdbDatasetExtrapolation(
        data_conf=_data_conf,
        diffuser=_diffuser,
        is_training=False
            )
        a = test_dataset.__getitem__(0)
        print(a[0].keys())
        for k in a[0].keys():
            print(f'key:{k} value:{a[0][k].shape}')
        exit()

        if not _use_ddp:
            train_sampler = TrainSampler(
                data_conf=_data_conf,
                dataset=train_dataset,
                batch_size=_exp_conf.batch_size,
                sample_mode=_exp_conf.sample_mode,
            )
        else:
            # train_sampler = DistributedTrainSampler(
            #     data_conf=_data_conf,
            #     dataset=train_dataset,
            #     batch_size=_exp_conf.batch_size,
            # )
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        print(len(train_sampler))
        num_workers = _exp_conf.num_loader_workers
        persistent_workers = True if num_workers > 0 else False
        prefetch_factor=2
        prefetch_factor = 2 if num_workers == 0 else prefetch_factor

        train_loader  = data.DataLoader(
                train_dataset,
                batch_size=_exp_conf.batch_size if not _exp_conf.use_ddp else _exp_conf.batch_size // ddp_info['world_size'],
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                sampler=train_sampler,
                # Need fork https://github.com/facebookresearch/hydra/issues/964
                multiprocessing_context='fork' if num_workers != 0 else None,
        )


        # train_loader  = du.create_data_loader(
        #     train_dataset,
        #     np_collate=False,
        #     batch_size=_exp_conf.batch_size if not _exp_conf.use_ddp else _exp_conf.batch_size // ddp_info['world_size'],
        #     shuffle=False,
        #     num_workers=num_workers,
        #     drop_last=False,
        #     max_squared_res=_exp_conf.max_squared_res,
        # )
        print(len(train_loader))
        # print((train_loader._index_sampler()))
        # exit()
        for epoch in range(20):
            print('='*50,epoch)
            for train_feats in train_loader:
                pass

            # for k in train_feats:
            #     print(k,train_feats[k].shape)
            # exit()
        exit()
    run()
