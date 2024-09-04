"""PDB dataset loader."""
import sys
import math
from typing import Optional
from src.experiments import utils as eu
import torch
import torch.distributed as dist

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
from src.data import pdb_data_loader
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
import mdtraj as md

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
    #  aatype:(255, 21):<class 'numpy.ndarray'> || 
    #  between_segment_residues:(255,):<class 'numpy.ndarray'> || 
    # domain_name:(1,):<class 'numpy.ndarray'> || 
    # residue_index:(255,):<class 'numpy.ndarray'> ||
    # seq_length:(255,):<class 'numpy.ndarray'> ||
    # sequence:(1,):<class 'numpy.ndarray'> || 
    # all_atom_positions:(4, 255, 37, 3):<class 'numpy.ndarray'> || 
    # all_atom_mask:(255, 37):<class'numpy.ndarray'> || 
    # resolution:(1,):<class 'numpy.ndarray'> || 
    # is_distillation:():<class 'numpy.ndarray'> ||
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
    bb_pos = chain_feats['all_atom_positions'][0,:, ca_idx] # [F,N,37,3]->[N,3] select first protein as anchor
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [3]
    # print(np.sum(chain_feats['all_atom_positions'][:,:, ca_idx], axis=1) / (np.sum(chain_feats['bb_mask']) + 1e-5))
    # exit()
    # bb_pos = chain_feats['all_atom_positions'][:,:, ca_idx]
    # bb_center = np.sum(bb_pos, axis=1) / (np.sum(chain_feats['bb_mask']) + 1e-5)

    # print("=====>>>",chain_feats['bb_mask'].shape,
    #       'bb_center',bb_center.shape,
    #       'bb_pos',bb_pos.shape,'all_atom_positions',
    #       chain_feats['all_atom_positions'].shape)
    # exit()

    centered_pos = chain_feats['all_atom_positions'] - bb_center[None, None, None, :] # [F,N,37,3]
    scaled_pos = centered_pos / scale_factor
    chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
    chain_feats['bb_positions'] = chain_feats['all_atom_positions'][:,:,ca_idx]# [F,N,3]
    # print("=====>>>",chain_feats['bb_mask'].shape,
    #       'bb_center',bb_center.shape,
    #       'bb_pos',bb_pos.shape,'all_atom_positions','centered_pos',centered_pos.shape,
    #       chain_feats['all_atom_positions'].shape,'bb_positions',
    #       chain_feats['bb_positions'].shape)
    # exit()
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
            pdb_csv = pd.read_csv(self.data_conf.csv_path) # 读取CSV文件
            #print(len(pdb_csv))
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            # pdb_csv = pdb_csv.head(10)
            # pdb_csv = pdb_csv.sort_values('seq_len', ascending=False) # For Single/All mmcif new
        elif self._is_test:
            pdb_csv = pd.read_csv(self.data_conf.test_csv_path) # 读取CSV文件
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            #pdb_csv = pdb_csv.head(10)
        else:
            pdb_csv = pd.read_csv(self.data_conf.val_csv_path)
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            # pdb_csv = pdb_csv.sort_values('seq_len', ascending=False)
            #pdb_csv = pdb_csv.head(10)# 保留10个用于val
            # print(pdb_csv)
            # pdb_csv = pdb_csv.sort_values('seqlen', ascending=False)
        #pdb_csv = pdb_csv.head(10) 
        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv#[pdb_csv.split == 'train']
            self._log.info(f'Training: {len(self.csv)} examples')
        else:
            self.csv = pdb_csv#[pdb_csv.split == 'val']
            self._log.info(f'Validation: {len(self.csv)} examples')

    def select_random_samples(self, arr, arr2, arr3, t, k):
        n = arr.shape[0]  # Obtain the size of the first dimension, the number of samples
        if t > n:
            raise ValueError("t cannot be greater than the number of samples")
        start_index = np.random.randint(0, n - (t)*k + 1)  # randomly select the start indexnp.random.randint(0, n - t*(k-1))
        # print('=========>>>>',start_index)
        end_index = start_index + (t)*k # the end index
        #print(start_index, n - (t)*k + 1)

        selected_samples = arr[start_index:end_index:k]  # select with step k
        selected_samples2 = arr2[start_index:end_index:k]  # select with step k
        selected_samples3 = arr3[start_index:end_index:k]  # select with step k
        return selected_samples, selected_samples2, selected_samples3

    def select_first_samples(self, arr, arr2, arr3, t, k):                                                                                                                                                     
        n = arr.shape[0]  # 获取第一个维度的大小，即样本数量                                                                                                                                      
        if t > n:                                                                                                                                                                                 
            raise ValueError("t cannot be greater than the number of samples")                                                                                                                    
                                                                                                                                                                                                  
        start_index = 0 #np.random.randint(0, n - (t)*k + 1)  # 随机选择起始索引                                                                                                                  
        end_index = start_index + (t)*k # 计算结束索引                                                                                                                                            
        selected_samples = arr[start_index:end_index:k]  # 选择连续的t个样本，考虑步长k  
        selected_samples2 = arr2[start_index:end_index:k]  # select with step k
        selected_samples3 = arr3[start_index:end_index:k]  # select with step k                                                                                                         
        return selected_samples, selected_samples2, selected_samples3

    # @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path, force_file_path, vel_file_path, pdb_file_path):
        # here to sample frame_time continuous positions.
        processed_feats = dict(np.load(processed_file_path,allow_pickle=True,mmap_mode='r'))
        with open(force_file_path.replace('.pkl', '_Ca.pkl'), 'rb') as f:
           force_feats = pickle.load(f)
        with open(vel_file_path.replace('.pkl', '_ca.pkl'), 'rb') as f:
           vel_feats = pickle.load(f)

        #traj = md.load(pdb_file_path)                                                                                                                                                                    
        #ca_indices = traj.topology.select('name CA')                                                                                                                                                
        #print(ca_indices)                                                                                                                                                                                           
        #ca_indices = np.array(ca_indices)[None, :, None]  
        #print(ca_indices)                                                                                                                                          
        #ca_indices = np.repeat(ca_indices, force_feats.shape[0], axis=0)                                                                                                                       
        #ca_indices = np.repeat(ca_indices, force_feats.shape[2], axis=2)                                                                                                                       
        #ca_indices = torch.from_numpy(ca_indices).expand(force_feats.shape[0], -1, force_feats.shape[2])                                                                                                                                                                                           
        #force_feats = torch.gather(input=torch.from_numpy(force_feats), dim=1, index=ca_indices).numpy()                                                                
        #vel_feats = torch.gather(input=torch.from_numpy(vel_feats), dim=1, index=ca_indices).numpy() 

        #vel_feats = dict(np.load(vel_file_path,allow_pickle=True))
        # here to sample frame_time continuous positions.
        frame_time = self.data_conf.frame_time

        if self._is_training:
            tmp, tmp2, tmp3 = self.select_random_samples(processed_feats['all_atom_positions'][:self.data_conf.keep_first],force_feats[:self.data_conf.keep_first],vel_feats[:self.data_conf.keep_first],t=frame_time,k=self.data_conf.frame_sample_step)
        else:
            tmp, tmp2, tmp3 = self.select_first_samples(processed_feats['all_atom_positions'][self.data_conf.fix_sample_start:],force_feats[self.data_conf.fix_sample_start:],vel_feats[self.data_conf.fix_sample_start:],t=frame_time,k=self.data_conf.frame_sample_step)
        # # centrailize the backbone
        # processed_feats = parse_dynamics_chain_feats_with_ref(processed_feats,first_frame=first_ref_feats) # centralized by the first tine

        new_feats = {k:v for k,v in processed_feats.items() if k != 'all_atom_positions'}
        new_feats['all_atom_positions'] = tmp
        new_feats = parse_dynamics_chain_feats_no_norm(new_feats)
        #processed_feats['all_atom_positions'] = tmp
        # processed_feats = parse_dynamics_chain_feats(processed_feats) # centralized by the first tine

        # processed_feats = parse_dynamics_chain_feats_split(processed_feats)
        chain_feats = {
            'aatype': torch.tensor(np.argmax(processed_feats['aatype'],axis=-1)).long().unsqueeze(0).expand(frame_time, -1),
            'all_atom_positions': torch.tensor(new_feats['all_atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['all_atom_mask']).double().unsqueeze(0).expand(frame_time, -1, -1),
            'force': torch.tensor(tmp2).double(),
            'vel':  torch.tensor(tmp3).double(),
        }
        
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        final_feats = {
            'aatype': chain_feats['aatype'],
            'seq_idx':  torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(self.data_conf.frame_time, -1),
            # 'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
            'residue_index': torch.tensor(processed_feats['residue_index']).unsqueeze(0).expand(self.data_conf.frame_time, -1),
            'res_mask': torch.tensor(new_feats['bb_mask']).unsqueeze(0).expand(self.data_conf.frame_time, -1),
            'atom37_pos': chain_feats['all_atom_positions'],
            'atom37_mask': chain_feats['all_atom_mask'],
            'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
            'alt_torsion_angles_sin_cos':chain_feats['alt_torsion_angles_sin_cos'],
            'torsion_angles_mask':chain_feats['torsion_angles_mask'],
            'force': chain_feats['force'],
            'vel': chain_feats['vel']
        }
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
        processed_file_path = csv_row['atlas_npz']
        force_file_path = csv_row['force_path']
        #print(force_file_path)
        vel_file_path = csv_row['vel_path']
        pdb_file_path = csv_row['pdb_path']

        chain_feats = self._process_csv_row(processed_file_path, force_file_path, vel_file_path, pdb_file_path)
        #chain_feats = parse_dynamics_chain_feats_no_norm(chain_feats) 
        #chain_feats = self._process_csv_row(processed_file_path, force_file_path, vel_, pdb_file_path)
        frame_time = chain_feats['aatype'].shape[0]
        node_edge_feature_path = csv_row['embed_path']  # here
        attr_dict = dict(np.load(node_edge_feature_path))
        # chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr']).unsqueeze(0).expand(frame_time, -1,-1)})
        # chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr']).unsqueeze(0).expand(frame_time, -1,-1,-1)})
        chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr'])})
        chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr'])})
        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:,:, 0] # 为什么要取0
        # exit()
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())
        # Sample t and diffuse.
        if self.is_training:
            # TODO here add different t
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
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
        # Convert all features to tensors.
        final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name



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
        prefetch_factor = 2
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
