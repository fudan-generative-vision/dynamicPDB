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
from torch.utils import data
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils
import pickle

def parse_dynamics_chain_feats_no_norm(chain_feats, scale_factor=1.):
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
    chain_feats['all_atom_positions'] = chain_feats['all_atom_positions'] * (chain_feats['all_atom_mask'][..., None][np.newaxis, ...])
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
            pdb_csv = pd.read_csv(self.data_conf.csv_path) 
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
        elif self._is_test:
            pdb_csv = pd.read_csv(self.data_conf.test_csv_path)
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            pdb_csv = pdb_csv.head(3) # select three for validation
            
        else:
            pdb_csv = pd.read_csv(self.data_conf.val_csv_path)
            pdb_csv = pdb_csv[pdb_csv.seq_len <= filter_conf.max_len]
            pdb_csv = pdb_csv.head(3) # select three for validation
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
        end_index = start_index + (t)*k # the end index

        selected_samples = arr[start_index:end_index:k]  # select with step k
        selected_samples2 = arr2[start_index:end_index:k]  # select with step k
        selected_samples3 = arr3[start_index:end_index:k]  # select with step k
        return selected_samples, selected_samples2, selected_samples3

    def select_first_samples(self, arr, arr2, arr3, t, k):                                                                                                                                                     
        n = arr.shape[0]                                                                                                                                 
        if t > n:                                                                                                                                                                                 
            raise ValueError("t cannot be greater than the number of samples")                                                                                                                    
                                                                                                                                                                                                  
        start_index = 0 #np.random.randint(0, n - (t)*k + 1)                                                                                                                
        end_index = start_index + (t)*k                                                                                                                                        
        selected_samples = arr[start_index:end_index:k]   
        selected_samples2 = arr2[start_index:end_index:k]  # select with step k
        selected_samples3 = arr3[start_index:end_index:k]  # select with step k                                                                                                         
        return selected_samples, selected_samples2, selected_samples3

    def _process_csv_row(self, processed_file_path, force_file_path, vel_file_path, pdb_file_path):
        # here to sample frame_time continuous positions.
        processed_feats = dict(np.load(processed_file_path,allow_pickle=True,mmap_mode='r'))
        with open(force_file_path, 'rb') as f:
           force_feats = pickle.load(f)
        with open(vel_file_path, 'rb') as f:
           vel_feats = pickle.load(f)

        frame_time = self.data_conf.frame_time

        if self._is_training:
            tmp, tmp2, tmp3 = self.select_random_samples(processed_feats['all_atom_positions'][:self.data_conf.keep_first],force_feats[:self.data_conf.keep_first],vel_feats[:self.data_conf.keep_first],t=frame_time,k=self.data_conf.frame_sample_step)
        else:
            tmp, tmp2, tmp3 = self.select_first_samples(processed_feats['all_atom_positions'][self.data_conf.fix_sample_start:],force_feats[self.data_conf.fix_sample_start:],vel_feats[self.data_conf.fix_sample_start:],t=frame_time,k=self.data_conf.frame_sample_step)


        new_feats = {k:v for k,v in processed_feats.items() if k != 'all_atom_positions'}
        new_feats['all_atom_positions'] = tmp
        new_feats = parse_dynamics_chain_feats_no_norm(new_feats)

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
        processed_file_path = csv_row['dynamic_npz']
        force_file_path = csv_row['force_path']
        #print(force_file_path)
        vel_file_path = csv_row['vel_path']
        pdb_file_path = csv_row['pdb_path']

        chain_feats = self._process_csv_row(processed_file_path, force_file_path, vel_file_path, pdb_file_path)
        frame_time = chain_feats['aatype'].shape[0]
        node_edge_feature_path = csv_row['embed_path']  # here
        attr_dict = dict(np.load(node_edge_feature_path))
        chain_feats.update({'node_repr':torch.tensor(attr_dict['node_repr'])})
        chain_feats.update({'edge_repr':torch.tensor(attr_dict['edge_repr'])})
        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:,:, 0] 
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

