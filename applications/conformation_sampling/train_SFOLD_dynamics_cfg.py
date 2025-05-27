"""Pytorch script for training SE(3) protein diffusion.

To run:

> python experiments/train_se3_diffusion.py

Without Wandb,

> python experiments/train_se3_diffusion.py experiment.use_wandb=False

To modify config options with the command line,

> python experiments/train_se3_diffusion.py experiment.batch_size=32

"""
import os
import torch
import GPUtil
import time
import tree
import numpy as np
import wandb
import copy
import hydra
import logging
import copy
import random
import pandas as pd
from collections import defaultdict,deque
from datetime import datetime
from omegaconf import DictConfig,OmegaConf
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from Bio.SVDSuperimposer import SVDSuperimposer
import gc
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
import tempfile

from src.data import Sfold_data_loader_dynamic
from src.analysis import utils as au
from src.analysis import metrics

from src.data import se3_diffuser,all_atom
from src.data import utils as du

from src.model import Sfoldnetwork_dynamic
from src.experiments import utils as eu
from openfold.utils.loss import lddt, lddt_ca,torsion_angle_loss
from openfold.np import residue_constants#
from openfold.utils.superimposition import superimpose
from openfold.utils.validation_metrics import gdt_ts,gdt_ha,drmsd
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils import rigid_utils as ru
from openfold.data import data_transforms
from openfold.np import protein
from openfold.data.data_pipeline import make_protein_features

from src.toolbox.rot_trans_error import average_quaternion_distances,average_translation_distances
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis import rms,align,rdf,contacts
from scipy.stats import pearsonr
import pickle
import atexit
def format_func(value, tick_number):
    return f'{value:.1f}'
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(format_func)

from typing import List
import mdtraj as mdtraj

#https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
def rmsdalign(a, b, weights=None): # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return b @ C.mT + a_mean


def svd_superimpose_and_align(gt, pred, gt_all, pred_all, pred_atom_mask):

    """
    Perform SVD-based alignment on each corresponding first-dimension element of two tensors.
    Calculate the RMSD, rotation matrix, and translation vector.
    Apply the obtained rotation and translation to pred_all to align it with gt_all,
    then multiply the aligned pred_all with pred_atom_mask and calculate the MSE.

    Parameters:
    gt: torch.Tensor, shape [N, 38, 3]
    pred: torch.Tensor, shape [N, 38, 3]
    gt_all: torch.Tensor, shape [N, 38, 37, 3]
    pred_all: torch.Tensor, shape [N, 38, 37, 3]
    pred_atom_mask: torch.Tensor, shape [N, 38, 37]

    Returns:
    rmsd_list: List[float], RMSD values for each pair of point sets
    mse_list: List[float], MSE values between aligned pred_all and gt_all after applying mask
    """
    # Check if input tensors have the correct shapes
    if gt.shape != pred.shape or gt_all.shape != pred_all.shape or gt.shape[0] != gt_all.shape[0] or gt_all.shape[:3] != pred_atom_mask.shape:
        print(f"gt shape: {gt.shape}")
        print(f"pred shape: {pred.shape}")
        print(f"gt_all shape: {gt_all.shape}")
        print(f"pred_all shape: {pred_all.shape}")
        print(f"pred_atom_mask shape: {pred_atom_mask.shape}")
        raise ValueError("Input tensors must have matching shapes")

    # Convert tensors from Torch to Numpy arrays
    points1 = gt.numpy()
    points2 = pred.numpy()
    points1_all = gt_all.numpy()
    points2_all = pred_all.numpy()
    mask = pred_atom_mask.numpy()

    # Initialize result lists
    rmsd_list = []
    mse_list = []

    # Create SVDSuperimposer object
    sup = SVDSuperimposer()

    # Iterate over the first dimension
    for i in range(points1.shape[0]):
        # Get the corresponding sets of points
        fixed_points = points1[i]
        moving_points = points2[i]

        # Set fixed and moving points for SVD superimposition
        sup.set(fixed_points, moving_points)

        # Run the SVD superimposition
        sup.run()

        # Get the RMSD
        rmsd = sup.get_rms()
        # Get the rotation matrix and translation vector
        rotation, translation = sup.get_rotran()

        # Append the RMSD to the list
        rmsd_list.append(torch.tensor(rmsd))

        # Apply rotation and translation to pred_all
        pred_all_aligned = np.dot(points2_all[i], rotation.T) + translation

        # Expand the mask to match the shape of pred_all_aligned
        expanded_mask = np.expand_dims(mask[i], axis=-1)

        # Apply the mask to the aligned pred_all
        pred_all_masked = pred_all_aligned * expanded_mask

        # Calculate the MSE between masked pred_all and gt_all
        mse = np.mean((points1_all[i] - pred_all_masked) ** 2)
        # Append the MSE to the list
        mse_list.append(torch.tensor(mse))

    return torch.stack(rmsd_list,dim=0), torch.stack(mse_list,dim=0)

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    #print(score.shape, dists_to_score.shape)#torch.Size([1, 135])
    #exit()
    return score


def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = False,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


@torch.no_grad()
def compute_validation_metrics_all(gt_pos, out_pos,gt_mask,superimposition_metrics=False):
    metrics = {}
        
    gt_coords =gt_pos# batch["atom37_pos"]
    pred_coords = out_pos#outputs["atom37_pos"]
    all_atom_mask = gt_mask

    gt_coords = gt_coords#torch.from_numpy(gt_coords)
    pred_coords = torch.from_numpy(pred_coords)
    all_atom_mask = all_atom_mask#torch.from_numpy(all_atom_mask)

    # print(gt_coords.shape,pred_coords.shape, all_atom_mask[..., None].shape)
    # This is super janky for superimposition. Fix later
    gt_coords_masked = gt_coords * all_atom_mask[..., None]
    pred_coords_masked = pred_coords * all_atom_mask[..., None] 

    frame_time = gt_coords_masked.shape[0]
    gt_coords_masked_sum = gt_coords_masked.reshape([frame_time, -1, 3]).sum(dim=1,keepdim=True)
    all_atom_mask_sum = all_atom_mask[..., None].reshape([frame_time, -1, 1]).sum(dim=1, keepdim=True)
    gt_coords_masked_centered = gt_coords_masked.reshape([frame_time, -1, 3]) - gt_coords_masked_sum / all_atom_mask_sum
    gt_coords_masked = gt_coords_masked_centered.reshape([frame_time, -1, 37, 3]) *  all_atom_mask[..., None]

    pred_coords_masked_sum = pred_coords_masked.reshape([frame_time, -1, 3]).sum(dim=1,keepdim=True)
    all_atom_mask_sum = all_atom_mask[..., None].reshape([frame_time, -1, 1]).sum(dim=1, keepdim=True)
    pred_coords_masked_centered = pred_coords_masked.reshape([frame_time, -1, 3]) - pred_coords_masked_sum / all_atom_mask_sum
    pred_coords_masked = pred_coords_masked_centered.reshape([frame_time, -1, 37, 3]) *  all_atom_mask[..., None]    

    ca_pos = residue_constants.atom_order["CA"]
    gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :] # [11,N,3]
    pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]# [11,N,3]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]

    def align_rot(A, B):                                                                                                                                                                      
        H = A.T.mm(B)                                                                                                                                                                     
        U, S, Vt = torch.linalg.svd(H)                                                                                                                                                    
        R = Vt.T @ U.T                                                                                                                                                                    
        if torch.linalg.det(R) < 0.0:
            #print("reflection")
            Vt[2] = -Vt[2]
            R = Vt.T @ U.T                                                                                                                                                       
        return R.T

    N = pred_coords_masked_ca.shape[0]
    #for i in range(1, N):
    #    rot_i = align_rot(pred_coords_masked_ca[i], pred_coords_masked_ca[0])
    #    pred_coords_masked_ca[i] = torch.matmul(pred_coords_masked_ca[i], rot_i)
    #    pred_coords_masked[i] = torch.matmul(pred_coords_masked[i].reshape([frame_time, -1, 3]), rot_i).reshape([frame_time, -1, 37, 3])

    pairwise_rmsd = 0
    for i in range(N):
        for j in range(N):
            #pred_coords_masked_ca_j = pred_coords_masked_ca[j] - torch.mean(pred_coords_masked_ca[j], dim=0, keepdim=True)
            #pred_coords_masked_ca_i = pred_coords_masked_ca[i] - torch.mean(pred_coords_masked_ca[i], dim=0, keepdim=True)
            pred_coords_masked_ca_j = pred_coords_masked_ca[j]
            pred_coords_masked_ca_i = pred_coords_masked_ca[i]
            rot_i = align_rot(pred_coords_masked_ca_j, pred_coords_masked_ca_i)
            pred_coords_masked_ca_j = torch.matmul(pred_coords_masked_ca_j, rot_i) 
            #rmsd = torch.sqrt(torch.mean(torch.linalg.norm(pred_coords_masked_ca_j - pred_coords_masked_ca[i], dim=1)**2))
            rmsd = torch.sqrt(torch.sum((pred_coords_masked_ca_j - pred_coords_masked_ca_i)**2) / pred_coords_masked_ca_i.shape[0])
            pairwise_rmsd = pairwise_rmsd + rmsd
    
    pairwise_rmsd = pairwise_rmsd / N / N
    metrics['pairwise_rmsd'] = pairwise_rmsd.reshape([1,1])


    pairwise_rmsd = 0                                                                                                                                                                             
    for i in range(N):                                                                                                                                                                            
        for j in range(N):                                                                                                                                                                        
            #pred_coords_masked_ca_j = pred_coords_masked_ca[j] - torch.mean(pred_coords_masked_ca[j], dim=0, keepdim=True)                                                                       
            #pred_coords_masked_ca_i = pred_coords_masked_ca[i] - torch.mean(pred_coords_masked_ca[i], dim=0, keepdim=True)                                                                       
            gt_coords_masked_ca_j = gt_coords_masked_ca[j]                                                                                                                                    
            gt_coords_masked_ca_i = gt_coords_masked_ca[i]                                                                                                                                    
            rot_i = align_rot(gt_coords_masked_ca_j, gt_coords_masked_ca_i)                                                                                                                   
            gt_coords_masked_ca_j = torch.matmul(gt_coords_masked_ca_j, rot_i)                                                                                                                
            #rmsd = torch.sqrt(torch.mean(torch.linalg.norm(pred_coords_masked_ca_j - pred_coords_masked_ca[i], dim=1)**2))                                                                       
            rmsd = torch.sqrt(torch.sum((gt_coords_masked_ca_j - gt_coords_masked_ca_i)**2) / gt_coords_masked_ca_i.shape[0])                                                               
            pairwise_rmsd = pairwise_rmsd + rmsd                                                                                                                                                  
                                                                                                                                                                                                  
    pairwise_rmsd = pairwise_rmsd / N / N                                                                                                                                                         
    metrics['gt_pairwise_rmsd'] = pairwise_rmsd.reshape([1,1])

    '''
    pred_coords_masked_ca_npy = pred_coords_masked_ca.cpu().detach().numpy()
    pairwise_rmsd = 0 
    for i in range(N):
        for j in range(N):
            sup = SVDSuperimposer()
            sup.set(pred_coords_masked_ca_npy[j], pred_coords_masked_ca_npy[i])
            sup.run()
            sup.get_transformed()
            rmsd = sup.get_rms()
            pairwise_rmsd = pairwise_rmsd + rmsd

    pairwise_rmsd = pairwise_rmsd / N / N                                                                                                                                                         
    metrics['pairwise_rmsd_bio'] = torch.tensor([pairwise_rmsd]).to(pred_coords_masked_ca.device).reshape([1,1])
    '''
    min_rmsds = 0
    for i in range(N):
        min_rmsd = 0.0
        for j in range(N):
            #rot_i = align_rot(pred_coords_masked_ca[j], gt_coords_masked_ca[i])
            #pred_coords_masked_ca_j = torch.matmul(pred_coords_masked_ca[j], rot_i)
            #rmsd = torch.sqrt(torch.sum((pred_coords_masked_ca_j - gt_coords_masked_ca[i])**2) / pred_coords_masked_ca_j.shape[0]) 
            #rmsd = torch.sqrt(torch.mean(torch.linalg.norm(pred_coords_masked_ca_j - gt_coords_masked_ca[i], dim=1)**2))
            #print(pred_coords_masked.shape,  all_atom_mask.shape)
            #exit()
            rmsd = lddt_ca(pred_coords_masked[j:j+1], gt_coords_masked[i:i+1], all_atom_mask[i:i+1])
            #print(rmsd)
            #exit()
            if rmsd > min_rmsd:
                min_rmsd = rmsd

        min_rmsds = min_rmsds + min_rmsd
    min_rmsds = min_rmsds / N
    metrics['min_rmsds'] = min_rmsds.reshape([1,1])

    coverage = 0
    for i in range(N):
        min_rmsd = 0.0
        for j in range(N):
            #rot_i = align_rot(gt_coords_masked_ca[j], pred_coords_masked_ca[i])
            #gt_coords_masked_ca_j = torch.matmul(gt_coords_masked_ca[j], rot_i)
            #rmsd = torch.sqrt(torch.sum((pred_coords_masked_ca[i] - gt_coords_masked_ca_j)**2) / pred_coords_masked_ca[i].shape[0])
            #rmsd = torch.sqrt(torch.mean(torch.linalg.norm(pred_coords_masked_ca_j - gt_coords_masked_ca[i], dim=1)**2))
            rmsd = lddt_ca(gt_coords_masked[j:j+1], pred_coords_masked[i:i+1], all_atom_mask[i:i+1])
            if rmsd > min_rmsd:
                min_rmsd = rmsd

        coverage = coverage + min_rmsd
    coverage = coverage / N
    metrics['coverage'] = coverage.reshape([1,1])
    print(metrics)
    #print(gt_coords.shape, gt_coords_masked_ca.shape) torch.Size([32, 135, 37, 3]) torch.Size([32, 135, 3])
    #exit()
    #[11,N]
    drmsd_ca_score = drmsd(
        pred_coords_masked_ca,
        gt_coords_masked_ca,
        mask=all_atom_mask_ca, # still required here to compute n
    )

    metrics["drmsd_ca"] = drmsd_ca_score



    frame_time,Nseq,_,_ = gt_coords_masked.shape

    # gt_coords_masked = gt_coords_masked.reshape([frame_time,-1,3])
    # pred_coords_masked = pred_coords_masked.reshape([frame_time,-1,3]) 

    diff = gt_coords_masked.reshape([frame_time,-1,3]) - pred_coords_masked.reshape([frame_time,-1,3])  # [F,N*37,3] 
    # print(diff.shape,all_atom_mask.shape)
    # torch.Size([4, 2516(68*37), 3]) torch.Size([4, 68, 37]) )
    #all_atom_mask [F,N,37]

    # RMSE
    metrics["rmse_all"]  = torch.sqrt(torch.sum(diff**2,axis=(-1,-2))/(torch.sum(all_atom_mask, dim=(-1, -2)) + 1e-4))
    diff = gt_coords_masked_ca - pred_coords_masked_ca # [F,N,3]
    #all_atom_mask_ca [F,N]
    metrics["rmse_ca"]  = torch.sqrt(torch.sum(diff**2,axis=(-1,-2))/(torch.sum(all_atom_mask_ca, dim=-1) + 1e-4))
    #print(metrics["rmse_ca"].shape, metrics["drmsd_ca"].shape, metrics['coverage'].shape) 
    #exit()
    # RMSD with SVDSuperimposer()
    # alignment_rmsd_ca,alignment_rmsd = svd_superimpose_and_align(gt=gt_coords_masked_ca,pred=pred_coords_masked_ca,
    #                                 gt_all=gt_coords_masked,pred_all=pred_coords_masked,
    #                                 pred_atom_mask=all_atom_mask)
    superimposed_pred, alignment_rmsd_ca = superimpose(
        gt_coords_masked_ca, pred_coords_masked_ca,
    )
    # print(alignment_rmsd_ca,alignment_rmsd)
    # exit()
    metrics["rmsd_ca_aligned"] = alignment_rmsd_ca
    # metrics["alignment_rmsd"] = alignment_rmsd
    return metrics

@torch.no_grad()
def plot_curve_merged(metric_merged,save_path,row_num=2,col_num=5,suffer_fix=None):
    total_width = col_num * 2
    total_height = row_num * 2
    fig, axes = plt.subplots(row_num, col_num,figsize=(total_width, total_height),dpi=300)
    # 输出每一行的信息
    for key in metric_merged.keys():
        data = metric_merged[key]
        for index, row in data.iterrows():
            name = row['pdb_name']
            col_id = index
            if col_num == 1:
                axes[0].plot(row['rmse_all'],label=key, marker='o', linestyle='-')
                axes[1].plot(row['rmsd_ca_aligned'],label=key, marker='o', linestyle='-')
                axes[2].plot(row['rmse_ca'],label=key, marker='o', linestyle='-')

                axes[0].set_title(name+' | RMSE')
                axes[1].set_title(name+' | RMSD_ca_a')
                axes[2].set_title(name+' | RMSE_ca')
            else:
                axes[0, col_id].plot(row['rmse_all'],label=key, marker='o', linestyle='-')
                axes[1, col_id].plot(row['rmsd_ca_aligned'],label=key, marker='o', linestyle='-')
                axes[2, col_id].plot(row['rmse_ca'],label=key, marker='o', linestyle='-')

                axes[0, col_id].set_title(name+' | RMSE')
                axes[1, col_id].set_title(name+' | RMSD_ca_a')
                axes[2, col_id].set_title(name+' | RMSE_ca')
        # 在每个子图上显示图例
    plt.suptitle('RSME over Atoms')
    plt.tight_layout()
    plt.legend()
    # plt.axis('off')
    if suffer_fix is not None:
        plt.savefig(f'{save_path}/rmse_rmsd_{suffer_fix}.png')
    else:
        plt.savefig(f'{save_path}/rmse_rmsd.png')
    return fig


# rot_trans_error_dict = {'name':[],"ave_rot":[],"ave_trans":[],"first_rot":[],"first_trans":[]}
@torch.no_grad()
def plot_rot_trans_curve(error_dict,save_path,frame_step=1):
    rows,cols = 2,len(error_dict['name'])
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2,rows*2),dpi=300)
    # motion_pkl_path = f'/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/DFOLDv2/plot/4ue8_B_simulation_1_frame_step_{frame_step}_wj.pickle'# 1a62_A_ 4ue8_B
    # try:# atlas_average_motion_traj1
    #     with open(motion_pkl_path, 'rb') as handle:
    #         loaded_data = pickle.load(handle)
    # except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
    #     # 在这里捕获可能的异常并跳过处理
    #     loaded_data=None
    loaded_data=None
    for idx,name in enumerate(error_dict['name']):
        if cols==1:
            axes[0].plot(error_dict['ave_rot'][idx],label='Pred', marker='o', linestyle='-')
            axes[0].plot(error_dict['first_rot'][idx],label='RefAsPred', marker='o', linestyle='-')
            # x = np.arange(1, len(error_dict['first_rot'][idx]))
            # axes[0].plot(x,error_dict['time_rot_dif'][idx][1:],label='RM', marker='o', linestyle='-')

            axes[1].plot(error_dict['ave_trans'][idx],label='Pred', marker='o', linestyle='-')
            axes[1].plot(error_dict['first_trans'][idx],label='RefAsPred', marker='o', linestyle='-')
            # axes[1].plot(x,error_dict['time_trans_dif'][idx][1:],label='RM', marker='o', linestyle='-')

            # plot percent compare with traj motion
            if loaded_data is not None:
                rots_traj_motion = loaded_data[name]['ARC_rot']
                rots_traj_motion = np.array([rots_traj_motion]* len(error_dict['ave_rot'][idx]))
                axes[0].plot(rots_traj_motion,label='TrajMotion', marker='o', linestyle='-')

                percent_rots = error_dict['ave_rot'][idx]/rots_traj_motion

                x  = np.arange(0, len(percent_rots))
                for i in range(len(x)):
                    axes[0].annotate(f'{percent_rots[i]:.2f}',
                                xy=(x[i], error_dict['ave_rot'][idx][i]),
                                xytext=(2, 0),  # points vertical offset
                                textcoords="offset points",
                                ha='left', va='center',fontsize=8)


                trans_traj_motion = loaded_data[name]['ARC_trans_MSE']
                trans_traj_motion = np.array([trans_traj_motion]* len(error_dict['ave_trans'][idx]))
                axes[1].plot(trans_traj_motion,label='TrajMotion', marker='o', linestyle='-')

                percent_trans = error_dict['ave_trans'][idx]/trans_traj_motion

                x  = np.arange(0, len(percent_trans))
                for i in range(len(x)):
                    axes[1].annotate(f'{percent_trans[i]:.2f}',
                                xy=(x[i], error_dict['ave_trans'][idx][i]),
                                xytext=(2, 0),  # points vertical offset
                                textcoords="offset points",
                                ha='left', va='center',fontsize=8)


            axes[0].set_title(name)

            axes[1].yaxis.set_major_formatter(formatter)
            axes[0].set_ylabel('Rotation/°')
            axes[1].set_ylabel('Translation/Å')
        else:
            axes[0,idx].plot(error_dict['ave_rot'][idx],label='Pred', marker='o', linestyle='-')
            axes[0,idx].plot(error_dict['first_rot'][idx],label='RefAsPred', marker='o', linestyle='-')
            # x = np.arange(1, len(error_dict['first_rot'][idx]))
            # axes[0,idx].plot(x,error_dict['time_rot_dif'][idx][1:],label='RM', marker='o', linestyle='-')


            axes[1,idx].plot(error_dict['ave_trans'][idx],label='Pred', marker='o', linestyle='-')
            axes[1,idx].plot(error_dict['first_trans'][idx],label='RefAsPred', marker='o', linestyle='-')
            # axes[1,idx].plot(x,error_dict['time_trans_dif'][idx][1:],label='RM', marker='o', linestyle='-')

            if loaded_data is not None:
                rots_traj_motion = loaded_data[name]['ARC_rot']
                rots_traj_motion = np.array([rots_traj_motion]* len(error_dict['ave_rot'][idx]))
                axes[0,idx].plot(rots_traj_motion,label='TrajMotion', marker='o', linestyle='-')

                percent_rots = error_dict['ave_rot'][idx]/rots_traj_motion
                x  = np.arange(0, len(percent_rots))
                for i in range(len(x)):
                    axes[0,idx].annotate(f'{percent_rots[i]:.2f}',
                                xy=(x[i], error_dict['ave_rot'][idx][i]),
                                xytext=(2, 0),  # points vertical offset
                                textcoords="offset points",
                                ha='left', va='center',fontsize=8)


                trans_traj_motion = loaded_data[name]['ARC_trans_MSE']
                trans_traj_motion = np.array([trans_traj_motion]* len(error_dict['ave_trans'][idx]))
                axes[1,idx].plot(trans_traj_motion,label='TrajMotion', marker='o', linestyle='-')

                percent_trans = error_dict['ave_trans'][idx]/trans_traj_motion
                x  = np.arange(0, len(percent_trans))
                for i in range(len(x)):
                    axes[1,idx].annotate(f'{percent_trans[i]:.2f}',
                                xy=(x[i], error_dict['ave_trans'][idx][i]),
                                xytext=(2, 0),  # points vertical offset
                                textcoords="offset points",
                                ha='left', va='center',fontsize=8)

            axes[0, idx].set_title(name)

            axes[1,idx].yaxis.set_major_formatter(formatter)
            if idx==0:
                axes[0,idx].set_ylabel('Rotation/°')
                axes[1,idx].set_ylabel('Translation/Å')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{save_path}/rot_trans_error.png')
    return fig

@torch.no_grad()
def plot_curve(metric_dict,save_path,row_num=2,col_num=5,suffer_fix=None):
    fig, axes = plt.subplots(row_num, col_num, figsize=(16, 12),dpi=300)
    data = metric_dict
    # 输出每一行的信息
    for index, row in data.iterrows():
        name = row['pdb_name']
        row_id = index//col_num
        col_id = index% col_num
        axes[row_id, col_id].plot(row['rmse_all'],label='RMSE')
        axes[row_id, col_id].plot(row['rmsd_ca_aligned'],label='RMSD_ca_a')
        axes[row_id, col_id].plot(row['rmse_ca'],label='RMSE_ca')
        axes[row_id, col_id].set_title(name)
        # 在每个子图上显示图例
    plt.suptitle('RSME over Atoms')
    plt.tight_layout()
    plt.legend()
    # plt.axis('off')
    if suffer_fix is not None:
        plt.savefig(f'{save_path}/rmse_rmsd_{suffer_fix}.png')
    else:
        plt.savefig(f'{save_path}/rmse_rmsd.png')
    return fig

@torch.no_grad()
def calculate_rmsf(pdb_file, reference_select="protein and name CA"):
    u = mda.Universe(pdb_file)
    atoms = u.select_atoms(reference_select)
    aligner = align.AlignTraj(u, atoms, select=reference_select, in_memory=True).run()
    atoms = u.select_atoms(reference_select)
    rmsf_analysis = rms.RMSF(atoms).run()
    return rmsf_analysis.rmsf

class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join([str(x) for x in GPUtil.getAvailable(order='memory', limit = 8)])

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_tensorboard = self._exp_conf.use_tensorboard
        self._use_ddp = self._exp_conf.use_ddp
        self.dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        # 1. initialize ddp info if in ddp mode
        # 2. silent rest of logger when use ddp mode
        # 3. silent wandb logger
        # 4. unset checkpoint path if rank is not 0 to avoid saving checkpoints and evaluation
        if self._use_ddp :
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_tensorboard = False
                self._exp_conf.ckpt_dir = None


        self.trained_epochs = 0
        self.trained_steps = 0

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = Sfoldnetwork_dynamic.FullScoreNetwork(self._model_conf, self.diffuser)
        
        if conf.experiment.warm_start:
            ckpt_path = conf.experiment.warm_start
            self.load_pretrianed_model(ckpt_path=ckpt_path)
        # print(next(self._model.parameters()).device)

        num_parameters = sum(p.numel() for p in self._model.parameters())

        if self._conf.model.ipa.temporal and self._conf.model.ipa.frozen_spatial:
            self._log.info('Frozen model and only train temporal module')
            # only train motion module
            for param in self._model.parameters():
                param.requires_grad = False
            for name, param in self._model.named_parameters():
                if 'temporal' in name: # 'frame'
                    param.requires_grad = True

        # 冻结/解冻后计算总参数数量（应与初始值相同）
        trainable_num_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self._exp_conf.num_parameters = num_parameters
        self._exp_conf.trainable_num_parameters  = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}, trainable parameters:{trainable_num_parameters}')
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._exp_conf.learning_rate,amsgrad=True)#,weight_decay=1e-3

        self._init_log()
        self._init_best_eval()
        if not self.conf.experiment.training:
            seed = 0
        else:
            seed = dist.get_rank()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        
        
    def _init_best_eval(self):
        self.best_trained_steps = 0
        self.best_trained_epoch = 0
        self.best_rmse_ca = 10000
        self.best_rmse_all = 10000
        self.best_drmsd = 10000
        self.best_rmsd_ca_aligned = 10000
        self.best_rot_error=1000
        self.best_trans_error = 1000
        self.best_ref_rot_error = 1000
        self.best_ref_trans_error = 1000

    def _init_log(self):

        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
                self.dt_string )
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self._log.info('Checkpoint not being saved.')

        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                self.dt_string )
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')
        # self._aux_data_history = deque(maxlen=100)


    def load_pretrianed_model(self, ckpt_path):
        try:
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = torch.load(ckpt_path, map_location='cpu')

            if ckpt_pkl is not None and 'model' in ckpt_pkl:
                ckpt_model = ckpt_pkl['model']
                # if 'epoch' in ckpt_pkl:
                # self.trained_epochs = ckpt_pkl['epoch']
                # if 'step' in ckpt_pkl:
                # self.trained_steps = ckpt_pkl['step']

                if ckpt_model is not None:
                    ckpt_model = {k.replace('module.', ''): v for k, v in ckpt_model.items()}
                    model_state_dict = self._model.state_dict()
                    # pretrained_dict = {k: v for k, v in ckpt_model.items() if k in model_state_dict}
                    pretrained_dict = {k: v for k, v in ckpt_model.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                    model_state_dict.update(pretrained_dict)
                    self._model.load_state_dict(model_state_dict)
                    self._log.info(f'Warm starting from: {ckpt_path}')
                    del ckpt_pkl,ckpt_model,pretrained_dict,model_state_dict
                    gc.collect()
                    return True
                else:
                    self._log.error("Checkpoint model is None.")
                    return False
            else:
                self._log.error("Checkpoint or model not found in checkpoint file.")
                return False
        except Exception as e:
            self._log.error(f"Error loading checkpoint: {e}")
            return False


    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):
        
        if self._data_conf.is_extrapolation:
            train_dataset = Sfold_data_loader_dynamic.PdbDatasetExtrapolation(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
            )

            valid_dataset = Sfold_data_loader_dynamic.PdbDatasetExtrapolation(
                data_conf=self._data_conf,
                diffuser=self._diffuser,
                is_training=False
            )
        else:
            # Datasets
            train_dataset = Sfold_data_loader_dynamic.PdbDataset(
                data_conf=self._data_conf,
                diffuser=self._diffuser,
                is_training=True
            )

            valid_dataset = Sfold_data_loader_dynamic.PdbDataset(
                data_conf=self._data_conf,
                diffuser=self._diffuser,
                is_training=False
            )
        # Loaders
        num_workers = self._exp_conf.num_loader_workers

        persistent_workers = True if num_workers > 0 else False
        prefetch_factor=2
        prefetch_factor = 2 if num_workers == 0 else prefetch_factor

        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        train_loader = data.DataLoader(
                train_dataset,
                batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                sampler=sampler,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )
        valid_loader = data.DataLoader(
                valid_dataset,
                batch_size=self._exp_conf.eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )



        return train_loader, valid_loader

        
    def init_tensorboard(self):
        self._log.info('Initializing TensorBoard.')
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        # Initialize TensorBoard SummaryWriter
        tensorboard_log_dir = os.path.join(self._exp_conf.tensorboard_dir,self._exp_conf.name, self.dt_string ,self._exp_conf.name)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        # Log configuration details
        self.writer.add_text('Config', str(conf_dict))
        # You can also log other initial details if needed
        self._exp_conf.run_id = 'unique_run_id'  # You may need to generate this appropriately
        self._log.info(f'TensorBoard: run_id={self._exp_conf.run_id}, log_dir={tensorboard_log_dir}')


    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_tensorboard and replica_id == 0:
                self.init_tensorboard()
        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)
        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus==1 :
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            #muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                device_ids = [f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.num_gpus]]
                #DDP mode
                if self._use_ddp :
                    device = torch.device("cuda",self.ddp_info['local_rank'])
                    model = self.model.to(device)
                    self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'],find_unused_parameters=False)
                    self._log.info(f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}, devices: {device_ids}")
                #DP mode
                else:
                    if len(self._available_gpus) < self._exp_conf.num_gpus:
                        raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                    self._log.info(f"Multi-GPU training on GPUs in DP mode: {device_ids}")
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        # if self.conf.experiment.warm_start:
        #     for state in self._optimizer.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.to(device)

        self._model.train()
                    
        (train_loader,valid_loader) = self.create_dataset()

        logs = []
        # torch.cuda.empty_cache()
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            self.trained_epochs = epoch
            train_loader.sampler.set_epoch(epoch)
            epoch_log = self.train_epoch(
                train_loader,
                valid_loader,
                device,
                return_logs=return_logs
            )
            # self._schedule.step()

            if return_logs:
                logs.append(epoch_log)
        if self._exp_conf.ckpt_dir is not None:
            ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'last_step_{self.trained_steps}.pth')
            du.write_checkpoint(
                ckpt_path,
                copy.deepcopy(self.model.state_dict()),
                self._conf,
                copy.deepcopy(self._optimizer.state_dict()),
                self.trained_epochs,
                self.trained_steps,
                logger=self._log,
                use_torch=True
            )
        self._log.info('Done')
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self._optimizer.step()
        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device,return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        # Run evaluation
        
        # start_time = time.time()
        # self._model.eval()
        # eval_dir = os.path.join( self._exp_conf.eval_dir, f'debug')
        # eval_dir = os.path.join(self._exp_conf.eval_dir, f'step_{self.trained_steps}')
        # if not os.path.exists(eval_dir):
        #     os.makedirs(eval_dir, exist_ok=True)
        # ckpt_metrics = self.eval_fn(
        #     eval_dir, valid_loader, device,
        #     noise_scale=self._exp_conf.noise_scale
        # )
        # exit()
        for train_feats in train_loader:
            self.model.train()
            train_feats = tree.map_structure(lambda x: x.to(device), train_feats)
            
            # TODO flatten the dim of batch and frame_time
            for k in train_feats.keys():
                v = train_feats[k]
                if len(v.shape)>1:
                    reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维
                    train_feats[k] = reshaped_tensor

            loss, aux_data = self.update_fn(train_feats)
            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(f'Epoch[{self.trained_epochs}/{self._exp_conf.num_epoch}] trained_steps: [{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # # catch abnormal model
            # if self.trained_steps>10000 and aux_data['total_loss']>10:
            #     ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'catched_model_{self.trained_steps}.pth')
            #     du.write_checkpoint(
            #         ckpt_path,
            #         copy.deepcopy(self.model.state_dict()),
            #         self._conf,
            #         copy.deepcopy(self._optimizer.state_dict()),
            #         self.trained_epochs,
            #         self.trained_steps,
            #         logger=self._log,
            #         use_torch=True
            #     )
            #     exit()

            # Take checkpoint
            if self._exp_conf.ckpt_dir is not None and ((self.trained_steps % self._exp_conf.ckpt_freq) == 0 or (self._exp_conf.early_ckpt and self.trained_steps == 100)):
            #if self._exp_conf.ckpt_dir is not None and self.trained_steps > 1:
                ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                du.write_checkpoint(
                    ckpt_path,
                    copy.deepcopy(self.model.state_dict()),
                    self._conf,
                    copy.deepcopy(self._optimizer.state_dict()),
                    self.trained_epochs,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )

                # Run evaluation
                self._log.info(f'Running evaluation of {ckpt_path}')
                start_time = time.time()
                eval_dir = os.path.join(self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                os.makedirs(eval_dir, exist_ok=True)
                # ckpt_metrics,curve_fig,curve_fig_aligned,error_fig,model_ckpt_update,rot_trans_error_mean = self.eval_fn(
                #     eval_dir, valid_loader, device,
                #     noise_scale=self._exp_conf.noise_scale
                # )
                ckpt_metrics,curve_fig,curve_fig_aligned,model_ckpt_update = self.eval_fn(
                    eval_dir, valid_loader, device,
                    noise_scale=self._exp_conf.noise_scale
                )
                eval_time = time.time() - start_time
                self._log.info(f'Finished evaluation in {eval_time:.2f}s')
            else:
                ckpt_metrics = None
                eval_time = None
                model_ckpt_update=False
            # Remote log to Wandb.
            if self._use_tensorboard:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                # Logging basic metrics
                self.writer.add_scalar('Loss/Total', loss, self.trained_steps)
                self.writer.add_scalar('Loss/Rotation', aux_data['rot_loss'], self.trained_steps)
                self.writer.add_scalar('Loss/Translation', aux_data['trans_loss'], self.trained_steps)
                self.writer.add_scalar('Loss/Torsion', aux_data['torsion_loss'], self.trained_steps)
                self.writer.add_scalar('Loss/BB_Atom', aux_data['bb_atom_loss'], self.trained_steps)
                self.writer.add_scalar('Loss/Dist_Mat', aux_data['dist_mat_loss'], self.trained_steps)
                # self.writer.add_scalar('Other/Examples_per_sec', example_per_sec, self.trained_steps)
                # self.writer.add_scalar('Other/Batch_size', aux_data['examples_per_step'], self.trained_steps)
                # self.writer.add_scalar('Other/Res_length', aux_data['res_length'], self.trained_steps)
                # self.writer.add_scalar('Other/Num_epochs', self.trained_epochs, self.trained_steps)
                self.writer.add_scalar('UpRigids/Rot0', aux_data['update_rots'][0], self.trained_steps)
                self.writer.add_scalar('UpRigids/Rot1', aux_data['update_rots'][1], self.trained_steps)
                self.writer.add_scalar('UpRigids/Rot2', aux_data['update_rots'][2], self.trained_steps)
                self.writer.add_scalar('UpRigids/Trans0', aux_data['update_trans'][0], self.trained_steps)
                self.writer.add_scalar('UpRigids/Trans1', aux_data['update_trans'][1], self.trained_steps)
                self.writer.add_scalar('UpRigids/Trans2', aux_data['update_trans'][2], self.trained_steps)

                # for i in range(self._model_conf.ipa.num_blocks):
                #     bb_grads = [p.grad for name,p in self.model.named_parameters() if p.grad is not None and f'bb_update_{i}' in name ]
                #     bb_grads_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in bb_grads]), 2.0)
                #     self.writer.add_scalar(f'Grad/BB_update_{i}', bb_grads_norm, self.trained_steps)
                # for i in range(self._model_conf.ipa.num_blocks):
                bb_grads = [p.grad for name,p in self.model.named_parameters() if p.grad is not None and f'bb_update' in name ]
                if len(bb_grads)>0:
                    bb_grads_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in bb_grads]), 2.0)
                    self.writer.add_scalar(f'Grad/BB_update', bb_grads_norm, self.trained_steps)

                if model_ckpt_update:
                    info = (
                            f'best_trained_steps/epoch: {self.best_trained_steps}/{self.best_trained_epoch} | '
                            f'best_rmse_all: {self.best_rmse_all:.4f} | '
                            f'relat_rmse_ca: {self.best_rmse_ca:.4f} | '
                            # f'rot error:{self.best_rot_error:.4f}/{self.best_ref_rot_error:.4f} | '
                            # f'trans error:{self.best_trans_error:.4f}/{self.best_ref_trans_error:.4f} | '
                            f'relat_rmsd_ca_aligned: {self.best_rmsd_ca_aligned:.4f} | '
                            f'relat_drmsd: {self.best_drmsd:.4f}'
                        )
                    self.writer.add_text('Update Best Model',info, self.trained_steps)
        
                # Logging checkpoint metrics if available
                if ckpt_metrics is not None:
                    self.writer.add_scalar('Evaluation/Eval_time', eval_time, self.trained_steps)
                    # print(ckpt_metrics.columns)
                    for metric_name in ckpt_metrics.columns:
                        if metric_name == 'pdb_name':
                            continue
                        # print(ckpt_metrics[metric_name].mean())
                        self.writer.add_scalar(f'Evaluation/{metric_name}', ckpt_metrics[metric_name].mean(), self.trained_steps)
                    self.writer.add_scalar(f'Evaluation/{metric_name}', ckpt_metrics[metric_name].mean(), self.trained_steps)

                    # self.writer.add_scalars("RigidsError/rot", {'pred': rot_trans_error_mean['ave_rot'], 'ref': rot_trans_error_mean['first_rot']}, self.trained_steps)
                    # self.writer.add_scalars("RigidsError/trans", {'pred': rot_trans_error_mean['ave_trans'], 'ref': rot_trans_error_mean['first_trans']}, self.trained_steps)

                    self.writer.add_figure('dis_curve_plot', curve_fig, global_step=self.trained_steps)
                    self.writer.add_figure('dis_curve_plot_aligned', curve_fig_aligned, global_step=self.trained_steps)
                    # self.writer.add_figure('error_plot', error_fig, global_step=self.trained_steps)
                if torch.isnan(loss):
                    # TensorBoard does not have a direct alert system like WandB
                    self.writer.add_text("Alerts", f"Encountered NaN loss after {self.trained_epochs} epochs, {self.trained_steps} steps", self.trained_steps)
                    raise Exception('NaN encountered')
                


        if return_logs:
            return global_logs

    def eval_offline_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # initial some metrics and base save path
        # diff_first_metric_list=[]
        metric_list = []
        metric_all_list = []

        metric_aligned_list = []
        metric_aligned_all_list = []

        first_frame_all_list = []
        save_name_list = []
        rot_trans_error_dict = {'name':[],"ave_rot":[],"ave_trans":[],"first_rot":[],"first_trans":[],'time_rot_dif':[],'time_trans_dif':[]}

        sample_root_path = os.path.join(eval_dir,'sample')
        gt_root_path = os.path.join(eval_dir,'gt')

        sample_frame_root_path = os.path.join(eval_dir,'frame','sample')
        gt_frame_root_path = os.path.join(eval_dir,'frame','gt')

        prot_traj_root_path = os.path.join(eval_dir,'traj')
        rigids_path = os.path.join(eval_dir,'rigids')


        if not os.path.exists(rigids_path):
            os.makedirs(rigids_path,exist_ok=True)
        if not is_training:
            pred_numpy_path = os.path.join(eval_dir,'pred_npz')
            if not os.path.exists(pred_numpy_path):
                os.makedirs(pred_numpy_path,exist_ok=True)

        if not os.path.exists(prot_traj_root_path):
            os.makedirs(prot_traj_root_path,exist_ok=True)

        if not os.path.exists(sample_root_path):
            os.makedirs(sample_root_path,exist_ok=True)
        if not os.path.exists(gt_root_path):
            os.makedirs(gt_root_path,exist_ok=True)

        if not os.path.exists(sample_frame_root_path):
            os.makedirs(sample_frame_root_path,exist_ok=True)
        if not os.path.exists(gt_frame_root_path):
            os.makedirs(gt_frame_root_path,exist_ok=True)
        start_index_list = []
        # ergodic the validation                                                                                                                                                                  
        for valid_feats, pdb_names,start_index in valid_loader:                                                                                                                                   
            start_index_list.append(start_index)                                                                                                                                                  
            # print('motion_rigids_0',valid_feats['motion_rigids_0'].shape) motion_rigids_0 torch.Size([1, 3, 38, 7])                                                                             
            # initialize input data                                                                                                                                                               
            sample_length =  valid_feats['aatype'].shape[-1]                                                                                                                                      
            frame_time = self._model_conf.frame_time                                                                                                                                              
            ref_number = self._model_conf.ref_number                                                                                                                                              
            motion_number = self._model_conf.motion_number                                                                                                                                        

            res_mask = np.ones((frame_time,sample_length))                                                                                                                                        
            fixed_mask = np.zeros_like(res_mask)                                                                                                                                                  
            res_idx = torch.arange(1, sample_length+1).unsqueeze(0).repeat(frame_time,1)                                                                                                          
            ref_sample = self.diffuser.sample_ref(                                                                                                                                                
                n_samples=sample_length*frame_time,                                                                                                                                               
                as_tensor_7=True,                                                                                                                                                                 
            )                                                                                                                                                                                     
            ref_sample['rigids_t'] = ref_sample['rigids_t'].reshape([-1,frame_time,sample_length,7])                                                                                              


            ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)                                                                                                                   
            init_feats = {                                                                                                                                                                        
                'res_mask': res_mask[None],                                                                                                                                                       
                'seq_idx': res_idx[None],                                                                                                                                                         
                'fixed_mask': fixed_mask[None],                                                                                                                                                   
                'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],                                                                                                                  
                'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None],                                                                                                                         
                'node_repr':valid_feats['node_repr'],                                                                                                                                             
                'edge_repr':valid_feats['edge_repr'],                                                                                                                                             
                'aatype':valid_feats['aatype'],                                                                                                                                                   
                **ref_sample,                                                                                                                                                                     
                'rigids_0':valid_feats['rigids_0'], #TODO                                                                                                                                         
                'atom37_pos': valid_feats['atom37_pos'],                                                                                                                                          
                'atom37_mask': valid_feats['atom37_mask'],                                                                                                                                        
                # 'ref_rigids_0':valid_feats['ref_rigids_0'],                                                                                                                                     
                # 'motion_rigids_0':valid_feats['motion_rigids_0'],                                                                                                                               
                # 'ref_atom37_pos':valid_feats['ref_atom37_pos'],                                                                                                                                 
                # 'motion_atom37_pos':valid_feats['motion_atom37_pos'],                                                                                                                           
                # 'rigids_t': diff_rigids_t[None].to(device)  # rigids_t based on gt                                                                                                              
            }                                                                                                                                                                                     
                                                                                                                                                                                                  
            # TODO here                                                                                                                                                                           
            # fasta_aatype = du.move_to_np(valid_feats['aatype'])[0] # remove the batch(1,...) to (...),conver form [1,N] to [N,]                                                                 
            init_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)                                                                                   
            init_feats = tree.map_structure(lambda x: x.to(device), init_feats)                                                                                                                   
                                                                                                                                                                                                  
            # TODO flatten the dim of batch and frame_time                                                                                                                                        
            for k in init_feats.keys():                                                                                                                                                           
                v = init_feats[k]                                                                                                                                                                 
                if len(v.shape)>1:                                                                                                                                                                
                    reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维                                 
                    init_feats[k] = reshaped_tensor
    
            #offline_path = "/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/liuce/AlphaFlow/alphaflow_md_base_202402" 
            #offline_path = "/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/liuce/AlphaFlow/esmflow_md_base_202402"
            offline_path="/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/liuce/AlphaFlow/alphaflow_md_templates_base_202402"
            #print(pdb_names)
            #exit()
            sample_path = os.path.join(offline_path, pdb_names[0]+".pdb")
            traj = mdtraj.load_pdb(sample_path)
            #print(pdb.xyz.shape, pdb[0].xyz.shape)
            #exit()
            f, temp_path = tempfile.mkstemp(); os.close(f)
            positions_stacked = []
            for i in range(len(traj)):
                traj[i].save_pdb(temp_path)
                with open(temp_path) as f:
                    prot = protein.from_pdb_string(f.read())                                                                                                                                              
                    pdb_feats = make_protein_features(prot, pdb_names[0])                                                                                                 
                    positions_stacked.append(pdb_feats['all_atom_positions'])                                                                                                                             
            
            index_list = [] 
            for ls in range(0,250,3):
                index_list.append(ls)
            index_list = index_list[:64]                                                                                                                                                       
            traj = np.stack(positions_stacked)[index_list] 
            #print(traj.shape)
            sample_out = {}
            sample_out['prot_traj'] = traj[None]
            print(pdb_names, sample_out['prot_traj'].shape)
            # for k,v in sample_out.items():                                                                                                                                                      
            #     print(k,v.shape)                                                                                                                                                                
            # print('='*10)                                                                                                                                                                       
            ########################################################################                                                                                                              
            # generate aligned result, align to the reference(1st one)                                                                                                                            
            # may be useful in some case                                                                                                                                                          
            align_sample_list = []                                                                                                                                                                
            align_metric_list = []                                                                                                                                                                
            # align_sample_rigids_list = []                                                                                                                                                       
            # sample_rigids = sample_out['rigid_traj'][0]                                                                                                                                         

            # here just compute the rot and trans (1st frame vs reference) and apply to all frames                                                                                                
            for i in range(0,sample_out['prot_traj'][0].shape[0]): # 1,t,N,37,3                                                                                                                   
                    sup = SVDSuperimposer()                                                                                                                                                       
                    sup.set(sample_out['prot_traj'][0][0][:,1], sample_out['prot_traj'][0][i][:,1])                                                                                               
                    # align to the reference with C-alpha [:,1], 1 is the index of CA in atom37                                                                                                   
                    sup.run()                                                                                                                                                                     
                    rot,trans = sup.get_rotran() # with shape rot=(3, 3) trans=(3,)                                                                                                               
                    align_metric_list.append((rot,trans))                                                                                                                                         
                    tmp = np.dot(sample_out['prot_traj'][0][i],rot)+trans # apply the rotation and translation                                                                                    
                    tmp = torch.from_numpy(tmp) * valid_feats['atom37_mask'][0][i][...,None] # apply the atom_mask                                                                                
                    align_sample_list.append(tmp)                                                                                                                                                 

            align_sample = torch.stack(align_sample_list)                                                                                                                                         
            ########################################################################                                                                                                              
            # save the prediction                                                                                                                                                                 
            save_name = pdb_names[0].split('.')[0]                                                                                                                                                
            sample_path = os.path.join(sample_root_path, f'{save_name}.pdb')                                                                                                                      
            gt_path = os.path.join(gt_root_path, f'{save_name}_gt.pdb')                                                                                                                           
            sample_aligned_path = os.path.join(sample_root_path, f'{save_name}_aligned.pdb')                                                                                                      
            prot_traj_path = os.path.join(prot_traj_root_path, f'{save_name}_traj.pdb')                                                                                                           
            ########################################################################                                                                                                              
            diffuse_mask = np.ones(sample_length)                                                                                                                                                 
            b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))      

            eval_dic = compute_validation_metrics_all(                                                                                                                                            
                                    gt_pos=valid_feats['atom37_pos'][0], # [0] for the batchsize=1                                                                                                
                                    out_pos=sample_out['prot_traj'][0],                                                                                                                           
                                    gt_mask=valid_feats['atom37_mask'][0],                                                                                                                        
                                    superimposition_metrics=True)                                                                                                                                 
            mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}                                                                                                                     
            # aligned prediciton as results                                                                                                                                                       
            align_eval_dic = compute_validation_metrics_all(                                                                                                                                      
                                        gt_pos=valid_feats['atom37_pos'][0],                                                                                                                      
                                       out_pos=align_sample.cpu().numpy(),                                                                                                                        
                                       gt_mask=valid_feats['atom37_mask'][0],                                                                                                                     
                                       superimposition_metrics=True)                                                                                                                              
            # align_mean_eval_dic = {k: sum(v[1:]) / len(v[1:]) for k, v in align_eval_dic.items()}                                                                                               
            align_mean_eval_dic = {k: sum(v) / len(v) for k, v in align_eval_dic.items()}                                                                                                         
            ########################################################################                                                                                                              
            # save npz format prediction                                                                                                                                                          
            if not is_training:                                                                                                                                                                   
                np.savez(os.path.join(pred_numpy_path,f'{save_name}.npz'),                                                                                                                        
                        gt_37=valid_feats['atom37_pos'][0].cpu().numpy(),                                                                                                                         
                        pred_37 = sample_out['prot_traj'][0],                                                                                                                                     
                        aligned_pred_37=align_sample.cpu().numpy(),                                                                                                                               
                        gt_mask = valid_feats['atom37_mask'][0].cpu().numpy(),                                                                                                                    
                        aatype = valid_feats['aatype'][0].cpu().numpy(),                                                                                                                          
                        b_factors = b_factors                                                                                                                                                     
                        )                                                                                                                                                                         
                                                                                                                                                                                                  
                                                                                                                                                                                                  
            save_name_list.append(save_name)                                                                                                                                                      
            ########################################################################                                                                                                              
            # metric for un-aligned prediciton                                                                                                                                                    
            metric_list.append({k: mean_eval_dic[k].cpu().numpy() if torch.is_tensor(mean_eval_dic[k]) else mean_eval_dic[k] for k in mean_eval_dic})                                             
            metric_all_list.append({k: eval_dic[k].cpu().numpy() if torch.is_tensor(eval_dic[k]) else eval_dic[k] for k in eval_dic})                                                             
                                                                                                                                                                                                  
            # metric for aligned prediciton                                                                                                                                                       
            metric_aligned_list.append({k: align_mean_eval_dic[k].cpu().numpy() if torch.is_tensor(align_mean_eval_dic[k]) else align_mean_eval_dic[k] for k in align_mean_eval_dic})             
            metric_aligned_all_list.append({k: align_eval_dic[k].cpu().numpy() if torch.is_tensor(align_eval_dic[k]) else align_eval_dic[k] for k in align_eval_dic})

            all_atom37_pos = valid_feats['atom37_pos'][0].cpu().numpy() # gt position                                                                                                             
            _ = au.write_prot_to_pdb(                                                                                                                                                             
                prot_pos=all_atom37_pos,                                                                                                                                                          
                file_path=gt_path,                                                                                                                                                                
                aatype=init_feats['aatype'][0].cpu().numpy(),                                                                                                                                     
                no_indexing=True,                                                                                                                                                                 
                b_factors=b_factors
            )
            # save aligned prediction                                                                                                                                                             
            # align_sample.cpu().numpy()[0] = all_atom37_pos[0]                                                                                                                                   
            _ = au.write_prot_to_pdb(
                prot_pos=align_sample.cpu().numpy(),                                                                                                                                              
                file_path=sample_aligned_path,
                aatype=init_feats['aatype'][0].cpu().numpy(),
                no_indexing=True,
                b_factors=b_factors                                                                                                                                                               
            )                                                                                                                                                                                     
            # save un-aligned prediction                                                                                                                                                          
            _ = au.write_prot_to_pdb(                                                                                                                                                             
                prot_pos=sample_out['prot_traj'][0],
                file_path=sample_path,                                                                                                                                                            
                aatype=init_feats['aatype'][0].cpu().numpy(),                                                                                                                                     
                no_indexing=True,                                                                                                                                                                 
                b_factors=b_factors                                                                                                                                                               
            )                                                                                                                                                                                     
            # _ = au.write_prot_to_pdb(                                                                                                                                                           
            #     prot_pos=np.concatenate(( valid_feats['motion_atom37_pos'][0].cpu().numpy(),valid_feats['ref_atom37_pos'][0].cpu().numpy()), axis=0),                                           
            #     file_path=os.path.join(sample_root_path, f'{save_name}_first_motion.pdb'),                                                                                                      
            #     aatype=init_feats['aatype'][0].cpu().numpy(),                                                                                                                                   
            #     no_indexing=True,                                                                                                                                                               
            #     b_factors=b_factors                                                                                                                                                             
            # )                                                                                                                                                                                   

            if not is_training:                                                                                                                                                                   
                # save un-aligned prediction                                                                                                                                                      
                # _ = au.write_prot_to_pdb(                                                                                                                                                       
                #     prot_pos=valid_feats['ref_atom37_pos'][0].cpu().numpy(),                                                                                                                    
                #     file_path=os.path.join(sample_root_path, f'{save_name}_first.pdb'),                                                                                                         
                #     aatype=init_feats['aatype'][0].cpu().numpy(),
                #     no_indexing=True,                                                                                                                                                           
                #     b_factors=b_factors                                                                                                                                                         
                # )                                                                                                                                                                               
                # for i in range(frame_time):                                                                                                                                                     
                    # prot_traj_path = os.path.join(prot_traj_root_path, f'{save_name}_traj_frame_{i}.pdb')                                                                                       
                    # _ = au.write_prot_to_pdb(                                                                                                                                                   
                    #     prot_pos=sample_out['prot_traj'][:,i],                                                                                                                                  
                    #     file_path=prot_traj_path,                                                                                                                                               
                    #     aatype=init_feats['aatype'][0].cpu().numpy(),                                                                                                                           
                    #     no_indexing=True,                                                                                                                                                       
                    #     b_factors=b_factors                                                                                                                                                     
                    # )                                                                                                                                                                           
                diff_time = sample_out['prot_traj'].shape[0] 

                for i in range(diff_time):
                    if not os.path.exists(os.path.join(prot_traj_root_path, f'{save_name}')):
                        os.makedirs(os.path.join(prot_traj_root_path, f'{save_name}'))
                    prot_traj_path = os.path.join(prot_traj_root_path, f'{save_name}',f'traj_frame_{i}.pdb')
                    _ = au.write_prot_to_pdb(
                        prot_pos=sample_out['prot_traj'][i],                                                                                                                                      
                        file_path=prot_traj_path,                                                                                                                                                 
                        aatype=init_feats['aatype'][0].cpu().numpy(),                                                                                                                             
                        no_indexing=True,                                                                                                                                                         
                        b_factors=b_factors                                                                                                                                                       
                    )                                                                                                                                                                             
        
        # un-aligned prediciton metrics                                                                                                                                                           
        ckpt_eval_metrics_all = pd.DataFrame(metric_all_list)                                                                                                                                     
        ckpt_eval_metrics_all.insert(0,'pdb_name',save_name_list)                                                                                                                                 
        ckpt_eval_metrics_all.insert(1,'start_index',start_index_list)                                                                                                                            
        ckpt_eval_metrics_all.to_csv(os.path.join(eval_dir, 'metrics_wo_mean.csv'), index=False)                                                                                                  
        # aligned prediciton metrics                                                                                                                                                              
        ckpt_eval_metrics_all_aligned = pd.DataFrame(metric_aligned_all_list)                                                                                                                     
        ckpt_eval_metrics_all_aligned.insert(0,'pdb_name',save_name_list)                                                                                                                         
        ckpt_eval_metrics_all_aligned.to_csv(os.path.join(eval_dir, 'metrics_wo_mean_aligned.csv'), index=False)                                                                                  
        # reference prediciton metrics                                                                                                                                                            
        # first_frame_eval_metrics_all = pd.DataFrame(first_frame_all_list)                                                                                                                       
        # first_frame_eval_metrics_all.insert(0,'pdb_name',save_name_list)                                                                                                                        
        # first_frame_eval_metrics_all.to_csv(os.path.join(eval_dir, 'metrics_first_frame.csv'), index=False)                                                                                     
        # compare to ref                                                                                                                                                                          
        # compare_to_first_frame_eval_metrics_all = pd.DataFrame(diff_first_metric_list)                                                                                                          
        # compare_to_first_frame_eval_metrics_all.insert(0,'pdb_name',save_name_list)                                                                                                             
        # compare_to_first_frame_eval_metrics_all.to_csv(os.path.join(eval_dir, 'metrics_first_frame.csv'), index=False)                                                                          

        test_num = len(valid_loader)                                                                                                                                                              
        rows = 3                                                                                                                                                                                  
        cols = test_num                                                                                                                                                                           

        metric_merge_dict = {'Pred':ckpt_eval_metrics_all}                                                                                                                                        
        curve_fig = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols)                                                                                                       

        metric_merge_dict = {'Pred':ckpt_eval_metrics_all_aligned}                                                                                                                                
        curve_fig_aligned = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols,suffer_fix='aligned')                                                                          


        # metric_merge_dict = {'Compare2Ref':compare_to_first_frame_eval_metrics_all}                                                                                                             
        # curve_fig_aligned = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols,suffer_fix='compare_to_first')                                                               
        # print(rot_trans_error_dict)                                                                                                                                                             
        # error_fig = plot_rot_trans_curve(rot_trans_error_dict,save_path=eval_dir,frame_step=self._data_conf.frame_sample_step)                                                                  


        # un-aligned prediciton metric                                                                                                                                                            
        ckpt_eval_metrics = pd.DataFrame(metric_list)                                                                                                                                             
        ckpt_eval_metrics.insert(0,'pdb_name',save_name_list)                                                                                                                                     
        ckpt_eval_metrics.to_csv(os.path.join(eval_dir, 'metrics.csv'), index=False)                                                                                                              
        # aligned prediciton metric                                                                                                                                                               
        ckpt_eval_metrics_aligned = pd.DataFrame(metric_aligned_list)                                                                                                                             
        ckpt_eval_metrics_aligned.insert(0,'pdb_name',save_name_list)                                                                                                                             
        ckpt_eval_metrics_aligned.to_csv(os.path.join(eval_dir, 'metrics_aligned.csv'), index=False)
         


        test_num = len(valid_loader)                                                                                                                                                              
        rows = 3                                                                                                                                                                                  
        cols = test_num                                                                                                                                                                           

        metric_merge_dict = {'Pred':ckpt_eval_metrics_all}                                                                                                                                        
        curve_fig = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols)                                                                                                       

        metric_merge_dict = {'Pred':ckpt_eval_metrics_all_aligned}                                                                                                                                
        curve_fig_aligned = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols,suffer_fix='aligned')                                                                          


        # metric_merge_dict = {'Compare2Ref':compare_to_first_frame_eval_metrics_all}                                                                                                             
        # curve_fig_aligned = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols,suffer_fix='compare_to_first')                                                               
        # print(rot_trans_error_dict)                                                                                                                                                             
        # error_fig = plot_rot_trans_curve(rot_trans_error_dict,save_path=eval_dir,frame_step=self._data_conf.frame_sample_step)                                                                  


        # un-aligned prediciton metric                                                                                                                                                            
        ckpt_eval_metrics = pd.DataFrame(metric_list)                                                                                                                                             
        ckpt_eval_metrics.insert(0,'pdb_name',save_name_list)                                                                                                                                     
        ckpt_eval_metrics.to_csv(os.path.join(eval_dir, 'metrics.csv'), index=False)                                                                                                              
        # aligned prediciton metric                                                                                                                                                               
        ckpt_eval_metrics_aligned = pd.DataFrame(metric_aligned_list)                                                                                                                             
        ckpt_eval_metrics_aligned.insert(0,'pdb_name',save_name_list)                                                                                                                             
        ckpt_eval_metrics_aligned.to_csv(os.path.join(eval_dir, 'metrics_aligned.csv'), index=False)                                                                                              

        # use aligned prediciton metric to save the best model                                                                                                                                    
        mean_dict = ckpt_eval_metrics.mean()                                                                                                                                                      
        mean_dict = mean_dict.to_dict()                                                                                                                                                           
        info = f'Step:{self.trained_steps} '                                                                                                                                                      
        for k in mean_dict.keys():                                                                                                                                                                
            tmp = f'avg_{k}:{mean_dict[k]:.4f} '                                                                                                                                                  
            info+=tmp                                                                                                                                                                             
        # for k in rot_trans_error_mean.keys():                                                                                                                                                   
        #     if k != 'name':                                                                                                                                                                     
        #         tmp = f'avg_{k}:{rot_trans_error_mean[k]:.4f} '                                                                                                                                 
        #         info+=tmp                                                                                                                                                                       
        # if mean_dict['alignment_rmsd'] < self.bset_rmsd_ca:                                                                                                                                     
        model_ckpt_update=False                                                                                                                                                                   
        if mean_dict['rmse_all'] < self.best_rmse_all :# or rot_trans_error_mean['ave_rot']<self.best_rot_error or rot_trans_error_mean['ave_trans']<self.best_trans_error:                       
            self.best_rmse_all = mean_dict['rmse_all']                                                                                                                                            
            self.best_rmse_ca = mean_dict['rmse_ca']                                                                                                                                              
            self.best_drmsd = mean_dict['drmsd_ca']                                                                                                                                               
            self.best_rmsd_ca_aligned = mean_dict['rmsd_ca_aligned']                                                                                                                              
            # self.best_rot_error = rot_trans_error_mean['ave_rot']                                                                                                                               
            # self.best_trans_error = rot_trans_error_mean['ave_trans']                                                                                                                           
            # self.best_ref_rot_error = rot_trans_error_mean['first_rot']                                                                                                                         
            # self.best_ref_trans_error = rot_trans_error_mean['first_trans']                                                                                                                     

            self.best_trained_steps = self.trained_steps                                                                                                                                          
            self.best_trained_epoch = self.trained_epochs                                                                                                                                         
            model_ckpt_update=True                                                                                                                                                                
            # self._log.info('Evaluation Res:'+info+f'full_rmsd:{total_rmsd}')                                                                                                                    
        self._log.info('Evaluation Res:'+info) 
        self._log.info(                                                                                                                                                                           
                f'best_trained_steps/epoch: {self.best_trained_steps}/{self.best_trained_epoch} | '                                                                                               
                f'best_rmse_all: {self.best_rmse_all:.4f} | '                                                                                                                                     
                f'relat_rmse_ca: {self.best_rmse_ca:.4f} | '                                                                                                                                      
                f'rot error:{self.best_rot_error:.4f}/{self.best_ref_rot_error:.4f} | '                                                                                                           
                f'trans error:{self.best_trans_error:.4f}/{self.best_ref_trans_error:.4f} | '                                                                                                     
                f'relat_rmsd_ca_aligned: {self.best_rmsd_ca_aligned:.4f} | '                                                                                                                      
                f'relat_drmsd: {self.best_drmsd:.4f}'                                                                                                                                             
            )                                                                                                                                                                                     

        # should reture eval_dict                                                                                                                                                                 
        return ckpt_eval_metrics,curve_fig,curve_fig_aligned,model_ckpt_update#,rot_trans_error_mean   

    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # initial some metrics and base save path
        # diff_first_metric_list=[]
        metric_list = []
        metric_all_list = []

        metric_aligned_list = []
        metric_aligned_all_list = []

        first_frame_all_list = []
        save_name_list = []
        rot_trans_error_dict = {'name':[],"ave_rot":[],"ave_trans":[],"first_rot":[],"first_trans":[],'time_rot_dif':[],'time_trans_dif':[]}

        sample_root_path = os.path.join(eval_dir,'sample')
        gt_root_path = os.path.join(eval_dir,'gt')

        sample_frame_root_path = os.path.join(eval_dir,'frame','sample')
        gt_frame_root_path = os.path.join(eval_dir,'frame','gt')

        prot_traj_root_path = os.path.join(eval_dir,'traj')
        rigids_path = os.path.join(eval_dir,'rigids')


        if not os.path.exists(rigids_path):
            os.makedirs(rigids_path,exist_ok=True)
        if not is_training:
            pred_numpy_path = os.path.join(eval_dir,'pred_npz')
            if not os.path.exists(pred_numpy_path):
                os.makedirs(pred_numpy_path,exist_ok=True)
          
        if not os.path.exists(prot_traj_root_path):
            os.makedirs(prot_traj_root_path,exist_ok=True)

        if not os.path.exists(sample_root_path):
            os.makedirs(sample_root_path,exist_ok=True)
        if not os.path.exists(gt_root_path):
            os.makedirs(gt_root_path,exist_ok=True) 

        if not os.path.exists(sample_frame_root_path):
            os.makedirs(sample_frame_root_path,exist_ok=True)
        if not os.path.exists(gt_frame_root_path):
            os.makedirs(gt_frame_root_path,exist_ok=True) 
        start_index_list = []
        # ergodic the validation
        for valid_feats, pdb_names,start_index in valid_loader:
            start_index_list.append(start_index)
            # print('motion_rigids_0',valid_feats['motion_rigids_0'].shape) motion_rigids_0 torch.Size([1, 3, 38, 7])
            # initialize input data
            sample_length =  valid_feats['aatype'].shape[-1]
            frame_time = self._model_conf.frame_time
            ref_number = self._model_conf.ref_number
            motion_number = self._model_conf.motion_number

            res_mask = np.ones((frame_time,sample_length))
            fixed_mask = np.zeros_like(res_mask)
            res_idx = torch.arange(1, sample_length+1).unsqueeze(0).repeat(frame_time,1)
            ref_sample = self.diffuser.sample_ref(
                n_samples=sample_length*frame_time,
                as_tensor_7=True,
            )
            ref_sample['rigids_t'] = ref_sample['rigids_t'].reshape([-1,frame_time,sample_length,7])


            ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)
            init_feats = {
                'res_mask': res_mask[None],
                'seq_idx': res_idx[None],
                'fixed_mask': fixed_mask[None],
                'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],
                'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None],
                'node_repr':valid_feats['node_repr'],
                'edge_repr':valid_feats['edge_repr'],
                'aatype':valid_feats['aatype'],
                **ref_sample,
                'rigids_0':valid_feats['rigids_0'], #TODO
                'atom37_pos': valid_feats['atom37_pos'],
                'atom37_mask': valid_feats['atom37_mask'],
                'ref_rigids_0':valid_feats['ref_rigids_0'],
                # 'motion_rigids_0':valid_feats['motion_rigids_0'],
                'ref_atom37_pos':valid_feats['ref_atom37_pos'],
                # 'motion_atom37_pos':valid_feats['motion_atom37_pos'],
                # 'rigids_t': diff_rigids_t[None].to(device)  # rigids_t based on gt
            }

            # TODO here
            # fasta_aatype = du.move_to_np(valid_feats['aatype'])[0] # remove the batch(1,...) to (...),conver form [1,N] to [N,]
            init_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
            init_feats = tree.map_structure(lambda x: x.to(device), init_feats)
            
            # TODO flatten the dim of batch and frame_time
            for k in init_feats.keys():
                v = init_feats[k]
                if len(v.shape)>1:
                    reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维
                    init_feats[k] = reshaped_tensor
            # for k,v in init_feats.items():
            #     print(k,v[0])
            # exit()
            # rigids_t
            ########################################################################
            # start inference
            # start_time = time.time()
            sample_out = self.inference_fn(
                init_feats,
                num_t=num_t,
                min_t=min_t, 
                aux_traj=True,
                noise_scale=noise_scale,
            )
            # for k,v in sample_out.items():
            #     print(k,v.shape)
            # print('='*10)
            ########################################################################
            # generate aligned result, align to the reference(1st one)
            # may be useful in some case
            align_sample_list = []
            align_metric_list = []
            # align_sample_rigids_list = []
            # sample_rigids = sample_out['rigid_traj'][0]

            # here just compute the rot and trans (1st frame vs reference) and apply to all frames
            for i in range(0,sample_out['prot_traj'][0].shape[0]): # 1,t,N,37,3
                    sup = SVDSuperimposer()
                    sup.set(sample_out['prot_traj'][0][0][:,1], sample_out['prot_traj'][0][i][:,1])  
                    # align to the reference with C-alpha [:,1], 1 is the index of CA in atom37
                    sup.run()
                    rot,trans = sup.get_rotran() # with shape rot=(3, 3) trans=(3,)
                    align_metric_list.append((rot,trans))
                    tmp = np.dot(sample_out['prot_traj'][0][i],rot)+trans # apply the rotation and translation
                    tmp = torch.from_numpy(tmp) * valid_feats['atom37_mask'][0][i][...,None] # apply the atom_mask
                    align_sample_list.append(tmp)

            align_sample = torch.stack(align_sample_list)
            ########################################################################
            # save the prediction
            save_name = pdb_names[0].split('.')[0]
            sample_path = os.path.join(sample_root_path, f'{save_name}.pdb')
            gt_path = os.path.join(gt_root_path, f'{save_name}_gt.pdb')
            sample_aligned_path = os.path.join(sample_root_path, f'{save_name}_aligned.pdb')
            prot_traj_path = os.path.join(prot_traj_root_path, f'{save_name}_traj.pdb')
            ########################################################################
            diffuse_mask = np.ones(sample_length)
            b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))
            
            # reference as prediciton
            # fake_res = torch.stack([valid_feats['ref_atom37_pos'][0,-1]]*frame_time)

            # first_eval_dic = compute_validation_metrics_all(
            #                         gt_pos=valid_feats['atom37_pos'][0],
            #                         out_pos=fake_res.cpu().numpy(),
            #                         gt_mask=valid_feats['atom37_mask'][0],
            #                         superimposition_metrics=True)
            # prediction metrics
            print(pdb_names)
            eval_dic = compute_validation_metrics_all(
                                    gt_pos=valid_feats['atom37_pos'][0], # [0] for the batchsize=1
                                    out_pos=sample_out['prot_traj'][0],
                                    gt_mask=valid_feats['atom37_mask'][0],
                                    superimposition_metrics=True)
            mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}
            # aligned prediciton as results 
            '''
            align_eval_dic = compute_validation_metrics_all(
                                        gt_pos=valid_feats['atom37_pos'][0],
                                       out_pos=align_sample.cpu().numpy(),
                                       gt_mask=valid_feats['atom37_mask'][0],
                                       superimposition_metrics=True)
            # align_mean_eval_dic = {k: sum(v[1:]) / len(v[1:]) for k, v in align_eval_dic.items()}
            '''
            align_eval_dic = copy.deepcopy(eval_dic)

            align_mean_eval_dic = {k: sum(v) / len(v) for k, v in align_eval_dic.items()}
            ########################################################################
            # save npz format prediction
            if not is_training:
                np.savez(os.path.join(pred_numpy_path,f'{save_name}.npz'),
                        gt_37=valid_feats['atom37_pos'][0].cpu().numpy(),
                        pred_37 = sample_out['prot_traj'][0],
                        aligned_pred_37=align_sample.cpu().numpy(),
                        gt_mask = valid_feats['atom37_mask'][0].cpu().numpy(),
                        aatype = valid_feats['aatype'][0].cpu().numpy(),
                        b_factors = b_factors
                        )
                

            save_name_list.append(save_name)
            ########################################################################
            # metric for un-aligned prediciton
            metric_list.append({k: mean_eval_dic[k].cpu().numpy() if torch.is_tensor(mean_eval_dic[k]) else mean_eval_dic[k] for k in mean_eval_dic})
            metric_all_list.append({k: eval_dic[k].cpu().numpy() if torch.is_tensor(eval_dic[k]) else eval_dic[k] for k in eval_dic})

            # metric for aligned prediciton
            metric_aligned_list.append({k: align_mean_eval_dic[k].cpu().numpy() if torch.is_tensor(align_mean_eval_dic[k]) else align_mean_eval_dic[k] for k in align_mean_eval_dic})
            metric_aligned_all_list.append({k: align_eval_dic[k].cpu().numpy() if torch.is_tensor(align_eval_dic[k]) else align_eval_dic[k] for k in align_eval_dic})
            
            # metric for reference
            # first_frame_all_list.append({k: first_eval_dic[k].cpu().numpy() if torch.is_tensor(first_eval_dic[k]) else first_eval_dic[k] for k in first_eval_dic})

            ########################################################################


            # metric compare to ref
            # diff_first_metric_list.append({k: compare_to_first_dict[k].cpu().numpy() if torch.is_tensor(compare_to_first_dict[k]) else compare_to_first_dict[k] for k in compare_to_first_dict})
            
            # if not is_training:
            #     np.savez(os.path.join(rigids_path,f'{save_name}_rigds.npz'),
            #              pred=init_feats['rigids_0'].cpu().numpy(),
            #              gt=sample_out['rigid_traj'][0])
            # print(sample_out['rigid_traj'][0],init_feats['rigids_0'].cpu().numpy())
            # np.savez('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/DFOLDv2/debug_tool/1frame_pred_gt_rigids.npz',
            #          gt_rigids=sample_out['rigid_traj'][0],
            #          pred_rigids=init_feats['rigids_0'].cpu().numpy())
            # caculate the rotation and translation error
            # tmp_ref_rigids = valid_feats['ref_rigids_0'][0,-1].cpu().numpy()
            # ave_quat,ave_trans,ref_ave_quat,ref_ave_trans,time_rot_dif,time_trans_dif = self._calc_rot_trans_error(sample_out['rigid_traj'][0],
            #                                                                                                        gt_rigids=init_feats['rigids_0'].cpu().numpy(),
            #                                                                                                        ref_rigids=tmp_ref_rigids
            #                                                                                                        )
            # rot_trans_error_dict['name'].append(save_name) # pdb name
            # rot_trans_error_dict['ave_rot'].append(ave_quat) # average rotation error (degree)
            # rot_trans_error_dict['ave_trans'].append(ave_trans) # average translation error MSE
            # rot_trans_error_dict['first_rot'].append(ref_ave_quat)
            # rot_trans_error_dict['first_trans'].append(ref_ave_trans)
            # rot_trans_error_dict['time_rot_dif'].append(time_rot_dif)
            # rot_trans_error_dict['time_trans_dif'].append(time_trans_dif)

            # Calculating the mean of each list excluding the first element
            # if self._conf.data.frame_time>1:
            #     rot_trans_error_mean = {key: [np.mean(arr[1:]) for arr in values]
            #                 for key, values in rot_trans_error_dict.items() if key != 'name'}
            # else:

            # compute for each protein(traj)
            # rot_trans_error_mean = {key: [np.mean(arr) for arr in values] for key, values in rot_trans_error_dict.items() if key != 'name'} 
            # rot_trans_error_mean = {
            #     key: [np.mean(arr[1:]) if key in ['first_rot', 'first_trans'] else np.mean(arr) for arr in values] # excluding the 1st frame for reference metric 
            #     for key, values in rot_trans_error_dict.items() if key != 'name'
            # }
            # average the all
            # rot_trans_error_mean = {key: sum(values) / len(values) for key, values in rot_trans_error_mean.items() if key != 'name'}
            # save ground truth
            all_atom37_pos = valid_feats['atom37_pos'][0].cpu().numpy() # gt position
            _ = au.write_prot_to_pdb(
                prot_pos=all_atom37_pos,
                file_path=gt_path,
                aatype=init_feats['aatype'][0].cpu().numpy(),
                no_indexing=True,
                b_factors=b_factors
            )
            # save aligned prediction
            # align_sample.cpu().numpy()[0] = all_atom37_pos[0]
            _ = au.write_prot_to_pdb(
                prot_pos=align_sample.cpu().numpy(),
                file_path=sample_aligned_path,
                aatype=init_feats['aatype'][0].cpu().numpy(),
                no_indexing=True,
                b_factors=b_factors
            )   
            # save un-aligned prediction
            _ = au.write_prot_to_pdb(
                prot_pos=sample_out['prot_traj'][0],
                file_path=sample_path,
                aatype=init_feats['aatype'][0].cpu().numpy(),
                no_indexing=True,
                b_factors=b_factors
            )
            # _ = au.write_prot_to_pdb(
            #     prot_pos=np.concatenate(( valid_feats['motion_atom37_pos'][0].cpu().numpy(),valid_feats['ref_atom37_pos'][0].cpu().numpy()), axis=0),
            #     file_path=os.path.join(sample_root_path, f'{save_name}_first_motion.pdb'),
            #     aatype=init_feats['aatype'][0].cpu().numpy(),
            #     no_indexing=True,
            #     b_factors=b_factors
            # )

            if not is_training:
                # save un-aligned prediction
                # _ = au.write_prot_to_pdb(
                #     prot_pos=valid_feats['ref_atom37_pos'][0].cpu().numpy(),
                #     file_path=os.path.join(sample_root_path, f'{save_name}_first.pdb'),
                #     aatype=init_feats['aatype'][0].cpu().numpy(),
                #     no_indexing=True,
                #     b_factors=b_factors
                # )
                # for i in range(frame_time):
                    # prot_traj_path = os.path.join(prot_traj_root_path, f'{save_name}_traj_frame_{i}.pdb')
                    # _ = au.write_prot_to_pdb(
                    #     prot_pos=sample_out['prot_traj'][:,i],
                    #     file_path=prot_traj_path,
                    #     aatype=init_feats['aatype'][0].cpu().numpy(),
                    #     no_indexing=True,
                    #     b_factors=b_factors
                    # )
                diff_time = sample_out['prot_traj'].shape[0]
                for i in range(diff_time):
                    if not os.path.exists(os.path.join(prot_traj_root_path, f'{save_name}')):
                        os.makedirs(os.path.join(prot_traj_root_path, f'{save_name}'))
                    prot_traj_path = os.path.join(prot_traj_root_path, f'{save_name}',f'traj_frame_{i}.pdb')
                    _ = au.write_prot_to_pdb(
                        prot_pos=sample_out['prot_traj'][i],
                        file_path=prot_traj_path,
                        aatype=init_feats['aatype'][0].cpu().numpy(),
                        no_indexing=True,
                        b_factors=b_factors
                    )
        
        # un-aligned prediciton metrics
        ckpt_eval_metrics_all = pd.DataFrame(metric_all_list)
        ckpt_eval_metrics_all.insert(0,'pdb_name',save_name_list)
        ckpt_eval_metrics_all.insert(1,'start_index',start_index_list)
        ckpt_eval_metrics_all.to_csv(os.path.join(eval_dir, 'metrics_wo_mean.csv'), index=False)
        # aligned prediciton metrics
        ckpt_eval_metrics_all_aligned = pd.DataFrame(metric_aligned_all_list)
        ckpt_eval_metrics_all_aligned.insert(0,'pdb_name',save_name_list)
        ckpt_eval_metrics_all_aligned.to_csv(os.path.join(eval_dir, 'metrics_wo_mean_aligned.csv'), index=False)
        # reference prediciton metrics
        # first_frame_eval_metrics_all = pd.DataFrame(first_frame_all_list)
        # first_frame_eval_metrics_all.insert(0,'pdb_name',save_name_list)
        # first_frame_eval_metrics_all.to_csv(os.path.join(eval_dir, 'metrics_first_frame.csv'), index=False)
        # compare to ref
        # compare_to_first_frame_eval_metrics_all = pd.DataFrame(diff_first_metric_list)
        # compare_to_first_frame_eval_metrics_all.insert(0,'pdb_name',save_name_list)
        # compare_to_first_frame_eval_metrics_all.to_csv(os.path.join(eval_dir, 'metrics_first_frame.csv'), index=False)       

        test_num = len(valid_loader)
        rows = 3
        cols = test_num

        metric_merge_dict = {'Pred':ckpt_eval_metrics_all}
        curve_fig = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols)

        metric_merge_dict = {'Pred':ckpt_eval_metrics_all_aligned}
        curve_fig_aligned = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols,suffer_fix='aligned')


        # metric_merge_dict = {'Compare2Ref':compare_to_first_frame_eval_metrics_all}
        # curve_fig_aligned = plot_curve_merged(metric_merge_dict,eval_dir,row_num=rows,col_num=cols,suffer_fix='compare_to_first')
        # print(rot_trans_error_dict)
        # error_fig = plot_rot_trans_curve(rot_trans_error_dict,save_path=eval_dir,frame_step=self._data_conf.frame_sample_step)


        # un-aligned prediciton metric 
        ckpt_eval_metrics = pd.DataFrame(metric_list)
        ckpt_eval_metrics.insert(0,'pdb_name',save_name_list)
        ckpt_eval_metrics.to_csv(os.path.join(eval_dir, 'metrics.csv'), index=False)
        # aligned prediciton metric 
        ckpt_eval_metrics_aligned = pd.DataFrame(metric_aligned_list)
        ckpt_eval_metrics_aligned.insert(0,'pdb_name',save_name_list)
        ckpt_eval_metrics_aligned.to_csv(os.path.join(eval_dir, 'metrics_aligned.csv'), index=False)

        # use aligned prediciton metric to save the best model
        mean_dict = ckpt_eval_metrics.mean()
        mean_dict = mean_dict.to_dict()
        info = f'Step:{self.trained_steps} '
        for k in mean_dict.keys():
            tmp = f'avg_{k}:{mean_dict[k]:.4f} '
            info+=tmp
        # for k in rot_trans_error_mean.keys():
        #     if k != 'name':
        #         tmp = f'avg_{k}:{rot_trans_error_mean[k]:.4f} '
        #         info+=tmp
        # if mean_dict['alignment_rmsd'] < self.bset_rmsd_ca:
        model_ckpt_update=False 
        if mean_dict['rmse_all'] < self.best_rmse_all :# or rot_trans_error_mean['ave_rot']<self.best_rot_error or rot_trans_error_mean['ave_trans']<self.best_trans_error:
            self.best_rmse_all = mean_dict['rmse_all']
            self.best_rmse_ca = mean_dict['rmse_ca']
            self.best_drmsd = mean_dict['drmsd_ca']
            self.best_rmsd_ca_aligned = mean_dict['rmsd_ca_aligned']
            # self.best_rot_error = rot_trans_error_mean['ave_rot']
            # self.best_trans_error = rot_trans_error_mean['ave_trans']
            # self.best_ref_rot_error = rot_trans_error_mean['first_rot']
            # self.best_ref_trans_error = rot_trans_error_mean['first_trans']

            self.best_trained_steps = self.trained_steps
            self.best_trained_epoch = self.trained_epochs
            model_ckpt_update=True
            # self._log.info('Evaluation Res:'+info+f'full_rmsd:{total_rmsd}')
        self._log.info('Evaluation Res:'+info)
        self._log.info(
                f'best_trained_steps/epoch: {self.best_trained_steps}/{self.best_trained_epoch} | '
                f'best_rmse_all: {self.best_rmse_all:.4f} | '
                f'relat_rmse_ca: {self.best_rmse_ca:.4f} | '
                f'rot error:{self.best_rot_error:.4f}/{self.best_ref_rot_error:.4f} | '
                f'trans error:{self.best_trans_error:.4f}/{self.best_ref_trans_error:.4f} | '
                f'relat_rmsd_ca_aligned: {self.best_rmsd_ca_aligned:.4f} | '
                f'relat_drmsd: {self.best_drmsd:.4f}'
            )

        # should reture eval_dict
        return ckpt_eval_metrics,curve_fig,curve_fig_aligned,model_ckpt_update#,rot_trans_error_mean


    def eval_extension(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # ergodic the validation
        length = self._conf.eval.extrapolation_time
        npz_base_path = os.path.join(eval_dir,'extension_npz')
        pdb_base_path = os.path.join(eval_dir,'extension_pdb')
        if not os.path.exists(npz_base_path):
            os.makedirs(npz_base_path,exist_ok=True)
        if not os.path.exists(pdb_base_path):
            os.makedirs(pdb_base_path,exist_ok=True)
        print('*'*10,f'protein number:{len(valid_loader)}','*'*10)
        for valid_feats, pdb_names,start_index  in valid_loader:
            pbar = tqdm(range(length))
            atom_traj = []
            rigid_traj = []
            for j in pbar:
                pbar.set_description(f'Processing {pdb_names} step:{j}/{length}')
                # initialize input data
                sample_length =  valid_feats['aatype'].shape[-1]
                frame_time = self._model_conf.frame_time
                ref_number = self._model_conf.ref_number
                motion_number = self._model_conf.motion_number
                frame_time_ref_motion = frame_time+ref_number+motion_number

                res_mask = np.ones((frame_time,sample_length))
                fixed_mask = np.zeros_like(res_mask)
                res_idx = torch.arange(1, sample_length+1).unsqueeze(0).repeat(frame_time,1)
                ref_sample = self.diffuser.sample_ref(
                    n_samples=sample_length*frame_time,
                    as_tensor_7=True,
                )
                ref_sample['rigids_t'] = ref_sample['rigids_t'].reshape([-1,frame_time,sample_length,7])  # add the batch dim

                ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)

                init_feats = {
                    'res_mask': res_mask[None],
                    'seq_idx': res_idx[None],
                    'fixed_mask': fixed_mask[None],
                    'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],
                    'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None],
                    'node_repr':valid_feats['node_repr'],
                    'edge_repr':valid_feats['edge_repr'],
                    'aatype':valid_feats['aatype'],
                    **ref_sample,
                    'rigids_0':valid_feats['rigids_0'], #TODO


                    'atom37_pos': valid_feats['atom37_pos'],
                    'atom37_mask': valid_feats['atom37_mask'],

                    'ref_rigids_0':valid_feats['ref_rigids_0'],
                    'motion_rigids_0':valid_feats['motion_rigids_0'],
                    'ref_atom37_pos':valid_feats['ref_atom37_pos'],
                    'motion_atom37_pos':valid_feats['motion_atom37_pos'],
                    # 'rigids_t': diff_rigids_t[None].to(device)  # rigids_t based on gt
                }
                # TODO here
                # fasta_aatype = du.move_to_np(valid_feats['aatype'])[0] # remove the batch(1,...) to (...),conver form [1,N] to [N,]
                init_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
                init_feats = tree.map_structure(lambda x: x.to(device), init_feats)

                # TODO flatten the dim of batch and frame_time
                for k in init_feats.keys():
                    v = init_feats[k]
                    if len(v.shape)>1:
                        reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维
                        init_feats[k] = reshaped_tensor


                # start inference
                # start_time = time.time()
                sample_out = self.inference_fn(
                    init_feats,
                    num_t=num_t,
                    min_t=min_t,
                    aux_traj=True,
                    noise_scale=noise_scale,
                )

                atom_pred = sample_out['prot_traj'][0]
                rigid_pred = sample_out['rigid_traj'][0]

                atom_traj.append(atom_pred[-frame_time:]) # concate the last frame
                rigid_traj.append(rigid_pred[-frame_time:])

                # print(valid_feats['ref_rigids_0'].shape)
                # update motion atom and motion rigids
                # print(init_feats['motion_rigids_0'].shape)
                # print(rigid_pred.shape)
                # print(init_feats['motion_atom37_pos'].shape)
                # print(atom_pred.shape)

                # print('============>>>>>>>>',init_feats['motion_atom37_pos'].shape,init_feats['ref_atom37_pos'].shape)

                # TODO old version without normal
                # update motion and reference
                concat_rigids_with_pred = torch.concat([init_feats['motion_rigids_0'],init_feats['ref_rigids_0'],torch.from_numpy(rigid_pred).to(device).to(init_feats['motion_rigids_0'].dtype)],dim=0)
                concat_atom37_with_pred = torch.concat([init_feats['motion_atom37_pos'],init_feats['ref_atom37_pos'],torch.from_numpy(atom_pred).to(device).to(init_feats['motion_atom37_pos'].dtype)],dim=0)

                valid_feats['ref_rigids_0'] = concat_rigids_with_pred[-ref_number:].unsqueeze(0)
                valid_feats['ref_atom37_pos'] = concat_atom37_with_pred[-ref_number:].unsqueeze(0)

                valid_feats['motion_rigids_0'] = concat_rigids_with_pred[-(motion_number+ref_number):-ref_number].unsqueeze(0)
                valid_feats['motion_atom37_pos'] = concat_atom37_with_pred[-(motion_number+ref_number):-ref_number].unsqueeze(0)


                # new 
                # concat_atom37_with_pred = torch.concat([init_feats['motion_atom37_pos'],init_feats['ref_atom37_pos'],torch.from_numpy(atom_pred).to(device).to(init_feats['motion_atom37_pos'].dtype)],dim=0)
                
                # chain_feats = {
                #     'aatype': valid_feats['aatype'].expand(-1,motion_number+ref_number, -1), #  torch.Size([1, 1, 40])
                #     'all_atom_positions': concat_atom37_with_pred[-(motion_number+ref_number):], #  torch.Size([1, 3, 40, 37, 3])
                #     'all_atom_mask': valid_feats['atom37_mask'].expand(-1,motion_number+ref_number, -1, -1) #torch.Size([1, 1, 40, 37])
                # }
                # # normal with the first frame in snapshot
                # chain_feats = {key:value.squeeze(0).cpu().numpy() for key,value in chain_feats.items()}
                # ca_idx = residue_constants.atom_order['CA']
                # chain_feats['bb_mask'] = chain_feats['all_atom_mask'][:, ca_idx] # [N,37]
                # bb_pos = chain_feats['all_atom_positions'][0,:, ca_idx] # [F,N,37,3]->[N,3] select first protein as anchor
                # bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5) # [3]
                # centered_pos = chain_feats['all_atom_positions'] - bb_center[None, None, None, :] # [F,N,37,3]
                # scaled_pos = centered_pos / 1.0
                # chain_feats['all_atom_positions'] = scaled_pos * (chain_feats['all_atom_mask'][..., None])

                # chain_feats = {key:torch.from_numpy(value).to(device) for key,value in chain_feats.items()}
                # chain_feats = data_transforms.atom37_to_frames(chain_feats)
                # rigids =  ru.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'][:,:,0]).to_tensor_7()
                # pos = chain_feats['all_atom_positions']

                # valid_feats['ref_rigids_0'] =rigids[-ref_number:].unsqueeze(0)
                # valid_feats['ref_atom37_pos'] = pos[-ref_number:].unsqueeze(0)

                # valid_feats['motion_rigids_0'] = rigids[-(motion_number+ref_number):-ref_number].unsqueeze(0)
                # valid_feats['motion_atom37_pos'] = pos[-(motion_number+ref_number):-ref_number].unsqueeze(0)

                # print(valid_feats['ref_rigids_0'].shape,valid_feats['ref_atom37_pos'].shape,valid_feats['motion_rigids_0'].shape,valid_feats['motion_atom37_pos'].shape)
                # end

                # exit()

                
                # update ref rigids_t
                ref_sample = self.diffuser.sample_ref(
                    n_samples=sample_length*frame_time,
                    as_tensor_7=True,
                )
                ref_sample['rigids_t'] = ref_sample['rigids_t'].reshape([-1,frame_time,sample_length,7])

                ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)

                valid_feats['rigids_t'] = ref_sample['rigids_t']
                # for k,v in valid_feats.items():
                #     print(k,v.shape)
                # exit()
                # valid_feats['rigids_0'] = torch.from_numpy(np.concatenate([rigid_pred[1:], rigid_pred[-1:]], axis=0)).unsqueeze(0).to(valid_feats['rigids_0'].device).to(valid_feats['rigids_0'].dtype)
            atom_traj = np.concatenate(atom_traj, axis=0)
            rigid_traj = np.concatenate(rigid_traj, axis=0)
            save_path = os.path.join(npz_base_path,f'{pdb_names[0]}_time_{length}.npz')
            aatype = valid_feats['aatype'].cpu().numpy() # [1,T,N]
            np.savez_compressed(save_path,name=pdb_names[0], atom_traj=atom_traj, rigid_traj=rigid_traj, aatype=aatype)
            # change npz to pdb and save
            b_factors = np.tile((np.ones(aatype.shape[-1]) * 100)[:, None], (1, 37)) # N,37

            _ = au.write_prot_to_pdb(
                prot_pos=atom_traj,
                file_path=os.path.join(pdb_base_path,f'{pdb_names[0]}_time_{length}.pdb'),
                aatype=aatype[0,0],
                no_indexing=True,
                b_factors=b_factors
            )
            print(f'successfully save {pdb_names[0]} to {pdb_base_path}')


    def eval_fn_multi(self, eval_dir, valid_loader, device, exp_name, in_diffuser,data_conf,num_workers,eval_batch_size,
                    min_t=None, num_t=None, noise_scale=1.0,is_training=True,start_idx=0,end_idx=18000):
        
        
        new_data_conf = data_conf.copy()

        pdb_csv = pd.read_csv(new_data_conf.test_csv_path) # 读取CSV文件
        pdb_csv = pdb_csv[pdb_csv.seq_len <= new_data_conf.filtering.max_len]
        pdb_csv = pdb_csv.head(data_conf.max_protein_num)
        # pdb_csv = pdb_csv.iloc[35:] #13:23
        print(pdb_csv)
        print('Num of protein', len(pdb_csv))
        diffuser = in_diffuser
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        for pdb_idx_in_csv in range(len(pdb_csv)):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            res_dict_list = [] # store res for one protein
            csv_row = pdb_csv.iloc[pdb_idx_in_csv]
            pdb_name = csv_row['name']
            print('='*20)
            print(f'Processing {pdb_name} from {start_idx} to {end_idx}')
            test_dataset = Sfold_data_loader_dynamic.PdbEvalDataset(
                    data_conf=new_data_conf,
                    diffuser=diffuser,
                    traj_npz_path=csv_row['atlas_npz'],
                    omega_embed_path=csv_row['embed_path'],
                    sample_numbers=end_idx-start_idx,
                    pdb_name=pdb_name,
                    is_training=False,
                    is_testing=True,
                    is_random_test=False,
            )
            num_workers = num_workers
            persistent_workers = True if num_workers > 0 else False
            prefetch_factor=2
            prefetch_factor = 2 if num_workers == 0 else prefetch_factor
            valid_loader = data.DataLoader(
                    test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    drop_last=False,
                    multiprocessing_context='fork' if num_workers != 0 else None,
            )

            rot_trans_error_dict = {'name':[],"ave_rot":[],"ave_trans":[],"first_rot":[],"first_trans":[],'time_rot_dif':[],'time_trans_dif':[]}
            metric_list = []
            save_name_list = []
            pbars = tqdm(enumerate(valid_loader), total=len(valid_loader))
            for idx,(valid_feats, pdb_names) in pbars:
                pbars.set_description(f'Proccessing {idx}: ')
                # reset seed for different 
                save_name = pdb_names[0].split('.')[0]
                # initialize input data
                sample_length =  valid_feats['aatype'].shape[-1]
                frame_time =  valid_feats['aatype'].shape[1]
                res_mask = np.ones((frame_time,sample_length))
                fixed_mask = np.zeros_like(res_mask)
                res_idx = torch.arange(1, sample_length+1).unsqueeze(0).repeat(frame_time,1)
                ref_sample = self.diffuser.sample_ref(
                    n_samples=sample_length*frame_time,
                    as_tensor_7=True,
                )
                ref_sample = tree.map_structure(lambda x: x[None].to(device), ref_sample)

                init_feats = {
                    'res_mask': res_mask[None],
                    'seq_idx': res_idx[None],
                    'fixed_mask': fixed_mask[None],
                    'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],
                    'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None],
                    'node_repr':valid_feats['node_repr'],
                    'edge_repr':valid_feats['edge_repr'],
                    'aatype':valid_feats['aatype'],

                    **ref_sample,
                    'rigids_0':valid_feats['rigids_0'], #TODO
                    'atom37_pos': valid_feats['atom37_pos'],
                    'atom37_mask': valid_feats['atom37_mask'],
                    'ref_rigids_0':valid_feats['ref_rigids_0'],
                    'motion_rigids_0':valid_feats['motion_rigids_0'],
                    'ref_atom37_pos':valid_feats['ref_atom37_pos'],
                    'motion_atom37_pos':valid_feats['motion_atom37_pos'],
                }

                # TODO here
                # fasta_aatype = du.move_to_np(valid_feats['aatype'])[0] # remove the batch(1,...) to (...),conver form [1,N] to [N,]
                init_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
                init_feats = tree.map_structure(lambda x: x.to(device), init_feats)
                # TODO flatten the dim of batch and frame_time
                for k in init_feats.keys():
                    v = init_feats[k]
                    if len(v.shape)>1:
                        reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维
                        init_feats[k] = reshaped_tensor
                # print('===============================',i)
                # print(init_feats['rigids_t'][0,0])
                # print('===============================',i)
                sample_out = self.inference_fn(
                    init_feats,
                    num_t=num_t,
                    min_t=min_t, 
                    aux_traj=True,
                    noise_scale=noise_scale,
                )
                # prediction metrics
                eval_dic = compute_validation_metrics_all(
                                        gt_pos=valid_feats['atom37_pos'][0], # [0] for the batchsize=1
                                        out_pos=sample_out['prot_traj'][0],
                                        gt_mask=valid_feats['atom37_mask'][0],
                                        superimposition_metrics=True)
            
                # mean_eval_dic = {k: sum(v[1:]) / len(v[1:]) for k, v in eval_dic.items()}
                mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}
                save_name_list.append(save_name)
                ########################################################################
                # metric for un-aligned prediciton
                metric_list.append({k: mean_eval_dic[k].cpu().numpy() if torch.is_tensor(mean_eval_dic[k]) else mean_eval_dic[k] for k in mean_eval_dic})

                all_atom37_pos = valid_feats['atom37_pos'][0].cpu().numpy()
                tmp_ref_rigids = valid_feats['ref_rigids_0'][0,-1].cpu().numpy()

                ave_quat,ave_trans,ref_ave_quat,ref_ave_trans,time_rot_dif,time_trans_dif = self._calc_rot_trans_error(sample_out['rigid_traj'][0],
                                                                                                                   gt_rigids=init_feats['rigids_0'].cpu().numpy(),
                                                                                                                   ref_rigids=tmp_ref_rigids)
                rot_trans_error_dict['name'].append(save_name) # pdb name
                rot_trans_error_dict['ave_rot'].append(ave_quat) # average rotation error (degree)
                rot_trans_error_dict['ave_trans'].append(ave_trans) # average translation error MSE
                rot_trans_error_dict['first_rot'].append(ref_ave_quat)
                rot_trans_error_dict['first_trans'].append(ref_ave_trans)
                rot_trans_error_dict['time_rot_dif'].append(time_rot_dif)
                rot_trans_error_dict['time_trans_dif'].append(time_trans_dif)

                # Calculating the mean of each list excluding the first element
                if self._conf.data.frame_time>1:
                    rot_trans_error_mean = {key: [np.mean(arr[1:]) for arr in values]
                                for key, values in rot_trans_error_dict.items() if key != 'name'}
                else:
                    rot_trans_error_mean = {key: [np.mean(arr) for arr in values]
                                for key, values in rot_trans_error_dict.items() if key != 'name'}
                rot_trans_error_mean = {key: sum(values) / len(values) for key, values in rot_trans_error_mean.items() if key != 'name'}

                # error_fig = plot_rot_trans_curve(rot_trans_error_dict,save_path=eval_dir,frame_step=self._data_conf.frame_sample_step)

                # un-aligned prediciton metric 
                ckpt_eval_metrics = pd.DataFrame(metric_list)
                ckpt_eval_metrics.insert(0,'pdb_name',save_name_list)

                # use aligned prediciton metric to save the best model
                mean_dict = ckpt_eval_metrics.mean()
                mean_dict = mean_dict.to_dict()
                # if mean_dict['alignment_rmsd'] < self.bset_rmsd_ca:
                rmse_all = mean_dict['rmse_all'] # RMSE
                rmse_ca = mean_dict['rmse_ca'] # RMSE-Ca

                # rmsd_ca = mean_dict['rmsd_ca_aligned'] # metrics["rmsd_ca_aligned"] = alignment_rmsd
                # rmsd_all = mean_dict['alignment_rmsd']
                drmsd = mean_dict['drmsd_ca']

                rot_error = rot_trans_error_mean['ave_rot']
                trans_error = rot_trans_error_mean['ave_trans']
                ref_rot_error = rot_trans_error_mean['first_rot']
                ref_trans_error = rot_trans_error_mean['first_trans']

                tmp_res_dict =  {
                        'sample_idx':idx,
                        'rmse_all': rmse_all ,
                        'rmse_ca': rmse_ca,
                        'rot_error':rot_error,
                        'rot_ref_rot_error':ref_rot_error,
                        'trans_error':trans_error,
                        'ref_trans_error':ref_trans_error,
                        'relat_drmsd': drmsd,
                        'rigids':sample_out['rigid_traj'][0],
                        'atom_pos':sample_out['prot_traj'][0]
                        }
                res_dict_list.append(tmp_res_dict)
                print('========>>>>>>>  idx:',idx,save_name,tmp_res_dict['trans_error'],tmp_res_dict['rot_error'],sample_out['prot_traj'][0].shape,rmse_ca)
            
            save_path = f'{eval_dir}/{save_name}_{exp_name}_from_{start_idx}_to_{end_idx}.pkl'
            
            print(f'================>>>>> save to {save_path}')
            with open(save_path, 'wb') as pkl_file:
                pickle.dump(res_dict_list, pkl_file)

            print("save successful")
            aatype = valid_feats['aatype'].cpu().numpy() # [1,T,N]
            b_factors = np.tile((np.ones(aatype.shape[-1]) * 100)[:, None], (1, 37)) # N,37

            all_atom_positions = np.concatenate([d['atom_pos'] for d in res_dict_list], axis=0)

            _ = au.write_prot_to_pdb(
                prot_pos=all_atom_positions,
                file_path=os.path.join(eval_dir,f'{pdb_names[0]}_multi_pred_{len(valid_loader)}.pdb'),
                aatype=aatype[0,0],
                no_indexing=True,
                b_factors=b_factors
            )
            # # Convert the list of dictionaries to a DataFrame
            # df = pd.DataFrame(res_dict_list)

            # # Calculate the mean values for the specified keys
            # average_errors = df.mean()

            # # Print the results
            # print("Average Errors:")
            # print(average_errors)
            # exit()

    def eval_fn_multi_ref(self, eval_dir, valid_loader, device, exp_name, in_diffuser,data_conf,num_workers,eval_batch_size,
                    min_t=None, num_t=None, noise_scale=1.0,is_training=True,start_idx=0,end_idx=18000):
        
        print('='*50)
        print(f'*****Note, we are evaluating reference now!!!!*****')
        new_data_conf = data_conf.copy()

        pdb_csv = pd.read_csv(new_data_conf.test_csv_path) # 读取CSV文件
        pdb_csv = pdb_csv[pdb_csv.seq_len <= new_data_conf.filtering.max_len]
        pdb_csv = pdb_csv.head(6)
        
        diffuser = in_diffuser
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        for pdb_idx_in_csv in range(len(pdb_csv)):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            res_dict_list = [] # store res for one protein
            csv_row = pdb_csv.iloc[pdb_idx_in_csv]
            pdb_name = csv_row['name']
            print('='*20)
            print(f'Processing {pdb_name} from {start_idx} to {end_idx}')
            test_dataset = Sfold_data_loader_dynamic.PdbEvalDataset(
                    data_conf=new_data_conf,
                    diffuser=diffuser,
                    traj_npz_path=csv_row['atlas_npz'],
                    omega_embed_path=csv_row['embed_path'],
                    sample_numbers=end_idx-start_idx,
                    pdb_name=pdb_name,
                    is_training=False,
                    is_testing=True,
                    is_random_test=False,
            )
            num_workers = num_workers
            persistent_workers = True if num_workers > 0 else False
            prefetch_factor=2
            prefetch_factor = 2 if num_workers == 0 else prefetch_factor
            valid_loader = data.DataLoader(
                    test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    drop_last=False,
                    multiprocessing_context='fork' if num_workers != 0 else None,
            )

            rot_trans_error_dict = {'name':[],"ave_rot":[],"ave_trans":[],"first_rot":[],"first_trans":[],'time_rot_dif':[],'time_trans_dif':[]}
            metric_list = []
            save_name_list = []
            pbars = tqdm(enumerate(valid_loader), total=len(valid_loader))
            for idx,(valid_feats, pdb_names) in pbars:
                pbars.set_description(f'Proccessing {idx}: ')
                # reset seed for different 
                save_name = pdb_names[0].split('.')[0]
                # initialize input data
                sample_length =  valid_feats['aatype'].shape[-1]
                frame_time =  valid_feats['aatype'].shape[1]
                res_mask = np.ones((frame_time,sample_length))
                fixed_mask = np.zeros_like(res_mask)
                res_idx = torch.arange(1, sample_length+1).unsqueeze(0).repeat(frame_time,1)
                ref_sample = self.diffuser.sample_ref(
                    n_samples=sample_length*frame_time,
                    as_tensor_7=True,
                )
                ref_sample = tree.map_structure(lambda x: x[None].to(device), ref_sample)

                init_feats = {
                    'res_mask': res_mask[None],
                    'seq_idx': res_idx[None],
                    'fixed_mask': fixed_mask[None],
                    'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],
                    'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None],
                    'node_repr':valid_feats['node_repr'],
                    'edge_repr':valid_feats['edge_repr'],
                    'aatype':valid_feats['aatype'],

                    **ref_sample,
                    'rigids_0':valid_feats['rigids_0'], #TODO
                    'atom37_pos': valid_feats['atom37_pos'],
                    'atom37_mask': valid_feats['atom37_mask'],
                    'ref_rigids_0':valid_feats['ref_rigids_0'],
                    'motion_rigids_0':valid_feats['motion_rigids_0'],
                    'ref_atom37_pos':valid_feats['ref_atom37_pos'],
                    'motion_atom37_pos':valid_feats['motion_atom37_pos'],
                }

                # TODO here
                # fasta_aatype = du.move_to_np(valid_feats['aatype'])[0] # remove the batch(1,...) to (...),conver form [1,N] to [N,]
                init_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
                init_feats = tree.map_structure(lambda x: x.to(device), init_feats)
                # TODO flatten the dim of batch and frame_time
                for k in init_feats.keys():
                    v = init_feats[k]
                    if len(v.shape)>1:
                        reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维
                        init_feats[k] = reshaped_tensor
                # print('===============================',i)
                # print(init_feats['rigids_t'][0,0])
                # print('===============================',i)
                # sample_out = self.inference_fn(
                #     init_feats,
                #     num_t=num_t,
                #     min_t=min_t, 
                #     aux_traj=True,
                #     noise_scale=noise_scale,
                # )
                # prediction metrics
                ref_pos = init_feats['ref_atom37_pos'].cpu().numpy()
                ref_rigids = init_feats['ref_rigids_0'].cpu().numpy()
                ref_pos = np.repeat(ref_pos, len(init_feats['rigids_0'].cpu().numpy()), axis=0)
                ref_rigids = np.repeat(ref_rigids, len(init_feats['rigids_0'].cpu().numpy()), axis=0)
                # print(ref_pos.shape,valid_feats['atom37_pos'][0].shape,ref_rigids.shape)
                eval_dic = compute_validation_metrics_all(
                                        gt_pos=valid_feats['atom37_pos'][0], # [0] for the batchsize=1
                                        out_pos=ref_pos,
                                        gt_mask=valid_feats['atom37_mask'][0],
                                        superimposition_metrics=True)
            
                # mean_eval_dic = {k: sum(v[1:]) / len(v[1:]) for k, v in eval_dic.items()}
                mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}
                save_name_list.append(save_name)
                ########################################################################
                # metric for un-aligned prediciton
                metric_list.append({k: mean_eval_dic[k].cpu().numpy() if torch.is_tensor(mean_eval_dic[k]) else mean_eval_dic[k] for k in mean_eval_dic})
                tmp_ref_rigids = valid_feats['ref_rigids_0'][0,-1].cpu().numpy()

                ave_quat,ave_trans,ref_ave_quat,ref_ave_trans,time_rot_dif,time_trans_dif = self._calc_rot_trans_error(ref_rigids,
                                                                                                                   gt_rigids=init_feats['rigids_0'].cpu().numpy(),
                                                                                                                   ref_rigids=tmp_ref_rigids
                                                                                                                   )
                rot_trans_error_dict['name'].append(save_name) # pdb name
                rot_trans_error_dict['ave_rot'].append(ave_quat) # average rotation error (degree)
                rot_trans_error_dict['ave_trans'].append(ave_trans) # average translation error MSE
                rot_trans_error_dict['first_rot'].append(ref_ave_quat)
                rot_trans_error_dict['first_trans'].append(ref_ave_trans)
                rot_trans_error_dict['time_rot_dif'].append(time_rot_dif)
                rot_trans_error_dict['time_trans_dif'].append(time_trans_dif)

                # Calculating the mean of each list excluding the first element
                if self._conf.data.frame_time>1:
                    rot_trans_error_mean = {key: [np.mean(arr[1:]) for arr in values]
                                for key, values in rot_trans_error_dict.items() if key != 'name'}
                else:
                    rot_trans_error_mean = {key: [np.mean(arr) for arr in values]
                                for key, values in rot_trans_error_dict.items() if key != 'name'}
                rot_trans_error_mean = {key: sum(values) / len(values) for key, values in rot_trans_error_mean.items() if key != 'name'}

                # error_fig = plot_rot_trans_curve(rot_trans_error_dict,save_path=eval_dir,frame_step=self._data_conf.frame_sample_step)

                # un-aligned prediciton metric 
                ckpt_eval_metrics = pd.DataFrame(metric_list)
                ckpt_eval_metrics.insert(0,'pdb_name',save_name_list)

                # use aligned prediciton metric to save the best model
                mean_dict = ckpt_eval_metrics.mean()
                mean_dict = mean_dict.to_dict()
                # if mean_dict['alignment_rmsd'] < self.bset_rmsd_ca:
                rmse_all = mean_dict['rmse_all'] # RMSE
                rmse_ca = mean_dict['rmse_ca'] # RMSE-Ca

                # rmsd_ca = mean_dict['rmsd_ca_aligned'] # metrics["rmsd_ca_aligned"] = alignment_rmsd
                # rmsd_all = mean_dict['alignment_rmsd']
                drmsd = mean_dict['drmsd_ca']

                rot_error = rot_trans_error_mean['ave_rot']
                trans_error = rot_trans_error_mean['ave_trans']
                ref_rot_error = rot_trans_error_mean['first_rot']
                ref_trans_error = rot_trans_error_mean['first_trans']

                tmp_res_dict =  {
                        'sample_idx':idx,
                        'rmse_all': rmse_all ,
                        'rmse_ca': rmse_ca,
                        'rot_error':rot_error,
                        'rot_ref_rot_error':ref_rot_error,
                        'trans_error':trans_error,
                        'ref_trans_error':ref_trans_error,
                        'relat_drmsd': drmsd,
                        'rigids':ref_rigids,
                        'atom_pos':ref_pos,
                        }
                res_dict_list.append(tmp_res_dict)
                print('========>>>>>>>  idx:',idx,save_name,tmp_res_dict['trans_error'],tmp_res_dict['rot_error'],ref_pos.shape,rmse_ca)
            
            save_path = f'{eval_dir}/{save_name}_{exp_name}_from_{start_idx}_to_{end_idx}.pkl'
            
            print(f'================>>>>> save to {save_path}')
            with open(save_path, 'wb') as pkl_file:
                pickle.dump(res_dict_list, pkl_file)

            print("save successful")
            aatype = valid_feats['aatype'].cpu().numpy() # [1,T,N]
            b_factors = np.tile((np.ones(aatype.shape[-1]) * 100)[:, None], (1, 37)) # N,37

            all_atom_positions = np.concatenate([d['atom_pos'] for d in res_dict_list], axis=0)

            _ = au.write_prot_to_pdb(
                prot_pos=all_atom_positions,
                file_path=os.path.join(eval_dir,f'{pdb_names[0]}_multi_pred_{len(valid_loader)}.pdb'),
                aatype=aatype[0,0],
                no_indexing=True,
                b_factors=b_factors
            )
            # # Convert the list of dictionaries to a DataFrame
            # df = pd.DataFrame(res_dict_list)

            # # Calculate the mean values for the specified keys
            # average_errors = df.mean()

            # # Print the results
            # print("Average Errors:")
            # print(average_errors)
            # exit()

    
    def eval_TICA(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # ergodic the validation
        length = self._conf.eval.extrapolation_time
        npz_base_path = os.path.join(eval_dir,'extension_npz')
        pdb_base_path = os.path.join(eval_dir,'extension_pdb')
        if not os.path.exists(npz_base_path):
            os.makedirs(npz_base_path,exist_ok=True)
        if not os.path.exists(pdb_base_path):
            os.makedirs(pdb_base_path,exist_ok=True)
        print('*'*10,f'protein number:{len(valid_loader)}','*'*10)
        for valid_feats, pdb_names in valid_loader:
            pbar = tqdm(range(length))
            atom_traj = []
            rigid_traj = []
            for j in pbar:
                pbar.set_description(f'Processing {pdb_names} step:{j}/{length}')
                # initialize input data
                sample_length =  valid_feats['aatype'].shape[-1]
                frame_time = self._model_conf.frame_time
                ref_number = self._model_conf.ref_number
                motion_number = self._model_conf.motion_number
                frame_time_ref_motion = frame_time+ref_number+motion_number

                res_mask = np.ones((frame_time,sample_length))
                fixed_mask = np.zeros_like(res_mask)
                res_idx = torch.arange(1, sample_length+1).unsqueeze(0).repeat(frame_time,1)
                ref_sample = self.diffuser.sample_ref(
                    n_samples=sample_length*frame_time,
                    as_tensor_7=True,
                )
                ref_sample['rigids_t'] = ref_sample['rigids_t'].reshape([-1,frame_time,sample_length,7])  # add the batch dim

                ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)

                init_feats = {
                    'res_mask': res_mask[None],
                    'seq_idx': res_idx[None],
                    'fixed_mask': fixed_mask[None],
                    'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],
                    'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None],
                    'node_repr':valid_feats['node_repr'],
                    'edge_repr':valid_feats['edge_repr'],
                    'aatype':valid_feats['aatype'],
                    **ref_sample,
                    'rigids_0':valid_feats['rigids_0'], #TODO


                    'atom37_pos': valid_feats['atom37_pos'],
                    'atom37_mask': valid_feats['atom37_mask'],

                    'ref_rigids_0':valid_feats['ref_rigids_0'],
                    'motion_rigids_0':valid_feats['motion_rigids_0'],
                    'ref_atom37_pos':valid_feats['ref_atom37_pos'],
                    'motion_atom37_pos':valid_feats['motion_atom37_pos'],
                    # 'rigids_t': diff_rigids_t[None].to(device)  # rigids_t based on gt
                }
                # TODO here
                # fasta_aatype = du.move_to_np(valid_feats['aatype'])[0] # remove the batch(1,...) to (...),conver form [1,N] to [N,]
                init_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
                init_feats = tree.map_structure(lambda x: x.to(device), init_feats)

                # TODO flatten the dim of batch and frame_time
                for k in init_feats.keys():
                    v = init_feats[k]
                    if len(v.shape)>1:
                        reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维
                        init_feats[k] = reshaped_tensor


                # start inference
                # start_time = time.time()
                sample_out = self.inference_fn(
                    init_feats,
                    num_t=num_t,
                    min_t=min_t,
                    aux_traj=True,
                    noise_scale=noise_scale,
                )

                atom_pred = sample_out['prot_traj'][0]
                rigid_pred = sample_out['rigid_traj'][0]

                atom_traj.append(atom_pred[-frame_time:]) # concate the last frame
                rigid_traj.append(rigid_pred[-frame_time:])

                # # TODO old version without normal
                # # update motion and reference
                # concat_rigids_with_pred = torch.concat([init_feats['motion_rigids_0'],init_feats['ref_rigids_0'],torch.from_numpy(rigid_pred).to(device).to(init_feats['motion_rigids_0'].dtype)],dim=0)
                # concat_atom37_with_pred = torch.concat([init_feats['motion_atom37_pos'],init_feats['ref_atom37_pos'],torch.from_numpy(atom_pred).to(device).to(init_feats['motion_atom37_pos'].dtype)],dim=0)

                valid_feats['ref_rigids_0'] = concat_rigids_with_pred[-ref_number:].unsqueeze(0)
                valid_feats['ref_atom37_pos'] = concat_atom37_with_pred[-ref_number:].unsqueeze(0)

                # valid_feats['motion_rigids_0'] = concat_rigids_with_pred[-(motion_number+ref_number):-ref_number].unsqueeze(0)
                # valid_feats['motion_atom37_pos'] = concat_atom37_with_pred[-(motion_number+ref_number):-ref_number].unsqueeze(0)
                
                # update ref rigids_t
                ref_sample = self.diffuser.sample_ref(
                    n_samples=sample_length*frame_time,
                    as_tensor_7=True,
                )
                ref_sample['rigids_t'] = ref_sample['rigids_t'].reshape([-1,frame_time,sample_length,7])

                ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)

                valid_feats['rigids_t'] = ref_sample['rigids_t']
                # for k,v in valid_feats.items():
                #     print(k,v.shape)
                # exit()
                # valid_feats['rigids_0'] = torch.from_numpy(np.concatenate([rigid_pred[1:], rigid_pred[-1:]], axis=0)).unsqueeze(0).to(valid_feats['rigids_0'].device).to(valid_feats['rigids_0'].dtype)
            atom_traj = np.concatenate(atom_traj, axis=0)
            rigid_traj = np.concatenate(rigid_traj, axis=0)
            save_path = os.path.join(npz_base_path,f'{pdb_names[0]}_time_{length}.npz')
            aatype = valid_feats['aatype'].cpu().numpy() # [1,T,N]
            np.savez_compressed(save_path,name=pdb_names[0], atom_traj=atom_traj, rigid_traj=rigid_traj, aatype=aatype,
                                ref_rigid=init_feats['ref_rigids_0'].cpu().numpy(),ref_pos=init_feats['ref_atom37_pos'].cpu().numpy())
            # change npz to pdb and save
            b_factors = np.tile((np.ones(aatype.shape[-1]) * 100)[:, None], (1, 37)) # N,37

            _ = au.write_prot_to_pdb(
                prot_pos=atom_traj,
                file_path=os.path.join(pdb_base_path,f'For_TICA_{pdb_names[0]}_time_{length}.pdb'),
                aatype=aatype[0,0],
                no_indexing=True,
                b_factors=b_factors
            )
            # print(init_feats['ref_atom37_pos'].cpu().numpy().shape,atom_traj.shape)
            _ = au.write_prot_to_pdb(
                prot_pos=init_feats['ref_atom37_pos'].cpu().numpy(),
                file_path=os.path.join(pdb_base_path,f'For_TICA_Ref_{pdb_names[0]}_time_{length}.pdb'),
                aatype=aatype[0,0],
                no_indexing=True,
                b_factors=b_factors
            )

            print(f'successfully save {pdb_names[0]} to {pdb_base_path}')


    def eval_iterative(self, eval_dir, valid_loader, device, exp_name, in_diffuser,data_conf,num_workers,eval_batch_size,iterative_time=1,
                    min_t=None, num_t=None, noise_scale=1.0,is_training=True,start_idx=0,end_idx=18000):
        
        
        new_data_conf = data_conf.copy()

        pdb_csv = pd.read_csv(new_data_conf.test_csv_path) # 读取CSV文件
        pdb_csv = pdb_csv[pdb_csv.seq_len <= new_data_conf.filtering.max_len]
        pdb_csv = pdb_csv.head(6)
        
        diffuser = in_diffuser
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        for pdb_idx_in_csv in range(len(pdb_csv)):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            res_dict_list = [] # store res for one protein
            csv_row = pdb_csv.iloc[pdb_idx_in_csv]
            pdb_name = csv_row['name']
            print('='*20)
            print(f'Processing {pdb_name} from {start_idx} to {end_idx}')
            test_dataset = Sfold_data_loader_dynamic.PdbEvalDataset(
                    data_conf=new_data_conf,
                    diffuser=diffuser,
                    traj_npz_path=csv_row['atlas_npz'],
                    omega_embed_path=csv_row['embed_path'],
                    sample_numbers=end_idx-start_idx,
                    pdb_name=pdb_name,
                    is_training=False,
                    is_testing=True,
                    is_random_test=False,
            )
            num_workers = num_workers
            persistent_workers = True if num_workers > 0 else False
            prefetch_factor=2
            prefetch_factor = 2 if num_workers == 0 else prefetch_factor
            valid_loader = data.DataLoader(
                    test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    drop_last=False,
                    multiprocessing_context='fork' if num_workers != 0 else None,
            )

            rot_trans_error_dict = {'name':[],"ave_rot":[],"ave_trans":[],"first_rot":[],"first_trans":[],'time_rot_dif':[],'time_trans_dif':[]}
            metric_list = []
            save_name_list = []
            pbars = tqdm(enumerate(valid_loader), total=len(valid_loader))
            for idx,(valid_feats, pdb_names) in pbars:
                frame_time = self._model_conf.frame_time
                ref_number = self._model_conf.ref_number
                motion_number = self._model_conf.motion_number
                # reset seed for different 
                save_name = pdb_names[0].split('.')[0]
                # initialize input data
                sample_length =  valid_feats['aatype'].shape[-1]
                frame_time =  valid_feats['aatype'].shape[1]
                res_mask = np.ones((frame_time,sample_length))
                fixed_mask = np.zeros_like(res_mask)
                res_idx = torch.arange(1, sample_length+1).unsqueeze(0).repeat(frame_time,1)
                ref_sample = self.diffuser.sample_ref(
                    n_samples=sample_length*1, #sample_length*frame_time
                    as_tensor_7=True,
                )
                ref_sample = tree.map_structure(lambda x: x[None].to(device), ref_sample)
                atom_traj = []
                rigid_traj = []
                init_feats = {
                    'res_mask': res_mask[None][:,:1],
                    'seq_idx': res_idx[None][:,:1],
                    'fixed_mask': fixed_mask[None][:,:1],
                    'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],
                    'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None][:,:1],
                    'node_repr':valid_feats['node_repr'],
                    'edge_repr':valid_feats['edge_repr'],
                    'aatype':valid_feats['aatype'][:,:1],
                    'rigids_0':valid_feats['rigids_0'][:,:1], #TODO
                    'atom37_pos': valid_feats['atom37_pos'][:,:1],
                    'atom37_mask': valid_feats['atom37_mask'][:,:1],

                    **ref_sample,
                    'ref_rigids_0':valid_feats['ref_rigids_0'],
                    'motion_rigids_0':valid_feats['motion_rigids_0'],
                    'ref_atom37_pos':valid_feats['ref_atom37_pos'],
                    'motion_atom37_pos':valid_feats['motion_atom37_pos'],
                }
                # TODO here
                # fasta_aatype = du.move_to_np(valid_feats['aatype'])[0] # remove the batch(1,...) to (...),conver form [1,N] to [N,]
                init_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
                init_feats = tree.map_structure(lambda x: x.to(device), init_feats)
                # TODO flatten the dim of batch and frame_time
                for k in init_feats.keys():
                    v = init_feats[k]
                    if len(v.shape)>1:
                        reshaped_tensor = torch.flatten(v, start_dim=0, end_dim=1)#torch.reshape(v, (tensor.size(0), -1))  # 在这里修改形状，这里示例将张量展平为二维
                        init_feats[k] = reshaped_tensor
                for j in range(iterative_time):
                    pbars.set_description(f'Proccessing {idx}:iterative {j}/{iterative_time}')
                    # for k,v in init_feats.items():
                    #     print(k,v.shape)
                    # print('='*20)
                    sample_out = self.inference_fn(
                        init_feats,
                        num_t=num_t,
                        min_t=min_t, 
                        aux_traj=True,
                        noise_scale=noise_scale,
                    )
                    
                    atom_pred = sample_out['prot_traj'][0]
                    rigid_pred = sample_out['rigid_traj'][0]

                    atom_traj.append(atom_pred[-frame_time:]) # concate the last frame
                    rigid_traj.append(rigid_pred[-frame_time:])

                    concat_rigids_with_pred = torch.concat([init_feats['motion_rigids_0'],init_feats['ref_rigids_0'],torch.from_numpy(rigid_pred).to(device).to(init_feats['motion_rigids_0'].dtype)],dim=0)
                    concat_atom37_with_pred = torch.concat([init_feats['motion_atom37_pos'],init_feats['ref_atom37_pos'],torch.from_numpy(atom_pred).to(device).to(init_feats['motion_atom37_pos'].dtype)],dim=0)

                    init_feats['ref_rigids_0'] = concat_rigids_with_pred[-ref_number:]
                    init_feats['ref_atom37_pos'] = concat_atom37_with_pred[-ref_number:]

                    init_feats['motion_rigids_0'] = concat_rigids_with_pred[-(motion_number+ref_number):-ref_number]
                    init_feats['motion_atom37_pos'] = concat_atom37_with_pred[-(motion_number+ref_number):-ref_number]

                    # update ref rigids_t
                    ref_sample = self.diffuser.sample_ref(
                        n_samples=sample_length*1, #sample_length*frame_time
                        as_tensor_7=True,
                    )
                    ref_sample['rigids_t'] = ref_sample['rigids_t'].reshape([1,sample_length,7]) #frame_time,sample_length,7

                    ref_sample = tree.map_structure(lambda x: x.to(device), ref_sample)

                    init_feats['rigids_t'] = ref_sample['rigids_t']

                atom_traj = np.concatenate(atom_traj, axis=0)
                rigid_traj = np.concatenate(rigid_traj, axis=0)
                # print(valid_feats['atom37_pos'][0].shape,valid_feats['atom37_mask'][0].shape,atom_traj.shape,rigid_traj.shape,rigid_traj.shape,valid_feats['rigids_0'].shape)
                # prediction metrics
                eval_dic = compute_validation_metrics_all(
                                        gt_pos=valid_feats['atom37_pos'][0], # [0] for the batchsize=1
                                        out_pos=atom_traj,
                                        gt_mask=valid_feats['atom37_mask'][0],
                                        superimposition_metrics=True)
                # mean_eval_dic = {k: sum(v[1:]) / len(v[1:]) for k, v in eval_dic.items()}
                mean_eval_dic = {k: sum(v) / len(v) for k, v in eval_dic.items()}
                save_name_list.append(save_name)
                ########################################################################
                # metric for un-aligned prediciton
                metric_list.append({k: mean_eval_dic[k].cpu().numpy() if torch.is_tensor(mean_eval_dic[k]) else mean_eval_dic[k] for k in mean_eval_dic})

                all_atom37_pos = valid_feats['atom37_pos'][0].cpu().numpy()
                tmp_ref_rigids = valid_feats['ref_rigids_0'][0,-1].cpu().numpy()

                ave_quat,ave_trans,ref_ave_quat,ref_ave_trans,time_rot_dif,time_trans_dif = self._calc_rot_trans_error(rigid_traj,
                                                                                                                   gt_rigids=valid_feats['rigids_0'][0].cpu().numpy(),
                                                                                                                   ref_rigids=tmp_ref_rigids
                                                                                                                   )
                rot_trans_error_dict['name'].append(save_name) # pdb name
                rot_trans_error_dict['ave_rot'].append(ave_quat) # average rotation error (degree)
                rot_trans_error_dict['ave_trans'].append(ave_trans) # average translation error MSE
                rot_trans_error_dict['first_rot'].append(ref_ave_quat)
                rot_trans_error_dict['first_trans'].append(ref_ave_trans)
                rot_trans_error_dict['time_rot_dif'].append(time_rot_dif)
                rot_trans_error_dict['time_trans_dif'].append(time_trans_dif)

                # Calculating the mean of each list excluding the first element
                if self._conf.data.frame_time>1:
                    rot_trans_error_mean = {key: [np.mean(arr[1:]) for arr in values]
                                for key, values in rot_trans_error_dict.items() if key != 'name'}
                else:
                    rot_trans_error_mean = {key: [np.mean(arr) for arr in values]
                                for key, values in rot_trans_error_dict.items() if key != 'name'}
                rot_trans_error_mean = {key: sum(values) / len(values) for key, values in rot_trans_error_mean.items() if key != 'name'}

                # error_fig = plot_rot_trans_curve(rot_trans_error_dict,save_path=eval_dir,frame_step=self._data_conf.frame_sample_step)

                # un-aligned prediciton metric 
                ckpt_eval_metrics = pd.DataFrame(metric_list)
                ckpt_eval_metrics.insert(0,'pdb_name',save_name_list)

                # use aligned prediciton metric to save the best model
                mean_dict = ckpt_eval_metrics.mean()
                mean_dict = mean_dict.to_dict()
                # if mean_dict['alignment_rmsd'] < self.bset_rmsd_ca:
                rmse_all = mean_dict['rmse_all'] # RMSE
                rmse_ca = mean_dict['rmse_ca'] # RMSE-Ca

                # rmsd_ca = mean_dict['rmsd_ca_aligned'] # metrics["rmsd_ca_aligned"] = alignment_rmsd
                # rmsd_all = mean_dict['alignment_rmsd']
                drmsd = mean_dict['drmsd_ca']

                rot_error = rot_trans_error_mean['ave_rot']
                trans_error = rot_trans_error_mean['ave_trans']
                ref_rot_error = rot_trans_error_mean['first_rot']
                ref_trans_error = rot_trans_error_mean['first_trans']

                tmp_res_dict =  {
                        'sample_idx':idx,
                        'rmse_all': rmse_all ,
                        'rmse_ca': rmse_ca,
                        'rot_error':rot_error,
                        'rot_ref_rot_error':ref_rot_error,
                        'trans_error':trans_error,
                        'ref_trans_error':ref_trans_error,
                        'relat_drmsd': drmsd,
                        'rigids':sample_out['rigid_traj'][0],
                        'atom_pos':sample_out['prot_traj'][0]
                        }
                res_dict_list.append(tmp_res_dict)
                print('========>>>>>>>  idx:',idx,save_name,tmp_res_dict['trans_error'],tmp_res_dict['rot_error'],sample_out['prot_traj'][0].shape,rmse_ca,f'data frame {frame_time}')
            
            save_path = f'{eval_dir}/{save_name}_{exp_name}_from_{start_idx}_to_{end_idx}.pkl'
            
            print(f'================>>>>> save to {save_path}')
            with open(save_path, 'wb') as pkl_file:
                pickle.dump(res_dict_list, pkl_file)

            print("save successful")
            aatype = valid_feats['aatype'].cpu().numpy() # [1,T,N]
            b_factors = np.tile((np.ones(aatype.shape[-1]) * 100)[:, None], (1, 37)) # N,37

            all_atom_positions = np.concatenate([d['atom_pos'] for d in res_dict_list], axis=0)

            _ = au.write_prot_to_pdb(
                prot_pos=all_atom_positions,
                file_path=os.path.join(eval_dir,f'{pdb_names[0]}_multi_pred_{len(valid_loader)}.pdb'),
                aatype=aatype[0,0],
                no_indexing=True,
                b_factors=b_factors
            )

    def _self_conditioning(self, batch,drop_ref=False):
        model_sc = self.model(batch,drop_ref=drop_ref,is_training = self._exp_conf.training)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        # debug
        # output, used_params = self.model.debug_foward(batch,drop_ref=False)
        # # 打印使用的参数
        # for name, param in self.model.named_parameters():
        #     if param not in used_params:
        #         print(f'Parameter {name} is NOT used in forward pass.')
        # exit()
        
        if  self._model_conf.cfg_drop_in_train and (torch.rand(1).item()< self._model_conf.cfg_drop_rate):
            drop_ref = True
        else:
            drop_ref = False
        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch,drop_ref=drop_ref)
        model_out = self.model(batch,drop_ref=drop_ref,is_training = self._exp_conf.training)

        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']

        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        torsion_loss = torsion_angle_loss(
            a=model_out['angles'],
            a_gt=batch['torsion_angles_sin_cos'],
            a_alt_gt=batch['alt_torsion_angles_sin_cos'],
            mask=batch['torsion_angles_mask']) * self._exp_conf.torsion_loss_weight

        gt_rot_score = batch['rot_score']
        # gt_trans_score = batch['trans_score']
        rot_score_scaling = batch['rot_score_scaling']
        # trans_score_scaling = batch['trans_score_scaling']
        # for k,v in batch.items():
        #     print(k,v.shape)
        
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        # Translation score loss
        # trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
        # trans_score_loss = torch.sum(
        #     trans_score_mse / trans_score_scaling[:, None, None]**2,
        #     dim=(-1, -2)
        # ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        #print(gt_trans_x0.shape, pred_trans_x0.shape)
        #exit()
        gt_trans_x0 = batch['rigids_0'][..., 4:] #* self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] #* self._exp_conf.coordinate_scaling
        #print(gt_trans_x0.shape, pred_trans_x0.shape)
        #exit()
        # trans_x0_loss = torch.sum(
        #     (gt_trans_x0 - pred_trans_x0)**2 * loss_mask[..., None],
        #     dim=(-1, -2)
        # ) / (loss_mask.sum(dim=-1) + 1e-10)
        # ref_trans_loss = torch.sum(
        #     (gt_trans_x0 -  batch['ref_rigids_0'][..., 4:][-1].unsqueeze(0).expand_as(gt_trans_x0))**2 * loss_mask[..., None],
        #     dim=(-1, -2)
        # ) / (loss_mask.sum(dim=-1) + 1e-10)
        # ref_trans_loss *= self._exp_conf.trans_loss_weight
        # trans_loss = (
        #     trans_score_loss * (batch['t'] > self._exp_conf.trans_x0_threshold)
        #     + trans_x0_loss * (batch['t'] <= self._exp_conf.trans_x0_threshold)
        # )
        # trans_loss *= self._exp_conf.trans_loss_weight
        # trans_loss *= int(self._diff_conf.diffuse_trans)

        trans_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0).abs() * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        trans_loss *= self._exp_conf.trans_loss_weight
        if False:
            # self._exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis)**2 * loss_mask[..., None]
            axis_loss = torch.sum(
                axis_loss, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)

            ref_axis_loss = (gt_rot_axis - gt_rot_axis[0].unsqueeze(0).expand_as(gt_rot_axis))**2 * loss_mask[..., None]
            ref_axis_loss = torch.sum(
                ref_axis_loss, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)


            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle)**2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)

            ref_angle_loss = (gt_rot_angle - gt_rot_angle[0].unsqueeze(0).expand_as(gt_rot_angle))**2 * loss_mask[..., None]
            ref_angle_loss = torch.sum(
                ref_angle_loss / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            ref_angle_loss *= self._exp_conf.rot_loss_weight
            ref_rot_loss = ref_angle_loss + ref_axis_loss

            angle_loss *= self._exp_conf.rot_loss_weight
            angle_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_conf.rot_loss_weight

            # ref_rot_mse = (gt_rot_score - batch['ref_rot_score'][-1].unsqueeze(0).expand_as(gt_rot_score))**2 * loss_mask[..., None]
            # ref_rot_loss = torch.sum(
            #     ref_rot_mse / rot_score_scaling[:, None, None]**2,
            #     dim=(-1, -2)
            # ) / (loss_mask.sum(dim=-1) + 1e-10)
            # ref_rot_loss *= self._exp_conf.rot_loss_weight
            # print(ref_rot_loss.shape)
            rot_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
            # exit()

        rot_loss *= int(self._diff_conf.diffuse_rot)

        
        # Backbone atom loss
        pred_atom37 = model_out['atom37'][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]  # psi
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi) # 这里只考虑psi角度，因为只有五个atom

        gt_atom37 = gt_atom37
        atom37_mask = atom37_mask

        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37)**2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        # TODO here delete the filter
        bb_atom_loss *= batch['t'] < self._exp_conf.bb_atom_loss_t_filter  # here 小于这个阈值才有atom，为什么
       
        bb_atom_loss *= self._exp_conf.aux_loss_weight

        #print(pred_atom37.shape, gt_atom37.shape, atom37_mask.shape)torch.Size([64, 170, 5, 3]) torch.Size([64, 170, 5, 3]) torch.Size([64, 170, 5])
        #exit()

        '''
        def align_rot(A, B):
            H = A.T.mm(B)
            U, S, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T
            if torch.linalg.det(R) < 0.0:
                #print("reflection")
                return torch.eye(3, dtype=R.dtype, device=R.device)
                Vt[2] = -Vt[2]
                R = Vt.T @ U.T
            return R.T

        pairwise_lddts = 0 
        ca_pos = residue_constants.atom_order["CA"]
        for i in range(pred_atom37.shape[0]):                                                                                                                                                                  
            for j in range(pred_atom37.shape[0]):                                                                                                                                                          
                pred_coords_masked_ca_j = pred_atom37[j][...,ca_pos,:].clone()
                pred_coords_masked_ca_i = pred_atom37[i][...,ca_pos,:].clone()
                #print(pred_coords_masked_ca_i.shape, pred_atom37.shape)
                #exit()
                rot_i = align_rot(pred_coords_masked_ca_j, pred_coords_masked_ca_i)
                pred_coords_masked_ca_j_align = torch.matmul(pred_coords_masked_ca_j, rot_i)
                lddt_ij = torch.sqrt(torch.sum((pred_coords_masked_ca_j_align - pred_coords_masked_ca_i)**2) / pred_coords_masked_ca_i.shape[0])
                pairwise_lddts = pairwise_lddts + lddt_ij

        #print(pairwise_lddts)
        pairwise_lddts = -0.01 * pairwise_lddts / pred_atom37.shape[0] / pred_atom37.shape[0]                                                                                                                               '''
        pairwise_lddts = 0.0

        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*5, 3])
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res*5, 3])
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res*5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch['t'] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight
        # remove the loss of reference which may damage the performance 
        batch_loss_mask = batch_loss_mask#[1:]
        final_loss = (
            rot_loss#[1:]
            + trans_loss#[1:]
            + bb_atom_loss#[1:]
            + dist_mat_loss#[1:]
            + torsion_loss#[1:]
            + pairwise_lddts
        )
        # print(
        #     rot_loss.shape,
        #     trans_loss.shape,
        #     bb_atom_loss.shape,
        #     dist_mat_loss.shape,
        #     torsion_loss.shape
        # )
        # exit()
        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)
        aux_data = {
            'batch_train_loss': final_loss.detach(),
            'batch_rot_loss': rot_loss.detach(),
            'batch_trans_loss': trans_loss.detach(),
            'batch_bb_atom_loss': bb_atom_loss.detach(),
            'batch_dist_mat_loss': dist_mat_loss.detach(),
            'batch_torsion_loss':torsion_loss.detach(),
            'total_loss': normalize_loss(final_loss).detach(),
            'rot_loss': normalize_loss(rot_loss).detach(),
            # 'ref_rot_loss':normalize_loss(ref_rot_loss).detach(),
            'trans_loss': normalize_loss(trans_loss).detach(),
            # 'ref_trans_loss':normalize_loss(ref_trans_loss).detach(),
            'bb_atom_loss': normalize_loss(bb_atom_loss).detach(),
            'dist_mat_loss': normalize_loss(dist_mat_loss).detach(),
            'torsion_loss':normalize_loss(torsion_loss).detach(),
            # 'examples_per_step': torch.tensor(batch_size-1).detach(),
            # 'res_length': torch.mean(torch.sum(bb_mask, dim=-1)).detach(),
            'update_rots':torch.mean(torch.abs(model_out['rigid_update'][...,:3]),dim=(0,1)).detach(),
            'update_trans':torch.mean(torch.abs(model_out['rigid_update'][...,-3:]),dim=(0,1)).detach(),
        }

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)

        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """
        self._model.eval()
        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device

        # print(sample_feats['rigids_t'].shape[-1])
        # blackhole_rigids = ru.Rigid.identity(sample_feats['rigids_t'].shape[:-1]).to_tensor_7().to(device)
        # print(tmp_rigids.to_tensor_7().shape)
        # print(tmp_rigids.to_tensor_7()[1,1]) #[1,0,0,0,0,0,0]
        # print(tmp_rigids.to_tensor_7()[2,1])

        # exit()
        # sample_feats['rigids_t'] = blackhole_rigids
        # sample_feats['rigids_t']= sample_feats['rigids_0'][0].expand_as(sample_feats['rigids_t'])# 所有的noise初始化为第一帧初始化 GT,存疑？？？？？
        # sample_feats['rigids_t'][0] = sample_feats['rigids_0'][0] # 第一帧为groundtruth

        t_placeholder = torch.ones((1,)).to(device)


        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = []# frame_time,N,7 [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        # b,n,_ = sample_feats['rigids_t'].shape
        # atom37_t = all_atom.compute_backbone_atom37(
        #                 bb_rigids=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
        #                 aatypes=sample_feats['aatype'],
        #                 torsions = torch.rand(b, n, 7, 2)
        #             )[0]
        # all_bb_prots.append(du.move_to_np(atom37_t))
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(sample_feats, reverse_steps[0], t_placeholder)
                sample_feats = self._self_conditioning(sample_feats)
                # 这边先做了一次有ref的前向传播
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats,is_training = self._exp_conf.training)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    # use CFG inference
                    #print(t)
                    if self._conf.model.cfg_drop_rate > 0.01:
                        ##print('newbi')
                        model_out_unref = self.model(sample_feats,drop_ref = True,is_training = self._exp_conf.training)
                        rot_score_unref = model_out_unref['rot_score']
                        trans_score_unref = model_out_unref['trans_score']
                        cfg_gamma = self._conf.model.cfg_gamma
                        # rot_score_unref = model_out_unref['rot_score']
                        # print('aaaaa', trans_score_unref, trans_score)
                        #trans_score = trans_score_unref + cfg_gamma*(trans_score-trans_score_unref)
                        rot_score = rot_score_unref
                        trans_score = trans_score_unref
                        #print('aa', trans_score_unref, trans_score, cfg_gamma)
                        #print('aa', trans_score_unref)
                        #if t < 0.95:
                        #    exit()
                        #print('aa', (trans_score_unref - trans_score).abs().max())
                        #exit()
                        #(1-cfg_gamma)*trans_score+cfg_gamma*trans_score_unref is wrong
                        # rot_score = (1-self._model_conf.cfg_drop_rate)*rot_score+self._model_conf.cfg_drop_rate*rot_score_unref
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    rigids_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                        device=device
                    )
                else:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    #print(sample_feats)                                                                                                                                                          
                    #exit()                                                                                                                                                                       
                    model_out = self.model(sample_feats,is_training = self._exp_conf.training)                                                                                                    
                    rot_score = model_out['rot_score']                                                                                                                                            
                    trans_score = model_out['trans_score']                                                                                                                                        
                    rigid_pred = model_out['rigids']                                                                                                                                              
                                                                                                                                                                                                  
                    if self._conf.model.cfg_drop_rate > 0.01:                                                                                                                                     
                        model_out_unref = self.model(sample_feats,drop_ref = True,is_training = self._exp_conf.training)                                                                          
                        rot_score_unref = model_out_unref['rot_score']                                                                                                                            
                        trans_score_unref = model_out_unref['trans_score']                                                                                                                        
                        cfg_gamma = self._conf.model.cfg_gamma                                                                                                                                    
                        #print('bbbbb', trans_score_unref, trans_score)                                                                                                                           
                        trans_score = trans_score_unref + cfg_gamma*(trans_score-trans_score_unref)                                                                                               
                        print('aa', t, trans_score_unref, cfg_gamma)                                                                                                                              
                        exit()                                                                                                                                                                    
                    if self._model_conf.embed.embed_self_conditioning:                                                                                                                            
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]                                                                                                                             
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']                                                                                                            
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']                                                                                                    
                    #model_out = self.model(sample_feats,is_training = self._exp_conf.training)                                                                                                   
                    #rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])                                                                                                                       
                    rigids_t = self.diffuser.reverse(                                                                                                                                             
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),                                                                                                                 
                        rot_score=du.move_to_np(rot_score),                                                                                                                                       
                        trans_score=du.move_to_np(trans_score),                                                                                                                                   
                        diffuse_mask=du.move_to_np(diffuse_mask),                                                                                                                                 
                        t=t,                                                                                                                                                                      
                        dt=dt,                                                                                                                                                                    
                        center=center,                                                                                                                                                            
                        noise_scale=noise_scale,                                                                                                                                                  
                        device=device                                                                                                                                                             
                    ) 

                # for k,v in model_out.items():
                #     print(k,v.shape)
                # print('='*10)
                # for k,v in sample_feats.items():
                #     print(k,v.shape)
                # exit()
                # print(rigids_t.to_tensor_7().shape,sample_feats['rigids_t'].shape,sample_feats['rigids_t'][self._data_conf.ref_number+self._data_conf.motion_number:].shape)
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                # sample_feats['rigids_t'][0] = sample_feats['rigids_0'][0] # 第一帧为groundtruth 在每次reverse中保持reference的rigids_t

                if aux_traj:
                    all_rigids.append(du.move_to_np(model_out['rigids']))

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                
                angles = model_out['angles']
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone_atom37(
                        bb_rigids=ru.Rigid.from_tensor_7(rigid_pred),
                        aatypes=sample_feats['aatype'],
                        torsions = angles
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                    # print(atom37_0.shape,trans_pred_0.shape,angles.shape)
                atom37_t = all_atom.compute_backbone_atom37(
                    bb_rigids=rigids_t, 
                    aatypes=sample_feats['aatype'],
                    torsions = angles
                    )[0]
                # atom37_t = model_out['atom37'] 
                all_bb_prots.append(du.move_to_np(atom37_t))
        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)
        # print(sample_feats['rigids_t'][0][:,1])
        # exit()
        ret = {
            'prot_traj': all_bb_prots, # with reverse
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['psi_pred'] = angles
            ret['rigid_0_traj'] = all_bb_0_pred # model prediction
        return ret

    def _calc_rot_trans_error(self,pred_rigids,gt_rigids,ref_rigids):
        # print(pred_rigids.shape,gt_rigids.shape,ref_rigids.shape)
        first_gt_rigids = ref_rigids
        pred_rigids = pred_rigids# move out the ref
        first_gt_rigids_expands = np.repeat(first_gt_rigids[np.newaxis, :, :], len(gt_rigids), axis=0)
        # pred out
        average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], pred_rigids[...,:4])
        average_trans_distances = average_translation_distances(gt_rigids[...,4:], pred_rigids[...,4:],measurement='MAE')
        # ref frame out
        ref_average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], first_gt_rigids_expands[...,:4])
        ref_average_trans_distances = average_translation_distances(gt_rigids[...,4:], first_gt_rigids_expands[...,4:],measurement='MAE')
        # caculate relative motion
        time_rot_dif = average_quaternion_distances(gt_rigids[...,:4], np.roll(gt_rigids[...,:4],shift=1,axis=0))
        time_trans_dif = average_translation_distances(gt_rigids[...,4:], np.roll(gt_rigids[...,4:],shift=1,axis=0),measurement='MAE')

        return average_quat_distances,average_trans_distances,ref_average_quat_distances,ref_average_trans_distances,time_rot_dif,time_trans_dif




@hydra.main(version_base=None, config_path="./config", config_name="train_StateFold")
def run(conf: DictConfig) -> None:

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run()
