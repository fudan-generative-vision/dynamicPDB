import sys 

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from openfold.utils.validation_metrics import drmsd
import torch
import numpy as np
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import matplotlib.pyplot as plt
from openfold.np import residue_constants#
from openfold.utils.superimposition import superimpose
import MDAnalysis as mda
from MDAnalysis.analysis import rms,align,rdf,contacts
from scipy.stats import pearsonr
def format_func(value, tick_number):
    return f'{value:.1f}'
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(format_func)


@torch.no_grad()
def compute_validation_metrics_all(gt_pos, out_pos,gt_mask,superimposition_metrics=False):
    metrics = {}
        
    gt_coords =gt_pos# batch["atom37_pos"]
    pred_coords = out_pos#outputs["atom37_pos"]
    all_atom_mask = gt_mask

    gt_coords = gt_coords#torch.from_numpy(gt_coords)
    pred_coords = torch.from_numpy(pred_coords)
    all_atom_mask = all_atom_mask#torch.from_numpy(all_atom_mask)

    gt_coords_masked = gt_coords * all_atom_mask[..., None]
    pred_coords_masked = pred_coords * all_atom_mask[..., None] 

    ca_pos = residue_constants.atom_order["CA"]
    gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :] # [11,N,3]
    pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]# [11,N,3]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]
    #[11,N]
    drmsd_ca_score = drmsd(
        pred_coords_masked_ca,
        gt_coords_masked_ca,
        mask=all_atom_mask_ca, # still required here to compute n
    )

    metrics["drmsd_ca"] = drmsd_ca_score



    frame_time,Nseq,_,_ = gt_coords_masked.shape

    diff = gt_coords_masked.reshape([frame_time,-1,3]) - pred_coords_masked.reshape([frame_time,-1,3])  # [F,N*37,3] 
    #all_atom_mask [F,N,37]

    # RMSE
    metrics["rmse_all"]  = torch.sqrt(torch.sum(diff**2,axis=(-1,-2))/(torch.sum(all_atom_mask, dim=(-1, -2)) + 1e-4))
    diff = gt_coords_masked_ca - pred_coords_masked_ca # [F,N,3]
    #all_atom_mask_ca [F,N]
    metrics["rmse_ca"]  = torch.sqrt(torch.sum(diff**2,axis=(-1,-2))/(torch.sum(all_atom_mask_ca, dim=-1) + 1e-4))
    
    superimposed_pred, alignment_rmsd_ca = superimpose(
        gt_coords_masked_ca, pred_coords_masked_ca,
    )
    metrics["rmsd_ca_aligned"] = alignment_rmsd_ca

    return metrics

@torch.no_grad()
def plot_curve_merged(metric_merged,save_path,row_num=2,col_num=5,suffer_fix=None):
    total_width = col_num * 2
    total_height = row_num * 2
    fig, axes = plt.subplots(row_num, col_num,figsize=(total_width, total_height),dpi=300)
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
def plot_rot_trans_curve(error_dict,save_path,frame_step=1):
    rows,cols = 2,len(error_dict['name'])
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2,rows*2),dpi=300)
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
