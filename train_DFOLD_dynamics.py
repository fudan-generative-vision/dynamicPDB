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


from src.analysis import utils as au
from src.analysis import metrics

from src.data import Dfold_data_loader_dynamic,se3_diffuser,all_atom
from src.data import utils as du

from src.model import Dfold_network_dynamic
from src.experiments import utils as eu
from openfold.utils.loss import lddt, lddt_ca,torsion_angle_loss
from openfold.np import residue_constants#
from openfold.utils.superimposition import superimpose
from openfold.utils.validation_metrics import gdt_ts,gdt_ha,drmsd
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils import rigid_utils as ru
from src.toolbox.rot_trans_error import average_quaternion_distances,average_translation_distances

import MDAnalysis as mda
from MDAnalysis.analysis import rms,align,rdf,contacts
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
import mdtraj as md

def format_func(value, tick_number):
    return f'{value:.1f}'
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(format_func)


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

    gt_coords_masked = gt_coords_masked.reshape([frame_time,-1,3])
    pred_coords_masked = pred_coords_masked.reshape([frame_time,-1,3]) 

    diff = gt_coords_masked - pred_coords_masked # [F,N*37,3]
    # print(diff.shape,all_atom_mask.shape,all_atom_mask_ca.shape)
    # print(diff[1,:10,0],all_atom_mask.reshape([4,-1])[1,:10])
    # torch.Size([4, 2516(68*37), 3]) torch.Size([4, 68, 37]) torch.Size([4, 68])
    #xit()
    # diff torch.Size([4, 37N, 3])
    #all_atom_mask [F,N,37]
    metrics["rmsd_all"]  = torch.sqrt(torch.sum(diff**2,axis=(-1,-2))/(torch.sum(all_atom_mask, dim=(-1, -2)) + 1e-4))
    diff = gt_coords_masked_ca - pred_coords_masked_ca # [F,N,3]
    #all_atom_mask_ca [F,N]
    metrics["rmsd_ca"]  = torch.sqrt(torch.sum(diff**2,axis=(-1,-2))/(torch.sum(all_atom_mask_ca, dim=-1) + 1e-4))
    # print('='*100)
    superimposed_pred, alignment_rmsd = superimpose(
        gt_coords_masked_ca, pred_coords_masked_ca,
    )
    metrics["rmsd_ca_aligned"] = alignment_rmsd


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
                axes[0].plot(row['rmsd_all'],label=key, marker='o', linestyle='-')
                axes[1].plot(row['rmsd_ca_aligned'],label=key, marker='o', linestyle='-')
                axes[2].plot(row['rmsd_ca'],label=key, marker='o', linestyle='-')

                axes[0].set_title(name+' | RMSE')
                axes[1].set_title(name+' | RMSD_ca_a')
                axes[2].set_title(name+' | RMSE_ca')
            else:
                axes[0, col_id].plot(row['rmsd_all'],label=key, marker='o', linestyle='-')
                axes[1, col_id].plot(row['rmsd_ca_aligned'],label=key, marker='o', linestyle='-')
                axes[2, col_id].plot(row['rmsd_ca'],label=key, marker='o', linestyle='-')

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
    # /cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/DFOLDv2/plot/4ue8_B_simulation_1_frame_step_1_wj.pickle
    motion_pkl_path = f'/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/DFOLDv2/plot/4ue8_B_simulation_1_frame_step_{frame_step}_wj.pickle'# 1a62_A_ 4ue8_B

    # motion_pkl_path = f'/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/DFOLDv2/plot/4ue8_b_38_{frame_step}_atlas.pickle'
    try:# atlas_average_motion_traj1
        with open(motion_pkl_path, 'rb') as handle:
            loaded_data = pickle.load(handle)
    except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
        # 在这里捕获可能的异常并跳过处理
        loaded_data=None
    print(f'======>  motion step {frame_step}:',loaded_data)
    print('======>  error dict:',error_dict)
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
        axes[row_id, col_id].plot(row['rmsd_all'],label='RMSE')
        axes[row_id, col_id].plot(row['rmsd_ca_aligned'],label='RMSD_ca_a')
        axes[row_id, col_id].plot(row['rmsd_ca'],label='RMSE_ca')
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
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl')
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
        self._model = Dfold_network_dynamic.FullScoreNetwork(self._model_conf, self.diffuser)

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
        self.best_rmsd_ca = 10000
        self.best_rmsd_all = 10000
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
            train_dataset = Dfold_data_loader_dynamic.PdbDatasetExtrapolation(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
            )

            valid_dataset = Dfold_data_loader_dynamic.PdbDatasetExtrapolation(
                data_conf=self._data_conf,
                diffuser=self._diffuser,
                is_training=False
            )
        else:
            # Datasets
            train_dataset = Dfold_data_loader_dynamic.PdbDataset(
                data_conf=self._data_conf,
                diffuser=self._diffuser,
                is_training=True
            )

            valid_dataset = Dfold_data_loader_dynamic.PdbDataset(
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
                timeout=60000,
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
                    self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'],find_unused_parameters=True)
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

            # Take checkpoint
            if False:
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
                ckpt_metrics,curve_fig,curve_fig_aligned,error_fig,model_ckpt_update,rot_trans_error_mean = self.eval_fn(
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

        if self._exp_conf.ckpt_dir is not None and self.trained_epochs!=0 and self.trained_epochs % self._exp_conf.ckpt_freq ==0:
            old_best_rmsd = self.best_rmsd_all

            start_time = time.time()
            eval_dir = os.path.join(self._exp_conf.eval_dir, f'step_{self.trained_steps}')
            os.makedirs(eval_dir, exist_ok=True)

            ckpt_path = os.path.join(
                self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')

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

            eval_time = time.time() - start_time
            self._log.info(f'Finished evaluation in {eval_time:.2f}s')

            

        if return_logs:
            return global_logs

    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # initial some metrics and base save path
        # diff_first_metric_list=[]
        metric_list = []
        metric_all_list = []

        metric_aligned_list = []
        metric_aligned_all_list = []

        first_frame_all_list = []
        save_name_list = []
        rot_trans_error_dict = {'name':[],"ave_rot":[],"ave_trans":[],'all_atom_mae':[],'all_atom_mse':[],'all_atom_rmsd':[]}

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
        # ergodic the validation

        idx = 0
        for valid_feats, pdb_names in valid_loader:
            idx = idx + 1

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
                #'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2))[None],
                'torsion_angles_sin_cos':valid_feats['torsion_angles_sin_cos'],
                'torsion_angles_mask':valid_feats['torsion_angles_mask'],  
                'sc_ca_t': np.zeros((frame_time,sample_length, 3))[None],
                'node_repr':valid_feats['node_repr'],
                'edge_repr':valid_feats['edge_repr'],
                'aatype':valid_feats['aatype'],
                **ref_sample,
                'rigids_0':valid_feats['rigids_0'], #TODO
                'atom37_pos':valid_feats['atom37_pos'],
                'atom37_mask':valid_feats['atom37_mask'],
                'force':valid_feats['force'],
                'vel':valid_feats['vel']
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
            # 设置第一帧

            # align_sample_rigids_list = []
            sample_rigids = sample_out['rigid_traj'][0]

            # save the predication
            save_name = pdb_names[0].split('.')[0]
            sample_path = os.path.join(sample_root_path, f'{save_name}.pdb')
            gt_path = os.path.join(gt_root_path, f'{save_name}_gt.pdb')

            diffuse_mask = np.ones(sample_length)
            b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))
            
            save_name_list.append(save_name)

            # diff_first_metric_list.append({k: compare_to_first_dict[k].cpu().numpy() if torch.is_tensor(compare_to_first_dict[k]) else compare_to_first_dict[k] for k in compare_to_first_dict})
            all_atom37_pos = valid_feats['atom37_pos'][0].cpu().numpy()

            # caculate the rotation and translation error
            ave_quat, ave_trans, ref_ave_quat, ref_ave_trans = self._calc_rot_trans_error(sample_out['rigid_traj'][0],gt_rigids=init_feats['rigids_0'].cpu().numpy())
            rot_trans_error_dict['name'].append(save_name)
            rot_trans_error_dict['ave_rot'].append(ave_quat[-1:])
            rot_trans_error_dict['ave_trans'].append(ave_trans[-1:])


            atom_mask = valid_feats['atom37_mask'][0].unsqueeze(-1).expand(-1,-1,-1,3).cpu().numpy()
            atom_mask[-1, -1, -1] = 0.0
            atom_gt = valid_feats['atom37_pos'][0].cpu().numpy()
            atom_mae = (np.abs(atom_gt - sample_out['prot_traj'][0]) * atom_mask).sum(axis=(-1,-2,-3)) / atom_mask.sum(axis=(-1,-2,-3))
            rot_trans_error_dict['all_atom_mae'].append(atom_mae[-1:])

            atom_all_gt = atom_gt[-1].reshape(atom_gt.shape[1]*atom_gt.shape[2], 3) 
            atom_all_pred = sample_out['prot_traj'][0, -1].reshape(atom_gt.shape[1]*atom_gt.shape[2], 3) 
            atom_all_mask = atom_mask[-1].reshape(atom_gt.shape[1]*atom_gt.shape[2], 3)[:, 0] 

            atom_all_index = np.where(atom_all_mask>0)[0]

            sup = SVDSuperimposer()                                                                                                                                                               
            sup.set(atom_all_pred[atom_all_index], atom_all_gt[atom_all_index])                                                                                                                   
            sup.run()                                                                                                                                                                             
            sup.get_transformed()                                                                                                                                                                 
            rmsd = sup.get_rms()        
            rot_trans_error_dict['all_atom_rmsd'].append(np.array([rmsd]))  

            atom_mse = ((valid_feats['atom37_pos'][0].cpu().numpy() - sample_out['prot_traj'][0])**2 * atom_mask).sum(axis=(-1,-2,-3)) / atom_mask.sum(axis=(-1,-2,-3))
            rot_trans_error_dict['all_atom_mse'].append(atom_mse[-1:])
            

        rot_trans_error_dict['ave_rot'] = np.concatenate(rot_trans_error_dict['ave_rot'], axis=0) 
        rot_trans_error_dict['ave_trans'] = np.concatenate(rot_trans_error_dict['ave_trans'], axis=0)
        rot_trans_error_dict['all_atom_mae'] = np.concatenate(rot_trans_error_dict['all_atom_mae'], axis=0)
        rot_trans_error_dict['all_atom_rmsd'] = np.concatenate(rot_trans_error_dict['all_atom_rmsd'], axis=0)

        mask = rot_trans_error_dict['ave_trans'] < 10000.0

        print('ave_rot:', rot_trans_error_dict['ave_rot'].sum() / mask.sum())
        print('ave_trans:', rot_trans_error_dict['ave_trans'].sum() / mask.sum())
        print('ave_atom_mae:', rot_trans_error_dict['all_atom_mae'].sum() / mask.sum())
        print('ave_atom_rmsd:', rot_trans_error_dict['all_atom_rmsd'].sum() / mask.sum())
        #print('ave_atom_rmsd_median:', np.median(rot_trans_error_dict['all_atom_rmsd'])) 
        # should reture eval_dict
        return rot_trans_error_dict


    def eval_extension(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        # ergodic the validation
        length = 200

        atom_traj = []
        rigid_traj = []

        for valid_feats, pdb_names in valid_loader:
            for j in range(length):
                if j % 100 == 0:
                    print(f'Finish Setp {j}',end='\r')
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
                    'rigids_0':valid_feats['rigids_0'] #TODO
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

                atom_traj.append(atom_pred[-1:]) # concate the last frame
                rigid_traj.append(rigid_pred[-1:])

                valid_feats['rigids_0'] = torch.from_numpy(np.concatenate([rigid_pred[1:], rigid_pred[-1:]], axis=0)).unsqueeze(0).to(valid_feats['rigids_0'].device).to(valid_feats['rigids_0'].dtype)
            atom_traj = np.concatenate(atom_traj, axis=0)
            rigid_traj = np.concatenate(rigid_traj, axis=0)
            save_path = os.path.join(eval_dir,'extension.npz')
            np.savez_compressed(save_path, atom_traj=atom_traj, rigid_traj=rigid_traj, aatype=valid_feats['aatype'])


    def eval_fn_multi(self, eval_dir, valid_loader, device, exp_name, diffuser,data_conf,num_workers,eval_batch_size,
                       min_t=None, num_t=None, noise_scale=1.0,is_training=True):
        res_dict_list = []
        # print(data_conf)
        test_dataset = Dfold_data_loader_dynamic.ErgodicPdbDataset(
                data_conf=data_conf,
                diffuser=diffuser,
                is_training=False,
                is_testing=True,
                is_random_test=False,
                data_npz='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/processed_npz/4ue8_B.npz'
                # data_npz='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/datasets/simulation_data/processed_npz/4ue8_B_40ps/4ue8_B.npz'
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
        # print(len(valid_loader))
        # exit()
        # for i,(valid_feats, pdb_names) in tqdm(enumerate(valid_loader)):
        #     print(i)
        # exit()
        total_num = len(valid_loader)
        par = tqdm(enumerate(valid_loader))
        for i,(valid_feats, pdb_names) in par:
            par.set_description(f'Processing {i}/{total_num}')
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
                'atom37_pos':valid_feats['atom37_pos'],
                'atom37_mask':valid_feats['atom37_mask']
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
            ave_quat,ave_trans,ref_ave_quat,ref_ave_trans,time_rot_dif,time_trans_dif = self._calc_rot_trans_error(sample_out['rigid_traj'][0],gt_rigids=init_feats['rigids_0'].cpu().numpy())
            rot_trans_error_dict['name'].append(save_name)
            rot_trans_error_dict['ave_rot'].append(ave_quat)
            rot_trans_error_dict['ave_trans'].append(ave_trans)
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
            rmsd_all = mean_dict['rmsd_all']
            rmsd_ca = mean_dict['rmsd_ca']
            drmsd = mean_dict['drmsd_ca']

            rot_error = rot_trans_error_mean['ave_rot']
            trans_error = rot_trans_error_mean['ave_trans']
            ref_rot_error = rot_trans_error_mean['first_rot']
            ref_trans_error = rot_trans_error_mean['first_trans']

            # self._log.info(
            #         f'best_rmsd_all: {rmsd_all:.4f} | '
            #         f'relat_rmsd_ca: {rmsd_ca:.4f} | '
            #         f'rot error:{rot_error:.4f}/{ref_rot_error:.4f} | '
            #         f'trans error:{trans_error:.4f}/{ref_trans_error:.4f} | '
            #         f'relat_drmsd: {drmsd:.4f}'
            #     )
            tmp_res_dict =  {
                    'sample_idx':i,
                    'best_rmsd_all': rmsd_all ,
                    'relat_rmsd_ca': rmsd_ca,
                    'rot_error':rot_error,
                    'rot_ref_rot_error':ref_rot_error,
                    'trans_error':trans_error,
                    'ref_trans_error':ref_trans_error,
                    'relat_drmsd': drmsd,
                    'rigids':sample_out['rigid_traj'][0],
                    'atom_pos':sample_out['prot_traj'][0]
                    }
            res_dict_list.append(tmp_res_dict)

        # df = pd.DataFrame(res_dict_list)
        # df.set_index('sample_idx', inplace=True)
        # print(df)
        # 将列表保存为 PKL 文件
        # print(res_dict_list)
        save_path = f'model_predict_{exp_name}.pkl'
        print(f'================>>>>> save to {save_path}')
        with open(f'model_predict_{exp_name}.pkl', 'wb') as pkl_file:
            pickle.dump(res_dict_list, pkl_file)

        print("save successful")



    def _self_conditioning(self, batch,drop_ref=False):
        model_sc = self.model(batch,drop_ref=drop_ref)
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
        model_out = self.model(batch,drop_ref=drop_ref)

        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']

        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape
        torsion_loss = torsion_angle_loss(
            a=model_out['angles'],
            a_gt=batch['torsion_angles_sin_cos'],
            a_alt_gt=batch['alt_torsion_angles_sin_cos'],
            mask=batch['torsion_angles_mask'] 
        ) * self._exp_conf.torsion_loss_weight
        
        torsion_loss = torsion_loss[-1:].repeat(batch_size) #compute loss only on prediction

        gt_rot_score = batch['rot_score']
        gt_trans_score = batch['trans_score']
        rot_score_scaling = batch['rot_score_scaling']
        trans_score_scaling = batch['trans_score_scaling']
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        # Translation score loss
        trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch['rigids_0'][..., 4:] #* self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] #* self._exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0)**2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = ((gt_trans_x0[-1:]-pred_trans_x0[-1:])**2).mean(dim=(-1,-2)).repeat(batch_size) * self._exp_conf.trans_loss_weight
        # trans_loss = (
        #     trans_score_loss * (batch['t'] > self._exp_conf.trans_x0_threshold)
        #     + trans_x0_loss * (batch['t'] <= self._exp_conf.trans_x0_threshold)
        # )
        # trans_loss *= self._exp_conf.trans_loss_weight
        # trans_loss *= int(self._diff_conf.diffuse_trans)

        if self._exp_conf.separate_rot_loss:
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

            ref_rot_mse = (gt_rot_score - gt_rot_score[0].unsqueeze(0).expand_as(gt_rot_score))**2 * loss_mask[..., None]
            ref_rot_loss = torch.sum(
                ref_rot_mse / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            ref_rot_loss *= self._exp_conf.rot_loss_weight

            rot_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold

        rot_loss *= int(self._diff_conf.diffuse_rot)

        rot_loss = rot_loss[-1:].repeat(batch_size)
        #print(rot_loss.shape)
        #exit()
        
        # Backbone atom loss
        pred_atom37 = model_out['atom37'][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]  # psi
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi) # 这里只考虑psi角度，因为只有五个atom
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

        rot_loss = rot_loss * (trans_loss < 100.0).to(rot_loss.dtype)
        trans_loss = trans_loss * (trans_loss < 100.0).to(trans_loss.dtype)
        torsion_loss = torsion_loss * (trans_loss < 100.0).to(torsion_loss.dtype) 

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
            rot_loss + #[1:]
            trans_loss + #[1:]
            #bb_atom_loss#[1:]
            #+ dist_mat_loss#[1:]
            torsion_loss #[1:]
        )
        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)
        aux_data = {
            'batch_train_loss': final_loss.detach(),
            'batch_rot_loss': rot_loss.detach(),
            'batch_trans_loss': trans_loss.detach(),
            #'batch_bb_atom_loss': bb_atom_loss.detach(),
            #'batch_dist_mat_loss': dist_mat_loss.detach(),
            'batch_torsion_loss':torsion_loss.detach(),
            'total_loss': normalize_loss(final_loss).detach(),
            'rot_loss': normalize_loss(rot_loss).detach(),
            #'ref_rot_loss':normalize_loss(ref_rot_loss).detach(),
            'trans_loss': normalize_loss(trans_loss).detach(),
            #'ref_trans_loss':normalize_loss(ref_trans_loss).detach(),
            #'bb_atom_loss': normalize_loss(bb_atom_loss).detach(),
            #'dist_mat_loss': normalize_loss(dist_mat_loss).detach(),
            'torsion_loss':normalize_loss(torsion_loss).detach(),
            # 'examples_per_step': torch.tensor(batch_size-1).detach(),
            # 'res_length': torch.mean(torch.sum(bb_mask, dim=-1)).detach(),
            #'update_rots':torch.mean(torch.abs(model_out['rigid_update'][...,:3]),dim=(0,1)).detach(),
            #'update_trans':torch.mean(torch.abs(model_out['rigid_update'][...,-3:]),dim=(0,1)).detach(),
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
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    # use CFG inference
                    if self._conf.model.cfg_drop_rate > 0.01:
                        model_out_unref = self.model(sample_feats,drop_ref = True)
                        trans_score_unref = model_out_unref['trans_score']
                        cfg_gamma = self._conf.model.cfg_gamma
                        # rot_score_unref = model_out_unref['rot_score']
                        trans_score = trans_score_unref + cfg_gamma*(trans_score-trans_score_unref)
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
                    model_out = self.model(sample_feats)
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)

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
                # atom37_t = all_atom.compute_backbone_atom37(
                #     bb_rigids=rigids_t, 
                #     aatypes=sample_feats['aatype'],
                #     torsions = angles
                #     )[0]
                atom37_t = model_out['atom37'] 
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
            'prot_traj': all_bb_prots,
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['psi_pred'] = angles
            ret['rigid_0_traj'] = all_bb_0_pred
        return ret

    def _calc_rot_trans_error(self,pred_rigids,gt_rigids):
        first_gt_rigids = gt_rigids[-2]
        pred_rigids = pred_rigids# move out the ref
        gt_rigids = gt_rigids
        first_gt_rigids_expands = np.repeat(first_gt_rigids[np.newaxis, :, :], len(gt_rigids), axis=0)
        # pred out
        average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], pred_rigids[...,:4])
        average_trans_distances = average_translation_distances(gt_rigids[...,4:], pred_rigids[...,4:],measurement='MAE')
        # ref frame out
        ref_average_quat_distances = average_quaternion_distances(gt_rigids[...,:4], first_gt_rigids_expands[...,:4])
        ref_average_trans_distances = average_translation_distances(gt_rigids[...,4:], first_gt_rigids_expands[...,4:],measurement='MAE')

        #print(average_trans_distances)
        #print(ref_average_trans_distances)

        #print("==============")
        #print(pred_rigids[-1,0,:4], gt_rigids[-2,0,:4])
        #print(average_quat_distances)
        #print(ref_average_quat_distances)
        return average_quat_distances,average_trans_distances,ref_average_quat_distances,ref_average_trans_distances



@hydra.main(version_base=None, config_path="./config", config_name="train_DFOLDv2")
def run(conf: DictConfig) -> None:

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run()
