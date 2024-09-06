
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
import subprocess
from collections import defaultdict
from collections import deque
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils import data
from openfold.utils import rigid_utils as ru
from hydra.core.hydra_config import HydraConfig

from src.analysis import utils as au
from src.analysis import metrics
# from data import Dfold_data_loader_new
from src.data import Dfold_data_loader_dynamic
from src.data import se3_diffuser
from src.data import utils as du
from src.data import all_atom
from src.model import Dfold_network_dynamic
from src.experiments import utils as eu
from openfold.utils.loss import lddt, lddt_ca,torsion_angle_loss,supervised_chi_loss
from openfold.np import residue_constants#
from openfold.utils.superimposition import superimpose
from openfold.utils.validation_metrics import (
    gdt_ts,
    gdt_ha,
    drmsd
)
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from Bio.SVDSuperimposer import SVDSuperimposer
# from openfold.utils.loss import compute_fape
# from openfold.utils.rigid_utils import Rotation, Rigid
from typing import Dict
import train_DFOLD_dynamics
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
np.random.default_rng(seed)

class Evaluator:
    def __init__(
            self,
            conf: DictConfig,
            conf_overrides:Dict=None
    ):
        # 初始化参数

        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._eval_conf = conf.eval
        self._diff_conf = conf.diffuser
        self._data_conf = conf.data
        self._exp_conf = conf.experiment

        # Set-up GPU
        if torch.cuda.is_available():
            if self._eval_conf.gpu_id is None:
                available_gpus = ''.join([str(x) for x in GPUtil.getAvailable(order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._eval_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')



        # model weight
        self._weights_path = self._eval_conf.weights_path
        project_name = self._weights_path.split('/')[-3]

        output_dir =self._eval_conf.output_dir
        if self._eval_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._eval_conf.name
        self._output_dir = os.path.join(output_dir, project_name,dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')



        # Load models and experiment
        self._load_ckpt(conf_overrides)


        

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'===================>>>>>>>>>>>>>>>> Loading weights from {self._weights_path}')

        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(self._weights_path, use_torch=True, map_location=self.device)

        # Merge base experiment config with checkpoint config.
        self._conf.model = OmegaConf.merge(self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_DFOLD_dynamics.Experiment(conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {k.replace('module.', ''):v for k,v in model_weights.items()}
        # print(self.model.state_dict()['score_model.rigid_embeder.2.weight'])
        # exit()
        self.model.load_state_dict(model_weights)
        # print(self.model.state_dict()['score_model.rigid_embeder.2.weight'])
        # exit()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

        self._log.info(f'Loading model Successfully!!!')

    def create_dataset(self,is_random=False):
        if self._data_conf.is_extrapolation:
            test_dataset = Dfold_data_loader_dynamic.PdbDatasetExtrapolation(
            data_conf=self._data_conf,
            diffuser=self.exp._diffuser,
            is_training=False,
            is_testing=True,
            is_random_test=is_random
            )
        else:
            test_dataset = Dfold_data_loader_dynamic.PdbDataset(
                data_conf=self._data_conf,
                diffuser=self.exp._diffuser,
                is_training=False,
                is_testing=True,
                is_random_test=is_random
            )
        num_workers = self._exp_conf.num_loader_workers
        persistent_workers = True if num_workers > 0 else False
        prefetch_factor=2
        prefetch_factor = 2 if num_workers == 0 else prefetch_factor
        test_dataloader = data.DataLoader(
                test_dataset,
                batch_size=self._eval_conf.eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                drop_last=False,
                multiprocessing_context='fork' if num_workers != 0 else None,
        )

        return test_dataloader

    def start_evaluation(self):
        test_loader = self.create_dataset(is_random=self._conf.eval.random_sample)
        if self._eval_conf.name is None:
            eval_dir = os.path.join(self._output_dir,'eval_res')
        else:
            df = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
            eval_dir = os.path.join(self._output_dir,df)
        os.makedirs(eval_dir, exist_ok=True)

        config_path = os.path.join(eval_dir ,'eval_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')
        # for valid_feats, pdb_names in test_loader:
        #     print(pdb_names,valid_feats['atom37_pos'].shape,valid_feats['atom37_pos'][0,:,0,0])
        #     exit()
        # for test_feats, pdb_names in test_loader:
        #     print(pdb_names)
        # ckpt_eval_metrics,curve_fig,curve_fig_aligned,error_fig,model_ckpt_update,rot_trans_error_mean = self.exp.eval_fn(eval_dir,test_loader,self.device,noise_scale=self._exp_conf.noise_scale,is_training=False)
        # return ckpt_eval_metrics,curve_fig,curve_fig_aligned,error_fig,model_ckpt_update,rot_trans_error_mean
        # self.exp.eval_extension(eval_dir,test_loader,self.device,noise_scale=self._exp_conf.noise_scale,is_training=False)
        #self.exp.eval_fn_multi(eval_dir,test_loader,self.device,diffuser=self.exp._diffuser,exp_name=self._weights_path.split('/')[-3],data_conf=self._data_conf,
        #self.exp.eval_fn(eval_dir,test_loader,self.device,diffuser=self.exp._diffuser,exp_name=self._weights_path.split('/')[-3],data_conf=self._data_conf,
        #                       num_workers=self._exp_conf.num_loader_workers,eval_batch_size=self._eval_conf.eval_batch_size,
        #                       noise_scale=self._exp_conf.noise_scale,is_training=False)
        self.exp.eval_fn(eval_dir,test_loader,self.device,noise_scale=self._exp_conf.noise_scale,is_training=False)

@hydra.main(version_base=None, config_path="./config", config_name="eval_DFOLDv2")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Evaluator(conf)
    # here to infere multi times
    # for i in range(2):
    #     print(f"======>>>>>>>>>{i}")
    #ckpt_eval_metrics,curve_fig,curve_fig_aligned,error_fig,model_ckpt_update,rot_trans_error_mean = sampler.start_evaluation()
    rot_trans_error_mean = sampler.start_evaluation()
    # print(ckpt_eval_metrics,rot_trans_error_mean)
    # print('Rotation:',rot_trans_error_mean['ave_rot'],rot_trans_error_mean['first_rot'])
    # print('Translation:',rot_trans_error_mean['ave_trans'],rot_trans_error_mean['first_trans'])
    # print(rot_trans_error_mean)
    # 用于存储结果字典的列表
    # dict_list = []

    # # 遍历 DataFrame 的每一行
    # for index, row in ckpt_eval_metrics.iterrows():
    #     # 创建一个字典，将列名作为键，对应的值作为键的值
    #     row_dict = {col: (val.item() if isinstance(val, np.ndarray) else val) for col, val in row.items()}
    #     # 添加到结果列表中
    #     dict_list.append(row_dict)
    # # 还需要加入trans rots error
    # print('='*10)
    # print(dict_list)
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
