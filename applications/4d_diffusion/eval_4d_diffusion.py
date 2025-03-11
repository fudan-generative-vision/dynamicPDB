
import os
import torch
import GPUtil
import time
import numpy as np
import copy
import hydra
import logging
import copy
import random
import pandas as pd
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils import data
from src.data import diffusion_4d_data_loader_dynamic

from src.data import utils as du
from typing import Dict
import train_4d_difffusion

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
            self.device = 'cuda:0'
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
        weights_pkl = du.read_pkl(self._weights_path, use_torch=True, map_location='cpu')#self.device

        # Merge base experiment config with checkpoint config.
        # self._conf.model = OmegaConf.merge(self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_4d_difffusion.Experiment(conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {k.replace('module.', ''):v for k,v in model_weights.items()}

        self.model.load_state_dict(model_weights)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

        self._log.info(f'Loading model Successfully!!!')

    def create_dataset(self,is_random=False):
        test_dataset = diffusion_4d_data_loader_dynamic.PdbDataset(
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
        print(self._conf.experiment)
        if self._conf.eval.mode==0:
            print('='*10, 'Eval One Step')
            ckpt_eval_metrics,curve_fig,curve_fig_aligned,error_fig,model_ckpt_update,rot_trans_error_mean = self.exp.eval_fn(eval_dir,test_loader,self.device,
                                                                                                                              noise_scale=self._exp_conf.noise_scale,
                                                                                                                              is_training=False)
            # return ckpt_eval_metrics,curve_fig,curve_fig_aligned,error_fig,model_ckpt_update,rot_trans_error_mean
        
        elif self._conf.eval.mode==1:
            print('='*10, 'Eval Extrapolation')
            self.exp.eval_extension(eval_dir,test_loader,self.device,noise_scale=self._exp_conf.noise_scale,is_training=False)

        else:
            print('='*10, 'Eval Batch')
            self.exp.eval_fn_multi(eval_dir,test_loader,self.device,in_diffuser=self.exp._diffuser,exp_name=self._weights_path.split('/')[-3],data_conf=self._data_conf,
                                num_workers=self._exp_conf.num_loader_workers,eval_batch_size=self._eval_conf.eval_batch_size,
                                noise_scale=self._exp_conf.noise_scale,is_training=False,
                                start_idx=self._data_conf.eval_start_idx,end_idx=self._data_conf.eval_end_idx)
            


@hydra.main(version_base=None, config_path="./config", config_name="eval_4d_diffusion")
def run(conf: DictConfig) -> None:
    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Evaluator(conf)

    sampler.start_evaluation()

    elapsed_time = time.time() - start_time
    # print(sampler._conf)
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
