train_csv=train_data.csv
val_csv=valid_data.csv

data_first=9000
n_motion=0
n_frame=8
sample_step=1
name=init

CUDA_VISIBLE_DEVICES=0,1 \
/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/conda_envs/se3/bin/python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 10037 \
--use_env train_SFOLD_dynamics.py experiment.num_gpus=2 experiment.batch_size=2 \
data.csv_path=$train_csv data.val_csv_path=$val_csv \
data.frame_time=$n_frame data.motion_number=$n_motion data.frame_sample_step=$sample_step \
data.fix_sample_start=$data_first \
data.random_sample_train=True  data.keep_first=$data_first \
experiment.noise_scale=0.0001 \
experiment.ckpt_freq=5000 \
experiment.base_root=./StateFold_res/StateFold_t3 \
experiment.name=$name \
experiment.separate_rot_loss=False experiment.rot_loss_t_threshold=0.0 \
experiment.rot_loss_weight=7.0 \
experiment.num_epoch=750000 \
experiment.learning_rate=0.0001 \
model.ipa.temporal=True \
model.ipa.temporal_position_max_len=512 \
model.ipa.num_blocks=4
