train_csv=./examples/atlas_visual_se3_filter.csv
val_csv=./examples/atlas_visual_se3_filter.csv

data_first=9000
n_motion=2
n_frame=16
sample_step=1
name=frame${n_frame}_motion${n_motion}_step${sample_step}
NUM_GPU=1
# CUDA_VISIBLE_DEVICES=0,1,2,3 \4,5,6,7

CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port 22212 \
--use_env train_4d_difffusion.py experiment.num_gpus=${NUM_GPU} experiment.batch_size=${NUM_GPU}  \
data.csv_path=$train_csv data.val_csv_path=$val_csv \
data.frame_time=$n_frame data.motion_number=$n_motion data.frame_sample_step=$sample_step \
data.fix_sample_start=$data_first \
data.random_sample_train=True  data.keep_first=$data_first \
experiment.noise_scale=1.0 \
experiment.ckpt_freq=5000 \
experiment.base_root=./4d_dffusion_res/pub_train \
experiment.name=$name \
experiment.separate_rot_loss=False experiment.rot_loss_t_threshold=0.0 \
experiment.rot_loss_weight=7.0 \
experiment.num_epoch=750 \
model.ipa.temporal=True \
