train_csv=./examples/merged.csv
val_csv=./examples/merged.csv


NUM_GPU=1
CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port 10383 --use_env train_physics_condition.py \
experiment.num_gpus=${NUM_GPU} \
experiment.batch_size=${NUM_GPU} \
experiment.noise_scale=1.0 \
model.cfg_drop_rate=0.0 \
data.random_sample_train=True \
data.keep_first=100000 \
data.fix_sample_start=100000 \
experiment.ckpt_freq=400 \
experiment.num_epoch=500000 \
experiment.base_root=./physics_res/pub_train \
data.csv_path=$train_csv \
data.val_csv_path=$val_csv \
experiment.name=force_full_gen_10_release \
data.frame_time=2 \
data.frame_sample_step=1 \
experiment.separate_rot_loss=False \
experiment.trans_loss_weight=100.0 \
experiment.rot_loss_t_threshold=0.0 \
experiment.rot_loss_weight=7.0 \
experiment.torsion_loss_weight=1.0 \
model.ipa.temporal=False \
diffuser.r3.coordinate_scaling=1.0 \
experiment.log_freq=32
