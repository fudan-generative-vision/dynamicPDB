start_idx=0
sample_step=1
n_motion=0
n_frame=64
name=train_data_frame${n_frame}_motion${n_motion}_step${sample_step}
train_step=95000
model_path=./conformation_sampling.pth
eval_mode=0  # eval mode :  0 for one step; 1 for extension; 2 for batch evaluation
noise_scale=0.0001
extrapolation_time=500

test_data=test_data.csv
CUDA_VISIBLE_DEVICES=0 \
python eval_SFOLD_dynamics.py eval.weights_path=$model_path experiment.use_ddp=False \
eval.mode=${eval_mode} \
eval.extrapolation_time=${extrapolation_time} \
eval.name=${name}_mode_${eval_mode}_noise_scale_${noise_scale}_exp_start_${start_idx}_step_${train_step} \
model.ipa.temporal_position_encoding=True \
model.ipa.temporal=True \
model.ipa.temporal_position_max_len=512 \
model.cfg_drop_rate=0.1 \
model.cfg_gamma=1.0 \
experiment.batch_size=1 \
experiment.noise_scale=${noise_scale} \
experiment.base_root=StateFold_res/ \
data.frame_time=$n_frame data.motion_number=$n_motion data.frame_sample_step=$sample_step \
data.test_csv_path=$test_data \
data.fix_sample_start=$start_idx \
data.filtering.max_len=300 \
model.ipa.num_blocks=4
