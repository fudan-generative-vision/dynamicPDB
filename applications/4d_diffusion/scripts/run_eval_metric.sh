
# multi eval
start_idx=0

sample_step=1
n_motion=2
n_frame=16

name=train_data_frame${n_frame}_motion${n_motion}_step${sample_step}
train_step=95000
model_path=./ckpt/frame16_step_${train_step}.pth

eval_mode=2  # eval mode :  0 for one step; 1 for extension; 2 for batch evaluation
noise_scale=1.0


test_data=./examples/atlas_visual_se3_filter.csv


CUDA_VISIBLE_DEVICES=0 python eval_4d_diffusion.py eval.weights_path=$model_path experiment.use_ddp=False \
eval.mode=${eval_mode} \
eval.name=${name}_mode_${eval_mode}_noise_scale_${noise_scale}_step_${train_step} \
experiment.batch_size=1 \
experiment.noise_scale=${noise_scale} \
experiment.base_root=./4d_dffusion_res/pub_test_evaluation \
data.frame_time=$n_frame data.motion_number=$n_motion data.frame_sample_step=$sample_step \
data.test_csv_path=$test_data \
data.fix_sample_start=$start_idx \
data.eval_start_idx=$start_idx \
data.eval_end_idx=$(($start_idx + 500)) \
# data.max_protein_num=2 \
# max_protein_num: num of proteins for evaluations