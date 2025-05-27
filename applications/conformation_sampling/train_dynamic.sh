#export TORCH_DISTRIBUTED_DEBUG=INFO
#train_csv=/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/StateFold/atlas_train_val_se3_filter.csv # train data path
#train_csv=./out.csv
#train_csv=/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/ckh_tool/processing_atlas/train_merged_all.csv
train_csv=full.csv
val_csv=/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/code/StateFold/atlas_test_se3_filter.csv # val data path
  
data_first=9000 # the time used for training in the trajectory [0:10000]->[0:9000]
n_motion=0  # not used here
n_frame=64
sample_step=1 # sampling interval in the trajectory
name=init # saving dir name 
# CUDA_VISIBLE_DEVICES=0,1,2,3 \4,5,6,7

#CUDA_VISIBLE_DEVICES=4,5 \
CUDA_VISIBLE_DEVICES=0,1,2 \
/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/protein/workspace/chengkaihui/conda_envs/se3/bin/python3 -m torch.distributed.launch --nproc_per_node=3 --master_port 22230 \
--use_env train_SFOLD_dynamics.py experiment.num_gpus=3 experiment.batch_size=3 \
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
model.ipa.num_blocks=3
# experiment.warm_start=../DFOLDv2_res/4DFOLD/ckpt/new_ref_pred_atlas_all_loss_frame1_motion0_step1/29D_07M_2024Y_00h_39m_38s/step_75000.pth \

# experiment.torsion_loss_weight=0.0 \
# experiment.dist_mat_loss_weight=0.0 \
# experiment.bb_atom_loss_weight=0.0 \
# model.ipa.frozen_spatial=True \
# experiment.warm_start=../DFOLDv2_res/result_v2_Motion_Ref/ckpt/4ue8_B_step1_ipa_block_4_all_loss_frame1_step50_no_separate_rot_weight7_thres0_no_temporal_wjdata_no_torsion_no_temporal_aux/14D_07M_2024Y_22h_14m_01s/step_300000.pth \
