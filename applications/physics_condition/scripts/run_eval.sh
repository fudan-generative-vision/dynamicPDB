# atlas frame 1
test_csv=./examples/merged.csv
model_path=../DFOLDv2_res/result_v2_selfdata/ckpt/force_full_gen_10_release/11D_03M_2025Y_11h_15m_54s/step_31.pth
start_idx=1
python eval_DFOLD_dynamics.py eval.weights_path=$model_path \
experiment.use_ddp=False \
data.frame_time=2  \
model.cfg_drop_rate=0.0 \
experiment.batch_size=1 \
data.test_csv_path=$test_csv \
experiment.noise_scale=0.1 \
data.frame_sample_step=1 \
data.fix_sample_start=$start_idx \
model.ipa.temporal=False \
diffuser.r3.coordinate_scaling=1.0  
# eval.name=debug \
# data.test_csv_path=test.csv
# eval.name=exp_start_$start_idx \
# data.frame_sample_step=50 \
# model.ipa.temporal=False \
# model.ipa.temporal=True model.ipa.temporal_position_encoding=True \
# experiment.base_root=../DFOLDv2_res/result_v2_extension \
# data.time_norm=True
# model.ipa.num_blocks=1 \
