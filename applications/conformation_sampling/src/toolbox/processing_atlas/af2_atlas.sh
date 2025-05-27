data_root_path='/path/to/alignment/databases'
env_path='/path/to/openfold/env/bin'

echo $(date "+%Y-%m-%d %H:%M:%S")

CUDA_VISIBLE_DEVICES=0 \
python3 src/toolbox/processing_atlas/run_AF2_evo_feature.py \
     data/fasta \
     data/template_mmcif_files \
     --uniref90_database_path ${data_root_path}/uniref90/uniref90.fasta \
     --mgnify_database_path ${data_root_path}/mgnify/mgy_clusters_2022_05.fa \
     --pdb70_database_path ${data_root_path}/pdb70/pdb70 \
     --uniclust30_database_path ${data_root_path}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
     --bfd_database_path ${data_root_path}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
     --jackhmmer_binary_path ${env_path}/jackhmmer \
     --hhblits_binary_path ${env_path}/hhblits \
     --hhsearch_binary_path ${env_path}/hhsearch \
     --kalign_binary_path ${env_path}/kalign \
     --config_preset "model_1_ptm" \
     --model_device "cuda:0" \
     --openfold_checkpoint_path finetuning_ptm_2.pt \
     --output_dir data/af2_feat \
     --skip_relaxation \
     --use_precomputed_alignments data/msa_feat \

echo $(date "+%Y-%m-%d %H:%M:%S")
