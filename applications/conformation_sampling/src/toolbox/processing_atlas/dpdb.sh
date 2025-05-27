data_root_path='/path/to/alignment/databases'
env_path='/path/to/openfold/env/bin'

echo $(date "+%Y-%m-%d %H:%M:%S")

/path/to/openfold/env/bin/python src/toolbox/processing_atlas/precompute_alignments.py \
    data/fasta \
    data/msa_feat \
    --uniref90_database_path ${data_root_path}/uniref90/uniref90.fasta \
    --mgnify_database_path ${data_root_path}/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path ${data_root_path}/pdb70/pdb70 \
    --uniclust30_database_path ${data_root_path}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path ${data_root_path}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path ${env_path}/jackhmmer \
    --hhblits_binary_path ${env_path}/hhblits \
    --hhsearch_binary_path ${env_path}/hhsearch \
    --kalign_binary_path ${env_path}/kalign \
    --cpus_per_task 16 \

echo $(date "+%Y-%m-%d %H:%M:%S")
