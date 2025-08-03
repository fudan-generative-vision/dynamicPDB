import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir_root', type=str, default='./dataset/embeddings')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--lm_weights_path', default="release1.pt")
parser.add_argument('--data_csv_path', type=str, default='src/toolbox/processing_atlas/data_csv/atlas.csv')
parser.add_argument('--omegafold_num_recycling', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--worker_id', type=int, default=0)
parser.add_argument('--reference_only', action='store_true', default=False)
args, _ = parser.parse_known_args()

import pandas as pd
import numpy as np
import tqdm, os, torch
from omegafold.__main__ import OmegaFoldModel
import pickle
def main():
    splits = pd.read_csv(args.data_csv_path)
    splits = splits.set_index("name")
    splits = splits.iloc[args.worker_id::args.num_workers]
    print(splits)
    arg_keys = ['omegafold_num_recycling']
    model_suffix = get_args_suffix(arg_keys, args)

    out_dir = args.out_dir_root
    # load OmegaFold model
    omegafold = OmegaFoldModel(args.lm_weights_path, device=args.device)
    skipping = 0
    doing = 0
    for dir_name in tqdm.tqdm(splits.index):
        name_with_chain = dir_name.split('.')[0]

        seq = splits.loc[name_with_chain]["seqres"] 

        embeddings_dir = out_dir
        if not os.path.exists(embeddings_dir): 
            os.makedirs(embeddings_dir)
        embeddings_path = os.path.join(embeddings_dir, name_with_chain)  +  '.npz'#'.' + suffix

        if os.path.exists(embeddings_path): 
            skipping += 1
            continue
        
        doing += 1
        fasta_lines = [f">{name_with_chain}", seq]

        try:
            edge_results, node_results = omegafold.inference(
                fasta_lines, args.omegafold_num_recycling
            )
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'CUDA OOM, skipping {dir_name} len:{len(seq)}')
                torch.cuda.empty_cache()
                continue
            
            else:
                # logger.error("Uncaught error")
                raise e
        np.savez(embeddings_path, node_repr=node_results[0], edge_repr=edge_results[0])

    print(splits, 'DONE')
    print('Skipped', skipping)
    print('Done', doing)
    
def get_args_suffix(arg_keys, args):
    cache_name = []
    for k in arg_keys:
        cache_name.extend([k, args.__dict__[k]])
    return '.'.join(map(str, cache_name))

if __name__ == '__main__':
    # TODO  在csv中添加['processed_feature_path']，保存node的路径 npz字典形式
    main()
