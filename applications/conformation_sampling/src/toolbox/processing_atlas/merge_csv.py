import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# Index(['name', 'raw_path', 'processed_path', 'resolution', 'seq',
#        'release_date', 'seq_len', 'embed_path'],

a = pd.read_csv('atlas_test.csv')
root = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/processed_npz'
embed_root = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/embeddings/OmegaFold'
# embed_root = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/embeddings/OmegaFold_GeoFormer_recycling_1'
# embed_root = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/datasets/atlas/embeddings/OmegaFold_GeoFormer'
print(a.columns)
a = a.set_index('name')

npz_list = []
embed_path_list = []
seq_list = []
for name, row in tqdm(a.iterrows()):
    npz_path = os.path.join(root,name+'.npz')
    embed_path = os.path.join(embed_root,name+'.npz')
    seq_list.append(len(row.seqres))
    try:
        # x = np.load(npz_path,allow_pickle=True)
        # x = dict(x)

        # feature = np.load(embed_path,allow_pickle=True)
        # seq_list.append(len(x['sequence'][0].decode('utf-8') ))
        # print(x.keys())
        # for k in x.keys():
        #     print(k,x[k].shape)
        # exit()
        # if x['sequence'][0].decode('utf-8') !=a.loc[name]['seqres']:
        #     print(x['sequence'][0].decode('utf-8') )
        #     print(a.loc[name]['seqres'])
        #     print(name)
        #     npz_path = None

        npz_list.append(npz_path)
        embed_path_list.append(embed_path)
    except RuntimeError as e:
        print(e)

a['atlas_npz'] = npz_list
a['embed_path'] = embed_path_list
a['seq_len'] = seq_list
a.to_csv('test_merged_all.csv')