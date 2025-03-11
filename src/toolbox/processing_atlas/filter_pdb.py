import pandas as pd
import os
import numpy as np
import mdtraj as md
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import argparse
# 
def compute_metrics(row):
    pdb_path = row['pdb_path']
    traj = md.load(pdb_path)
    pdb_ss = md.compute_dssp(traj, simplified=True)
    row['coil_percent'] = np.mean(pdb_ss == 'C')
    row['helix_percent'] = np.mean(pdb_ss == 'H')
    row['strand_percent']  = np.mean(pdb_ss == 'E')
    row['radius_gyration'] = md.compute_rg(traj)[0]  # 计算旋转半径并获取第一个帧的值
    return row

def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='merged.csv')
    parser.add_argument('--atlas_dir', type=str, default='./data/atlas/atlas_unzip')
    parser.add_argument('--save_path', type=str, default='filtered_merged.csv')
    args = parser.parse_args()

    max_helix_percent=1.0
    max_loop_percent=0.5
    min_beta_percent=-1.0
    rog_quantile=0.96
    max_len=256
    min_len=0

    base_csv_path=args.csv
    atlas_data_path = args.atlas_dir
    
    atlas_csv = pd.read_csv(base_csv_path)
    atlas_csv = atlas_csv.head(2)
    atlas_csv['pdb_path'] = atlas_csv['name'].apply(lambda x: os.path.join(atlas_data_path, x,x+'.pdb'))


    atlas_csv = atlas_csv[atlas_csv.seq_len <= max_len]
    atlas_csv = atlas_csv[atlas_csv.seq_len >= min_len]
    print("After min/max length:",len(atlas_csv))

    # get the coil_percent, helix_percent and radius_gyration
    atlas_csv = atlas_csv.apply(compute_metrics, axis=1)

    

    atlas_csv = atlas_csv[atlas_csv.helix_percent < max_helix_percent]
    print('After helix percent:',len(atlas_csv))
    atlas_csv = atlas_csv[atlas_csv.coil_percent < max_loop_percent]
    print('After coil percent:',len(atlas_csv))
    atlas_csv = atlas_csv[atlas_csv.strand_percent > min_beta_percent]
    print('After beta percent:',len(atlas_csv))


    prot_rog_low_pass = _rog_quantile_curve(atlas_csv, rog_quantile,np.arange(max_len))
    row_rog_cutoffs = atlas_csv.seq_len.map(lambda x: prot_rog_low_pass[x-1])
    pdb_csv = atlas_csv[atlas_csv.radius_gyration < row_rog_cutoffs]
    print('After rog_quantile:',len(atlas_csv))
    pdb_csv.to_csv(args.save_path)