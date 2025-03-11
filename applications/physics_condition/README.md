<h1 align='center'>Dynamic PDB: A New Dataset and a SE(3) Model Extension by Integrating Dynamic Behaviors and Physical Properties in Protein Structures</h1>

<div align='center'>
    Ce Liu<sup>1*</sup>&emsp;
    Jun Wang<sup>1*</sup>&emsp;
    Zhiqiang Cai<sup>1*</sup>&emsp;
    Yingxu Wang<sup>1,3</sup>&emsp;
    Huizhen Kuang<sup>2</sup>&emsp;
    Kaihui Cheng<sup>2</sup>&emsp;
    Liwei Zhang<sup>1</sup>&emsp;
</div>
<div align='center'>
    Qingkun Su<sup>1</sup>&emsp;
    Yining Tang<sup>2</sup>&emsp;
    Fenglei Cao<sup>1</sup>&emsp;
    Limei Han<sup>2</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home/' target='_blank'>Siyu Zhu</a><sup>2†</sup>&emsp;
    Yuan Qi<sup>2†</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>Shanghai Academy of AI for Science&emsp;
    <sup>2</sup>Fudan University&emsp;
    <sup>3</sup>MBZUAI
</div>

<div align='center'>
    <sup>*</sup>Equal Contribution&emsp;
    <sup>†</sup>Corresponding Author
</div>

<br>
<div align='center'>
    <a href='https://github.com/fudan-generative-vision/dynamicPDB'><img src='https://img.shields.io/github/stars/fudan-generative-vision/dynamicPDB?style=social'></a>
    <a href='https://fudan-generative-vision.github.io/dynamicPDB/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://arxiv.org/abs/2408.12413'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://www.modelscope.cn/datasets/fudan-generative-vision/dynamicPDB'><img src='https://img.shields.io/badge/Modelscope-Model-purple'></a>
</div>

<br>


## 🔧️ Framework

![framework](./asserts/physics_frame.png)

## ⚙️ Installation

- System requirement: Ubuntu 20.04/Ubuntu 22.04
- Tested GPUs: A100

Create conda environment:

```bash
  conda create -n physics_condition python=3.9
  conda activate physics_condition
  pip install -r requirements.txt
```

## 🗝️️ Usage

Before testing or training your cases, please ensure that the working directory is set in `physics_condition`:



1. [Data Preparation](#Data-Preparation).
2. [Training](#Training).
3. [Inference](#Inference).



### 📥 Data Preparation
####  Downloading datasets
- Please follow the instructions in the `README` file of the dynamicPDB repository to download the dataset. Ensure you have sufficient disk space available, as `one protein` may require over 80GB of storage.

- From the `dynamicsPDB` repository root, run:
    - processing trajectory:
        ```text
        python src/toolbox/processing_dynamicPDB/prep_dynamicPDB.py --dynamic_dir [dynamicPDB dataset root] --outdir [DIR] --simulation_suffix [simulation suffix]
        ```
    - process physics information, we extract the physics information of `Cα` :
      ```
      python src/toolbox/processing_dynamicPDB/atom_select.py --dynamic_dir [dynamicPDB dataset root]
      ```
  This will preprocess the dynamicPDB trajectories into `.npz` files. The physics property (velocity & force) will be packed as `.pkl` files.
  
####  Extract Sequence Embeddings
- Download the OmegaFold weights and install the modified OmegaFold repository.
  ```
  wget https://helixon.s3.amazonaws.com/release1.pt
  git clone https://github.com/bjing2016/OmegaFold
  pip install --no-deps -e OmegaFold
  ```
- Run OmegaFold to make the embeddings:
  ```
  python src/toolbox/processing_dynamicPDB/extract_embedding.py --reference_only --out_dir_root=./dataset/embeddings --lm_weights_path [OmegaFold weight] --data_csv_path [data csv path]  --simulation_suffix [simulation params]
  ```
  The `data csv` is organized as:
  |name|seqres|seq_len|
  |----|----|----|
  |1ab1_A|TTCCPSIVA...|415|
  ...

- These datasets should be organized as follows:

  ```text
  ./dynamicPDB/
  |-- 1ab1_A_npt1000000.0_ts0.001
  |   |-- 1ab1_A_npt_sim_data
  |   |   |-- 1ab1_A_npt_sim_0.dat
  |   |   `-- ...
  |   |-- 1ab1_A_dcd
  |   |   |-- 1ab1_A_dcd_0.dcd
  |   |   `-- ...
  |   |-- 1ab1_A_T
  |   |   |-- 1ab1_A_T_0.pkl
  |   |   `-- ...
  |   |-- 1ab1_A_F
  |   |   |-- 1ab1_A_F_0.pkl
  |   |   `-- ...
  |   |-- 1ab1_A_V
  |   |   |-- 1ab1_A_V_0.pkl
  |   |   `-- ...
  |   |-- 1ab1_A.pdb
  |   |-- 1ab1_A_minimized.pdb
  |   |-- 1ab1_A_nvt_equi.dat
  |   |-- 1ab1_A_npt_equi.dat
  |   |-- 1ab1_A_T.dcd
  |   |-- 1ab1_A_T.pkl
  |   |-- 1ab1_A_F.pkl
  |   |-- 1ab1_A_V.pkl
  |   `-- 1ab1_A_state_npt1000000.0.xml
  |-- 1uoy_A_npt1000000.0_ts0.001
  |   |-- ...
  |   `-- ...
  `-- ...
  ```
  - Optionally, you could consolidate all the information into the relevant `.csv` files and apply filtering based on specific conditions, or you could directly use the provided `train.csv` and `test.csv` files for training and inference in `examples/atlas_visual_se3_filter.csv`.
  
  ```
  python src/toolbox/processing_atlas/merge_csv.py  --csv atlas.csv  --atlas_dir ./dataset/atlas/ --save_path merged.csv --processed_npz ./dataset/processed_npz --embeddings ./dataset/embeddings --simulation_suffix [simulation params]
  ```
  The merged `.csv` file will be formed as:
  |name|seqres|seq_len|dynamic_npz|embed_path|pdb_path|vel_path|force_path|
  |----|----|----|----|----|----|----|----|
  |1ab1_A|TTCCPSIVA...|46|.npz|.npz|.npz|.pkl|.pkl|
  ...
## 🔥 Training

Follow [Data Preparation](#Data-Preparation) to get data ready, and Update the date `.csv` path in configuration `YAML` files or change it in the training scripts. Start training with the following command:
```shell
cd applications/physics_condition
bash scripts/run_train.sh
```
**Note**: Ensure that `CUDA_VISIBLE_DEVICES numbers`,`nproc_per_node`, `experiment.num_gpus`, and `experiment.batch_size` are set to the same value.



## 🎮 Evaluation
To get the evaluation metrix, run the following command:
```bash
cd applications/physics_condition
bash scripts/run_eval.sh
```

## 📸 Showcase
We present the predicted 3D structures by our method and SE(3)-Trans.  
<table class="center">
  <tr>
    <td style="text-align: center"><b>SE(3) Trans</b></td>
    <td style="text-align: center"><b>Ours</b></td>
    <td style="text-align: center"><b>Ground Truth</b></td>
  </tr>
  <tr>
    <td style="text-align: center"><img src="./assets/qual/SE3-2ERL-1.png" style="width: 80%;"></a></td>
    <td style="text-align: center"><img src="./assets/qual/OURS-2ERL-1.png" style="width: 80%;"></a></td>
    <td style="text-align: center"><img src="./assets/qual/GT-2ERL-1.png" style="width: 80%;"></a></td>
  </tr>
  <tr>
    <td style="text-align: center"><img src="./assets/qual/SE3-3TVJ-9.png" style="width: 80%;"></a></td>
    <td style="text-align: center"><img src="./assets/qual/OURS-3TVJ-9.png" style="width: 80%;"></a></td>
    <td style="text-align: center"><img src="./assets/qual/GT-3TVJ-9.png" style="width: 80%;"></a></td>
  </tr>
</table> 

## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@misc{liu2024dynamicpdbnewdataset,
      title={Dynamic PDB: A New Dataset and a SE(3) Model Extension by Integrating Dynamic Behaviors and Physical Properties in Protein Structures},
      author={Ce Liu and Jun Wang and Zhiqiang Cai and Yingxu Wang and Huizhen Kuang and Kaihui Cheng and Liwei Zhang and Qingkun Su and Yining Tang and Fenglei Cao and Limei Han and Siyu Zhu and Yuan Qi},
      year={2024},
      eprint={2408.12413},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
}
```

## 🤗 Acknowledgements

We would like to thank the contributors to the [openfold](https://github.com/aqlaboratory/openfold), [AlphaFlow](https://github.com/bjing2016/alphaflow), [EigenFold](https://github.com/bjing2016/EigenFold), and [SE3-Diffusion](https://github.com/jasonkyuyim/se3_diffusion) repositories, for their open research and exploration.
If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.
