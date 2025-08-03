<h1 align='center'>4D Diffusion for Dynamic Protein Structure Prediction with Reference Guided Motion Alignment</h1>

<div align='center'>
    <a href='https://github.com/Kaihui-Cheng' target='_blank'>Kaihui Cheng</a><sup>1*</sup>&emsp;
    <a href='https://github.com/cnexah' target='_blank'>Ce Liu</a><sup>1*</sup>&emsp;
    <a href='https://github.com/subazinga' target='_blank'>Qingkun Su</a><sup>2</sup>&emsp;
    <a href='https://github.com/wanggaa' target='_blank'>Jun Wang</a><sup>2</sup>&emsp;
    <a href='https://github.com/AricGamma' target='_blank'>Liwei Zhang</a><sup>2</sup>&emsp;
    <a target='_blank'>Yining Tang</a><sup>1</sup>&emsp;

</div>
<div align='center'>
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>3</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>1,2</sup>&emsp;
    <a target='_blank'>Yuan Qi</a><sup>1,2</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>Fudan University&emsp; <sup>2</sup>SAIS &emsp;  <sup>3</sup>Nanjing University
</div>

<br>
<div align='center'>
    <a href='https://github.com/fudan-generative-vision/dynamicPDB'><img src='https://img.shields.io/github/stars/fudan-generative-vision/dynamicPDB?style=social'></a>
    <a href='https://arxiv.org/abs/2408.12419'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

<br>

## 📸 Showcase



### 🎬 Generated Trajectory

<table class="center">
  <tr>
    <td style="text-align: center"><b>3A1G_B</b></td>
    <td style="text-align: center"><b>6E7E_A</b></td>
    <td style="text-align: center"><b>6H49_A</b></td>
    <td style="text-align: center"><b>6IAH_A</b></td>
  </tr>
  <tr>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/3A1G_B.gif"><img src="./assets/gif/trajectory/3A1G_B.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/6E7E_A.gif"><img src="./assets/gif/trajectory/6E7E_A.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/6H49_A.gif"><img src="./assets/gif/trajectory/6H49_A.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/6IAH_A.gif"><img src="./assets/gif/trajectory/6IAH_A.gif"></a></td>
  </tr>
  <tr>
    <td style="text-align: center"><b>6JV8_A</b></td>
    <td style="text-align: center"><b>6LUS_A</b></td>
    <td style="text-align: center"><b>6Q9C_A</b></td>
    <td style="text-align: center"><b>7A66_B</b></td>
  </tr>
  <tr>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/6JV8_A.gif"><img src="./assets/gif/trajectory/6JV8_A.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/6LUS_A.gif"><img src="./assets/gif/trajectory/6LUS_A.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/6Q9C_A.gif"><img src="./assets/gif/trajectory/6Q9C_A.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/trajectory/7A66_B.gif"><img src="./assets/gif/trajectory/7A66_B.gif"></a></td>
  </tr>
</table>

### 🎬 Prediction
<table class="center">
  <tr>
    <td style="text-align: center"><b> </b></td>
    <td style="text-align: center"><b>4UE8_B</b></td>
    <td style="text-align: center"><b>6D7Y_A</b></td>
    <td style="text-align: center"><b>6GUS_A</b></td>
    <td style="text-align: center"><b>6J56_A</b></td>
  </tr>
  <tr>
    <td style="text-align: center">Prediction</td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/4UE8_B.gif"><img src="./assets/gif/short-term/VS_GT/4UE8_B.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/6D7Y_A.gif"><img src="./assets/gif/short-term/VS_GT/6D7Y_A.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/6GUS_A.gif"><img src="./assets/gif/short-term/VS_GT/6GUS_A.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/6J56_A.gif"><img src="./assets/gif/short-term/VS_GT/6J56_A.gif"></a></td>
  </tr>
  <tr>
    <td style="text-align: center">GT</td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/4UE8_B_GT.gif"><img src="./assets/gif/short-term/VS_GT/4UE8_B_GT.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/6D7Y_A_GT.gif"><img src="./assets/gif/short-term/VS_GT/6D7Y_A_GT.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/6GUS_A_GT.gif"><img src="./assets/gif/short-term/VS_GT/6GUS_A_GT.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/short-term/VS_GT/6J56_A_GT.gif"><img src="./assets/gif/short-term/VS_GT/6J56_A_GT.gif"></a></td>

  </tr>
</table>

### 🎬 Diffusion Process
<table class="center">
<tr>
    <tr>
      <th class="center" colspan="5" style="text-align:center;">6Q9C_A</th>
    </tr>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_99.gif"><img src="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_99.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_10.gif"><img src="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_10.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_2.gif"><img src="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_2.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_1.gif"><img src="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_1.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_0.gif"><img src="./assets/gif/resverse-diff/6Q9C_A_TRAJ_FRAME_0.gif"></a></td>
  </tr>

  <tr>
      <th class="center" colspan="5" style="text-align:center;">7A66_B</th>
    </tr>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_99.gif"><img src="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_99.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_10.gif"><img src="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_10.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_2.gif"><img src="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_2.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_1.gif"><img src="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_1.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_0.gif"><img src="./assets/gif/resverse-diff/7A66_B_TRAJ_FRAME_0.gif"></a></td>
  </tr>
</table>


## 🔧️ Framework

![framework](./assets/fig/pipeline.png)

## ⚙️ Installation

- System requirement: Ubuntu 20.04/Ubuntu 22.04
- Tested GPUs: A100

Create conda environment:

```bash
conda create -n 4d_diffusion python=3.9
conda activate 4d_diffusion
pip install -r requirements.txt
```

## 🗝️️ Usage

Before testing or training your cases, please ensure that the working directory is set in `dynamicPDB`:

1. [Data Preparation](#Data-Preparation).
2. [Inference](#Inference).
3. [Training](#Training).


### 📥 Data Preparation
####  Download and Preprocess datasets
- Download the ATLAS MD trajectory dataset. We provide a [data list](./test_data.csv) as an example.
  ```bash
  bash src/toolbox/download_atlas.sh
  ```

- To preprocess the ATLAS trajectories into `.npz` files, run:
  ```bash
  python src/toolbox/processing_atlas/prep_atlas.py --atlas_dir dataset/atlas --outdir dataset/processed_npz --num_workers [N] --split applications/4d_diffusion/test_data.csv
  ```

####  Extract Sequence Embeddings
- Download the OmegaFold weights and install the modified OmegaFold repository.
  ```bash
  wget https://helixon.s3.amazonaws.com/release1.pt
  git clone https://github.com/bjing2016/OmegaFold
  pip install --no-deps -e OmegaFold
  ```
- Run OmegaFold to make the embeddings:
  ```bash
  python src/toolbox/processing_atlas/extract_embedding.py --reference_only --out_dir_root dataset/embeddings --lm_weights_path release1.pt --data_csv_path applications/4d_diffusion/test_data.csv
  ```
  You can also set `num_workers` and `worker_id` to enable parallelized processing.
  ```bash
  CUDA_VISIBLE_DEVICES=$i python src/toolbox/processing_atlas/extract_embedding.py --reference_only --out_dir_root dataset/embeddings --num_workers 8 --worker_id $i &
  ```

- These datasets should be organized as follows:

  ```text
  ./dataset/
  |-- atlas/
  |   `-- 16pk_A
  |      |--16pk_A.pdb
  |      |--16pk_A_prod_R1.tpr
  |      |--16pk_A_prod_R1_fit.xtc
  |      |--16pk_A_prod_R2.tpr
  |      |--16pk_A_prod_R2_fit.xtc
  |      |--16pk_A_prod_R3.tpr
  |      |--16pk_A_prod_R3_fit.xtc
  |      |--README.txt
  |   |-- 1io1_A
  |   |-- 1qau_A
  |   |-- ...
  |-- embeddings
  |   |-- 16pk_A.npz
  |   |-- 1io1_A.npz
  |   |-- 1qau_A.npz
  |   |-- ...
  |-- processed_npz
  |   |-- 16pk_A.npz
  |   |-- 1io1_A.npz
  |   |-- 1qau_A.npz
  |   |-- ...
  ```
- Consolidate all the information into a `.csv` file.

  ```bash
  python src/toolbox/processing_atlas/merge_csv.py --csv applications/4d_diffusion/test_data.csv --atlas_dir dataset/atlas --save_path applications/4d_diffusion/merged.csv --processed_npz dataset/processed_npz --embeddings dataset/embeddings
  ```

  The merged `.csv` file will be formed as:
  |name|seqres|atlas_npz|embed_path|seq_len|pdb_path|
  |----|----|----|----|----|----|
  |3a1g_B|GGSMERIK...|.npz|.npz|40|.pdb|
  ...

### 🎮 Inference

Run the `scripts/run_eval_extrapolation.sh` and `scripts/run_eval_visual.sh` as follows:

1. Generate Trajectory

To generate a trajectory, run the following command:

```bash
bash applications/4d_diffusion/scripts/run_eval_extrapolation.sh
```

This will produce:

- `extension_npz`: for atom positions.
- `extension_pdb`: for generated trajectory PDB files.

2. Generate Predictions

If you only want to generate predictions, run:

```bash
bash applications/4d_diffusion/scripts/run_eval_visual.sh
```

This will generate:

- `gt`: for ground truth snapshots.
- `sample`: where predictions are stored, including the following files:

  ```
  1. {pdb_name}_aligned.pdb  # prediction aligned with reference
  2. {pdb_name}.pdb           # predicted pdb file
  3. {pdb_name}_first.pdb     # reference pdb file
  4. {pdb_name}_first_motion.pdb  # reference and motion pdb file
  ```

3. Evaluation

To get the evaluation metric, run the following command:

```bash
bash applications/4d_diffusion/scripts/run_eval_metric.sh
```

This will generate:
  - `{pdb_name}_*.pkl`: contains information for multi-time predictions.
  - `{pdb_name}_*.pdb`: multi-time prediction PDB files.
You can visualize these generated `.pdb` files  with [PyMol](https://pymol.org/) or [Protein Viewer Extension](https://marketplace.visualstudio.com/items?itemName=ArianJamasb.protein-viewer).

To get the statistics, run:
```bash
python applications/4d_diffusion/print_metric.py --metric_path [metric save path]
```

4. For more options:

```
option:
- data.frame_time: number of predicted frames per inference.
- data.motion_number: number of motion/kinetic frames per inference.
- data.test_csv_path: csv file storing the paths of embedding and position `.npz` files.
- eval.weights_path: model weights (.ckpt).
- eval.mode: mode of inference:
    0 - Single prediction
    1 - Trajectory
    2 - Batch evaluation
- eval.extrapolation_time: repetition of prediction time for generating the trajectory.
- eval.name: directory name to store this inference.
- experiment.base_root: root path to store this inference.
- data.fix_sample_start: the index in the trajectory where the prediction starts.
- data.eval_start_idx & data.eval_end_idx: define the time range for evaluation, the evaluation time is calculated as time = end - start.
```

### 🔥 Training

Follow [Data Preparation](#Data-Preparation) to get training data ready, and update the data `.csv` path in configuration `YAML` files or change it in the training scripts. Start training with the following command:

```bash
bash applications/4d_diffusion/scripts/run_train.sh
```

**Note**: Ensure that `CUDA_VISIBLE_DEVICES numbers`,`nproc_per_node`, `experiment.num_gpus`, and `experiment.batch_size` are set to the same value.


## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@misc{cheng20244ddiffusiondynamicprotein,
      title={4D Diffusion for Dynamic Protein Structure Prediction with Reference Guided Motion Alignment},
      author={Kaihui Cheng and Ce Liu and Qingkun Su and Jun Wang and Liwei Zhang and Yining Tang and Yao Yao and Siyu Zhu and Yuan Qi},
      year={2024},
      eprint={2408.12419},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.12419},
}

```

## 🤗 Acknowledgements

We would like to thank the contributors to the [openfold](https://github.com/aqlaboratory/openfold), [AlphaFlow](https://github.com/bjing2016/alphaflow), [EigenFold](https://github.com/bjing2016/EigenFold), and [SE3-Diffusion](https://github.com/jasonkyuyim/se3_diffusion) repositories, for their open research and exploration.

If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.

